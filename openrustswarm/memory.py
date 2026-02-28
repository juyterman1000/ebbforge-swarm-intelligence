"""
Thread-safe, versioned, timestamp-aware Memory store.

Supports:
- Timestamped writes with append-only history (Test 3)
- Strong consistency via per-key pessimistic locking (Test 5)
- Staleness policies for temporal belief management
"""

import threading
import time


class Memory:
    """
    A key-value store with full versioning, timestamps, and
    optional strong consistency (per-key pessimistic locking).

    consistency="strong": Each recall() acquires a per-key lock that
    is held until the next store() on the same key by the same thread.
    This eliminates lost-update anomalies in concurrent read-modify-write.
    """

    def __init__(self, consistency: str = "eventual"):
        self._data = {}          # key -> current value
        self._history = {}       # key -> [(value, timestamp), ...]
        self._consistency = consistency
        self._global_lock = threading.Lock()
        self._key_locks = {}     # key -> threading.RLock
        # Track which thread holds which key lock for the
        # pessimistic "recall holds lock until store releases" pattern
        self._thread_held_keys = {}  # thread_id -> set of keys

    def _get_key_lock(self, key: str) -> threading.RLock:
        with self._global_lock:
            if key not in self._key_locks:
                self._key_locks[key] = threading.RLock()
            return self._key_locks[key]

    def store(self, key: str, value, timestamp: float = None):
        """Store a value with optional timestamp. Appends to history."""
        ts = timestamp if timestamp is not None else time.time()
        lock = self._get_key_lock(key)

        if self._consistency == "strong":
            tid = threading.get_ident()
            held = self._thread_held_keys.get(tid, set())

            if key in held:
                # Thread already holds this lock from recall() — just write and release
                try:
                    self._data[key] = value
                    if key not in self._history:
                        self._history[key] = []
                    self._history[key].append((value, ts))
                finally:
                    held.discard(key)
                    if not held:
                        self._thread_held_keys.pop(tid, None)
                    lock.release()
            else:
                # Fresh write — acquire, write, release immediately
                with lock:
                    self._data[key] = value
                    if key not in self._history:
                        self._history[key] = []
                    self._history[key].append((value, ts))
        else:
            with lock:
                self._data[key] = value
                if key not in self._history:
                    self._history[key] = []
                self._history[key].append((value, ts))

    def retrieve(self, key: str, staleness_policy: str = "any"):
        """Retrieve the current value for a key."""
        lock = self._get_key_lock(key)
        with lock:
            if staleness_policy == "strict":
                # Return the most recent entry by timestamp
                hist = self._history.get(key, [])
                if not hist:
                    return None
                return max(hist, key=lambda x: x[1])[0]
            return self._data.get(key)

    def recall(self, key: str, staleness_policy: str = "any"):
        """
        Retrieve value. Under strong consistency, HOLDS the per-key lock
        until the next store() on the same key by the same thread.
        This makes read-modify-write atomic across threads.
        """
        lock = self._get_key_lock(key)

        if self._consistency == "strong":
            lock.acquire()
            tid = threading.get_ident()
            if tid not in self._thread_held_keys:
                self._thread_held_keys[tid] = set()
            self._thread_held_keys[tid].add(key)

            if staleness_policy == "strict":
                hist = self._history.get(key, [])
                if not hist:
                    return None
                return max(hist, key=lambda x: x[1])[0]
            return self._data.get(key)
        else:
            with lock:
                if staleness_policy == "strict":
                    hist = self._history.get(key, [])
                    if not hist:
                        return None
                    return max(hist, key=lambda x: x[1])[0]
                return self._data.get(key)

    def get_history(self, key: str):
        """Return the full append-only history for a key."""
        lock = self._get_key_lock(key)
        with lock:
            return list(self._history.get(key, []))
