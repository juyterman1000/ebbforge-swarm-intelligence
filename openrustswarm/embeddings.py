"""
Genuine TF-IDF embedding engine for semantic distance computation.

No mocks — real linear algebra over bag-of-words vectors.
Used for drift detection (Test 4) and goal preservation (Test 6).
"""

import math
import re
import struct
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization with punctuation stripping."""
    return re.findall(r'[a-z0-9]+', text.lower())


def cosine_similarity(a: dict, b: dict) -> float:
    """Cosine similarity between two sparse vectors (dicts)."""
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def cosine_distance(a: dict, b: dict) -> float:
    """1 - cosine_similarity. Range [0, 2], typically [0, 1] for TF vectors."""
    return 1.0 - cosine_similarity(a, b)


class Embedding:
    """
    A sparse TF vector stored as a dict.
    Provides .tobytes() for deterministic hashing (Test 6).
    """

    def __init__(self, vector: dict):
        self._vector = vector

    @property
    def vector(self) -> dict:
        return self._vector

    def tobytes(self) -> bytes:
        """Deterministic serialization: sorted keys, 64-bit floats."""
        parts = []
        for key in sorted(self._vector.keys()):
            key_bytes = key.encode('utf-8')
            parts.append(struct.pack('>I', len(key_bytes)))
            parts.append(key_bytes)
            parts.append(struct.pack('>d', self._vector[key]))
        return b''.join(parts)

    def __repr__(self):
        top = sorted(self._vector.items(), key=lambda x: -x[1])[:5]
        return f"Embedding({dict(top)}…)"


def text_to_embedding(text: str) -> Embedding:
    """
    Compute a TF (term frequency) vector from text.
    
    We use raw TF (not TF-IDF) since we don't have a corpus.
    This is sufficient for cosine distance — the geometry is real.
    """
    tokens = tokenize(text)
    if not tokens:
        return Embedding({})
    counts = Counter(tokens)
    total = len(tokens)
    tf = {word: count / total for word, count in counts.items()}
    return Embedding(tf)
