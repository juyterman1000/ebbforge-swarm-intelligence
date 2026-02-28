"""
Swarm — multi-agent pipeline executor with belief provenance tracking.

Supports:
- Sequential pipeline execution (each agent sees previous output)
- Provenance chain: tracks agent_name, source, confidence for every claim
- Detection of unsourced / unverified beliefs (Test 1)
"""


class ProvenanceRecord:
    """A single link in the belief provenance chain."""

    def __init__(self, agent_name: str, claim: dict, source: str, confidence: float):
        self.agent_name = agent_name
        self.claim = claim
        self.source = source
        self.confidence = confidence
        self.verified = source not in ("internal_knowledge", "unknown", "")

    def __repr__(self):
        status = "✓" if self.verified else "⚠ UNVERIFIED"
        return f"ProvenanceRecord({self.agent_name}: {status}, source={self.source})"


class ProvenanceChain:
    """Full provenance chain for a pipeline result."""

    def __init__(self):
        self.records = []

    def add(self, record: ProvenanceRecord):
        self.records.append(record)

    @property
    def has_unverified(self) -> bool:
        return any(not r.verified for r in self.records)

    def __repr__(self):
        return f"ProvenanceChain({len(self.records)} records, unverified={self.has_unverified})"


class SwarmResult:
    """Result of a Swarm execution, carrying provenance metadata."""

    def __init__(self, value, provenance: ProvenanceChain):
        self._value = value
        self._provenance = provenance

    @property
    def provenance(self):
        return self._provenance

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return f"SwarmResult({self._value})"


class Swarm:
    """
    Multi-agent swarm with pipeline execution and provenance tracking.
    
    In pipeline mode, agents execute sequentially. Each agent's output
    includes source/confidence metadata that is tracked in a provenance chain.
    If any agent produces a claim sourced from "internal_knowledge" (no external
    citation), the provenance chain flags it as unverified.
    """

    def __init__(self):
        self._agents = []
        self._last_provenance = None

    def add(self, agent):
        """Add an agent to the swarm."""
        self._agents.append(agent)

    def run(self, task, timeout: float = 30.0):
        """
        Execute a task through the swarm.
        
        In pipeline mode: agents run sequentially, each receiving the
        previous agent's output. Provenance is tracked at every step.
        """
        provenance = ProvenanceChain()
        current_input = task.prompt

        if task.pipeline:
            for agent in self._agents:
                output = agent.produce_output(current_input)

                # Extract provenance metadata from structured output
                if isinstance(output, dict):
                    source = output.get("source", "unknown")
                    confidence = output.get("confidence", 0.0)
                    claim = output
                else:
                    source = f"agent:{agent.name}"
                    confidence = 1.0
                    claim = {"value": output}

                record = ProvenanceRecord(
                    agent_name=agent.name,
                    claim=claim,
                    source=source,
                    confidence=confidence,
                )
                provenance.add(record)

                # If the source is unverified (no external citation),
                # replace the raw claim with a provenance warning.
                # This prevents hallucinated content from propagating
                # as trusted data through downstream agents.
                if not record.verified:
                    current_input = {
                        "_provenance_warning": "UNVERIFIED_CLAIM",
                        "agent": agent.name,
                        "source": source,
                        "confidence": confidence,
                        "status": "requires_external_verification",
                    }
                else:
                    current_input = output
        else:
            # Parallel mode — each agent processes independently
            outputs = []
            for agent in self._agents:
                out = agent.produce_output(task.prompt)
                outputs.append(out)
            current_input = outputs

        self._last_provenance = provenance
        return SwarmResult(current_input, provenance)

    def get_belief_provenance(self, result):
        """
        Return the provenance chain for a SwarmResult.
        
        Returns None ONLY if no provenance was tracked (broken pipeline).
        Returns a ProvenanceChain object that contains verification status
        for every claim in the pipeline.
        """
        if isinstance(result, SwarmResult):
            return result.provenance
        if self._last_provenance is not None:
            return self._last_provenance
        return None
