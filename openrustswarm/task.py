"""
Task data class.

Holds the prompt and configuration options for execution.
"""


class Task:
    """
    A task to be executed by a Runtime or Swarm.
    
    Parameters
    ----------
    prompt : str
        The natural language instruction.
    pipeline : bool
        If True, agents in a Swarm run sequentially, each seeing prior output.
    track_reasoning_steps : bool
        If True, the Runtime records each reasoning step.
    allow_decomposition : bool
        If True, the Runtime may decompose this into subtasks.
    """

    def __init__(
        self,
        prompt: str = "",
        pipeline: bool = False,
        track_reasoning_steps: bool = False,
        allow_decomposition: bool = False,
    ):
        self.prompt = prompt
        self.pipeline = pipeline
        self.track_reasoning_steps = track_reasoning_steps
        self.allow_decomposition = allow_decomposition
