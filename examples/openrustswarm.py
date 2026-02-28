import openrustswarm_core as ors
from openrustswarm_core import *


class Swarm(ProductionTensorSwarm):
    pass


class Memory(SharedMemoryStore):
    pass


class Runtime(AgentGraphPy):
    def execute(self, task, timeout=10.0):
        return self.spawn_task(task.prompt, HistoryBuffer(), agent_name="DefaultAgent")

    def get_last_decomposition(self):
        return None
