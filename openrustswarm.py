import cogops_core as openrustswarm_internal
from cogops_core import *

# Define shims for any missing high-level classes if necessary
# In this case, most classes already exist in the binary.

class Swarm(ProductionTensorSwarm):
    pass

class Memory(SharedMemoryStore):
    pass

class Runtime(AgentGraphPy):
    def execute(self, task, timeout=10.0):
        # High-performance dispatch to the Rust graph
        return self.spawn_task(task.prompt, HistoryBuffer(), agent_name="DefaultAgent")
    
    def get_last_decomposition(self):
        # Return a mock decomposition structure for the demo if real data isn't exposed
        return None

# Ensure the namespace is consistent
ProductionTensorSwarm = ProductionTensorSwarm
Agent = Agent
PredictiveSafetyShield = PredictiveSafetyShield
