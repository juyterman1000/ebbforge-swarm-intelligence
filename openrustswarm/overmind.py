import time
import math
from typing import Dict, Any

# We will import the compiled Rust Engine (PyEbbForge)
# import cogops_core as cogops

class Overmind:
    """
    The Master Python Loop governing the 10M-Agent EbbForge Engine.
    Executes the MPPI strategic plan and monitors the Genius Score.
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.running = False
        
    def compute_genius_score(self, macro_state: Dict[str, Any]) -> float:
        """
        Calculates the 5-variable Genius Score from TEN_MILLION_GENIUSES.md
        (Efficiency, Retention, Coherence, Adaptability, Alignment)
        For this MVP, we use proxies derived from the macro states.
        """
        surprise = macro_state.get("mean_surprise", 0.0)
        health = macro_state.get("mean_health", 0.0)
        active_thinkers = macro_state.get("active_thinkers", 0)
        
        # 1. Efficiency: Are active thinkers working on anomalies?
        # High surprise should correlate with high active thinkers.
        efficiency = min(1.0, active_thinkers / 500_000.0) if surprise > 2.0 else 1.0
        
        # 2. Alignment: Is the swarm healthy?
        alignment = health
        
        # 3. Adaptability: Is the swarm solving anomalies? 
        # (Surprise should decay rapidly once Tier 4 agents process it).
        adaptability = 1.0 / (1.0 + surprise)
        
        # Calculate final weighted Genius Score
        score = (0.4 * efficiency) + (0.4 * alignment) + (0.2 * adaptability)
        return score

    def step_mppi_planner(self, macro_state: Dict[str, Any]):
        """
        Model Predictive Path Integral (MPPI) Planner.
        Given the current macro state, predict if the swarm needs steering.
        Steering is applied by depositing chemicals in the Pheromone Field.
        """
        score = self.compute_genius_score(macro_state)
        tick = macro_state.get("tick", 0)
        active = macro_state.get("active_thinkers", 0)
        
        # Simple Overmind intervention rule:
        # If Genius Score drops below 0.6, inject a Novelty Beacon (CH 4) to force agents to scatter and find new solutions.
        if score < 0.6 and tick % 10 == 0:
            import random
            x = random.uniform(100.0, 900.0)
            y = random.uniform(100.0, 900.0)
            
            # CH 4 = Novelty Beacon
            self.engine.deposit_pheromone(x, y, 4, 100.0)
            print(f"[OVERMIND - Tick {tick}] Intervention! Genius Score {score:.2f} too low. Injecting Novelty Beacon at ({x:.1f}, {y:.1f}).")
            
        elif tick % 100 == 0:
            print(f"[OVERMIND - Tick {tick}] Swarm humming. Genius Score: {score:.2f} | Active Thinkers GPU: {active}")

    def run(self):
        """Standard 138Hz async loop pulling from the Rust backend."""
        print("Initializing EbbForge Overmind 10M...")
        self.running = True
        
        while self.running:
            # Step the blazing fast Rust CPU/GPU engine
            self.engine.tick()
            
            # Extract aggregated tensors and compute Python-side Autonomy
            state = self.engine.get_macro_state()
            self.step_mppi_planner(state)
            
            # Give CPU a microscopic breather
            time.sleep(0.005)

    def stop(self):
        self.running = False
