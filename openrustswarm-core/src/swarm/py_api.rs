use super::master_pipeline::EbbForgeMaster;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
pub struct PyEbbForge {
    engine: EbbForgeMaster,
}

#[pymethods]
impl PyEbbForge {
    #[new]
    #[pyo3(signature = (n_agents=10_000_000, width=1000.0, height=1000.0))]
    pub fn new(n_agents: usize, width: f32, height: f32) -> Self {
        Self {
            engine: EbbForgeMaster::new(n_agents, width, height),
        }
    }

    /// Advance the master simulation by 1 tick (target 138Hz)
    pub fn tick(&mut self) {
        self.engine.tick();
    }

    /// Inject pheromones into the stigmergic grid. 
    /// This is how the Overmind steers the 10M agents mathematically.
    /// Channel 0: Resources, Channel 1: Danger, Channel 4: Novelty
    pub fn deposit_pheromone(&mut self, x: f32, y: f32, channel: usize, amount: f32) {
        self.engine.pheromones.deposit(x, y, channel, amount);
    }

    /// Extract macro-state metrics for the Python Overmind to compute the Genius Score
    pub fn get_macro_state(&self) -> PyObject {
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            
            // Calculate Mean Surprise
            let sum_surprise: f32 = self.engine.pool.surprise.iter().sum();
            let mean_surprise = sum_surprise / self.engine.pool.n_agents as f32;
            
            // Calculate Active Thinkers (Anyone > Tier 0)
            let mut active_thinkers = 0;
            for t in self.engine.pool.tier.iter() {
                if *t > 0 { active_thinkers += 1; }
            }
            
            // Calculate Mean Health
            let sum_health: f32 = self.engine.pool.health.iter().sum();
            let mean_health = sum_health / self.engine.pool.n_agents as f32;

            dict.set_item("mean_surprise", mean_surprise).unwrap();
            dict.set_item("mean_health", mean_health).unwrap();
            dict.set_item("active_thinkers", active_thinkers).unwrap();
            dict.set_item("tick", self.engine.global_tick).unwrap();
            
            dict.into()
        })
    }
}
