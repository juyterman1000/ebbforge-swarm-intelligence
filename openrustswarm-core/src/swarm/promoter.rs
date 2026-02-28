//! Promotion Logic
//!
//! Logic to "promote" a Light Agent (swarm) to a Heavy Agent (LLM).
//! Triggered by conflict or complexity thresholds.

use super::tensor_engine::TensorSwarm;
use pyo3::prelude::*;
use tracing::info;

/// Logic for promoting agents
#[pyclass]
pub struct PromotionLogic {
    conflict_threshold: f32,
}

#[pymethods]
impl PromotionLogic {
    #[new]
    pub fn new() -> Self {
        PromotionLogic {
            conflict_threshold: 0.5, // 50% max interaction
        }
    }

    /// Check which agents need promotion
    /// In a real system, this would analyze interaction graphs.
    /// Here, we simulate density-based conflict promotion.
    pub fn find_promotion_candidates(
        &self,
        swarm: &TensorSwarm,
        density_map: &super::GridMap,
    ) -> Vec<u32> {
        let mut candidates = Vec::new();

        // Check only a subset for performance
        // (In production, use a dedicated conflict flag vector)
        for i in 0..100.min(swarm.ids.len()) {
            let x = swarm.x[i];
            let y = swarm.y[i];
            let density = density_map.get_density(x, y);

            // If crowded, promote to resolve conflict
            if density > 5 {
                candidates.push(swarm.ids[i]);
            }
        }

        if !candidates.is_empty() {
            info!(
                "🚀 [Promoter] Promoting {} agents to Heavy (LLM) status due to conflict.",
                candidates.len()
            );
        }

        candidates
    }

    /// Inflate a Light Agent struct into a full Agent prompt context
    pub fn inflate_context(&self, agent_id: u32, state: (f32, f32, f32)) -> String {
        format!(
            "You are Agent #{}. Status: Health={:.2}, Pos=({:.1}, {:.1}). You have been promoted to handle a local conflict.",
            agent_id, state.2, state.0, state.1
        )
    }
}
