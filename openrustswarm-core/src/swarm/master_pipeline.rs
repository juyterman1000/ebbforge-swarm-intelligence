use super::ebbforge_engine::{EbbPool, UnifiedKernel, run_unified_simd_physics};
use super::pheromone::PheromoneField;
use super::grid::SpatialHashGrid;
use std::time::Instant;

/// The master orchestrator for the 10-Million Agent Swarm.
/// Pure CPU SIMD Architecture.
pub struct EbbForgeMaster {
    pub pool: EbbPool,
    pub pheromones: PheromoneField,
    pub physics: UnifiedKernel,
    pub grid: SpatialHashGrid,
    
    // Config
    pub width: f32,
    pub height: f32,
    pub global_tick: u64,
}

impl EbbForgeMaster {
    pub fn new(n_agents: usize, width: f32, height: f32) -> Self {
        Self {
            pool: EbbPool::new(n_agents),
            pheromones: PheromoneField::new(1024, 1024, width / 1024.0),
            physics: UnifiedKernel::default(),
            grid: SpatialHashGrid::new(262144, 10.0, [0.0, 0.0]), // 2^18 buckets
            width,
            height,
            global_tick: 0,
        }
    }

    /// The highly-optimized master tick function representing 1 cycle of time 
    /// for 10 million agents. Target speed: < 7.2ms.
    pub fn tick(&mut self) {
        let start_time = Instant::now();
        self.global_tick += 1;

        // 1. Spatial Locality Sorting (L1/L2 Cache warmup)
        // Memory alignment every 100 ticks for O(N) cache preservation
        if self.global_tick % 100 == 0 {
            self.pool.update_spatial_hashes(self.width);
            self.pool.sort_memory_by_spatial_hash();
        }

        // 2. O(N) Spatial Hash Grid Rebuild 
        // We rebuild the fibonacci hash scatter for neighbor queries
        self.grid.rebuild(&self.pool);

        // 3. CPU AVX2 SIMD Physics
        run_unified_simd_physics(&mut self.pool, &self.physics, &self.pheromones, self.width, self.height);

        // 4. Update the Stigmergic Environment (Pheromone Field)
        self.pheromones.tick();

        // 5. Memory Decay (Ebbinghaus)
        // In this implementation, Health decays naturally while Resource interaction increases it.
        for i in 0..self.pool.n_agents {
             self.pool.health[i] *= 0.999; 
        }

        let elapsed = start_time.elapsed();
        if self.global_tick % 100 == 0 {
            println!("Tick {}: {}M agents processed in {:?}", self.global_tick, self.pool.n_agents / 1_000_000, elapsed);
        }
    }
}
