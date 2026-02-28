use super::mmap_pool::MmapSwarmPool;
use super::pheromone::PheromoneField;
use super::grid::SpatialHashGrid;
use std::time::Instant;

/// The master orchestrator for the 100-Million Agent Swarm.
///
/// v3.1.0 Architecture:
/// - Memory-mapped SoA pool (OS-managed paging)
/// - Fibonacci spatial hash grid with real neighbor queries
/// - Pheromone-based stigmergic coordination
/// - Agent-to-agent surprise propagation via spatial neighbors
pub struct SwarmEngineMaster {
    pub pool: MmapSwarmPool,
    pub pheromones: PheromoneField,
    pub grid: SpatialHashGrid,

    // Config
    pub width: f32,
    pub height: f32,
    pub perception_radius: f32,
    pub global_tick: u64,
}

impl SwarmEngineMaster {
    pub fn new(n_agents: usize, width: f32, height: f32) -> Self {
        let mut pool = MmapSwarmPool::new(n_agents);
        pool.randomize_positions(width, height);

        // Scale pheromone field resolution based on agent count
        let pheromone_res = if n_agents >= 10_000_000 { 2048 } else { 1024 };
        let perception = 10.0; // agents perceive neighbors within 10 units

        Self {
            pool,
            pheromones: PheromoneField::new(pheromone_res, pheromone_res, width / pheromone_res as f32),
            grid: SpatialHashGrid::new(
                if n_agents >= 10_000_000 { 1 << 20 } else { 1 << 18 },
                perception, // cell_size = perception_radius for optimal 3x3 query
                [0.0, 0.0],
            ),
            width,
            height,
            perception_radius: perception,
            global_tick: 0,
        }
    }

    /// The master tick function for 100M agents.
    ///
    /// Pipeline:
    /// 1. Spatial locality sorting (amortized every 100 ticks)
    /// 2. Rebuild spatial hash grid
    /// 3. Neighbor-driven physics: cohesion, separation, surprise propagation
    /// 4. Pheromone deposit + diffusion
    /// 5. Health decay
    pub fn tick(&mut self) {
        let start_time = Instant::now();
        self.global_tick += 1;

        // 1. Spatial locality sort (amortized O(N log N) every 100 ticks)
        if self.global_tick % 100 == 0 {
            self.pool.update_spatial_hashes(self.width);
            self.pool.sort_by_spatial_hash();
        }

        // 2. Rebuild spatial hash grid from current positions
        self.rebuild_grid();

        // 3. Real physics: neighbor queries drive agent behavior
        self.run_neighbor_physics();

        // 4. Agents deposit pheromones based on their surprise level
        self.deposit_agent_pheromones();

        // 5. Pheromone field diffusion + decay
        self.pheromones.tick();

        // 6. Health decay
        let health = self.pool.health.as_mut_slice();
        health.iter_mut().for_each(|h| *h *= 0.999);

        let elapsed = start_time.elapsed();
        if self.global_tick % 100 == 0 {
            println!(
                "Tick {}: {}M agents in {:?}",
                self.global_tick,
                self.pool.n_agents / 1_000_000,
                elapsed,
            );
        }
    }

    /// O(N) two-pass rebuild of the Fibonacci spatial hash grid.
    fn rebuild_grid(&mut self) {
        let n = self.pool.n_agents;
        let x = self.pool.x.as_slice();
        let y = self.pool.y.as_slice();

        // Pass 1: count agents per bucket
        self.grid.counts_reset();
        for i in 0..n {
            let (cx, cy) = self.grid.world_to_cell(x[i], y[i]);
            self.grid.count_agent(cx, cy);
        }

        // Prefix sum -> offsets
        self.grid.compute_offsets();

        // Pass 2: scatter agent indices into buckets
        for i in 0..n {
            let (cx, cy) = self.grid.world_to_cell(x[i], y[i]);
            self.grid.scatter_agent(cx, cy, i as u32);
        }
    }

    /// Real neighbor-driven physics.
    ///
    /// For each agent:
    /// 1. Query the spatial hash grid for neighbors within perception_radius
    /// 2. Compute cohesion force (move toward neighbor center-of-mass)
    /// 3. Compute separation force (avoid overlap with nearby agents)
    /// 4. Propagate surprise: if a neighbor has high surprise, absorb some
    /// 5. Follow pheromone trail gradients
    /// 6. Update velocity and position
    fn run_neighbor_physics(&mut self) {
        let n = self.pool.n_agents;
        let width = self.width;
        let height = self.height;
        let r = self.perception_radius;
        let r2 = r * r;

        // We need to read positions + surprise and write to velocity + surprise.
        // To avoid aliasing issues, we compute new velocities and surprise into
        // scratch buffers, then copy back.
        let mut new_vx = vec![0.0f32; n];
        let mut new_vy = vec![0.0f32; n];
        let mut new_surprise = vec![0.0f32; n];

        // Read-only slices for current state
        let x = self.pool.x.as_slice();
        let y = self.pool.y.as_slice();
        let vx = self.pool.vx.as_slice();
        let vy = self.pool.vy.as_slice();
        let surprise = self.pool.surprise.as_slice();

        for i in 0..n {
            let px = x[i];
            let py = y[i];

            // Neighbor aggregation
            let mut neighbor_count = 0u32;
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut sep_x = 0.0f32;
            let mut sep_y = 0.0f32;
            let mut max_neighbor_surprise = 0.0f32;

            // Query the spatial hash grid for real neighbor candidates
            self.grid.query_neighbors(i as u32, px, py, r, |j| {
                let jx = x[j as usize];
                let jy = y[j as usize];

                // Exact distance check (grid is a conservative filter)
                let dx = jx - px;
                let dy = jy - py;
                let d2 = dx * dx + dy * dy;

                if d2 < r2 && d2 > 0.001 {
                    let dist = d2.sqrt();
                    neighbor_count += 1;

                    // Cohesion: accumulate neighbor center of mass
                    sum_x += jx;
                    sum_y += jy;

                    // Separation: repel from close neighbors
                    let inv = 1.0 / dist;
                    sep_x -= dx * inv;
                    sep_y -= dy * inv;

                    // Surprise propagation: absorb neighbor's surprise
                    let js = surprise[j as usize];
                    if js > max_neighbor_surprise {
                        max_neighbor_surprise = js;
                    }
                }
            });

            // Compute steering forces
            let mut fx = vx[i] * 0.9; // momentum
            let mut fy = vy[i] * 0.9;

            if neighbor_count > 0 {
                let nc = neighbor_count as f32;

                // Cohesion: steer toward center of neighbors
                let cohesion_x = (sum_x / nc - px) * 0.01;
                let cohesion_y = (sum_y / nc - py) * 0.01;
                fx += cohesion_x;
                fy += cohesion_y;

                // Separation: push away from overlapping neighbors
                fx += sep_x * 0.05;
                fy += sep_y * 0.05;
            }

            // Pheromone gradient steering
            let (gx_trail, gy_trail) = self.pheromones.gradient(px, py, 2); // Trail
            let (gx_danger, gy_danger) = self.pheromones.gradient(px, py, 1); // Danger
            fx += 0.3 * gx_trail - 0.5 * gx_danger;
            fy += 0.3 * gy_trail - 0.5 * gy_danger;

            // Random exploration
            fx += (rand::random::<f32>() - 0.5) * 0.1;
            fy += (rand::random::<f32>() - 0.5) * 0.1;

            // Clamp velocity
            let mag = (fx * fx + fy * fy).sqrt().max(0.001);
            if mag > 2.0 {
                fx = fx / mag * 2.0;
                fy = fy / mag * 2.0;
            }

            new_vx[i] = fx;
            new_vy[i] = fy;

            // Surprise update: absorb neighbor surprise with decay
            let self_surprise = surprise[i] * 0.95; // natural decay
            new_surprise[i] = self_surprise.max(max_neighbor_surprise * 0.8);
        }

        // Write back computed values
        let out_vx = self.pool.vx.as_mut_slice();
        let out_vy = self.pool.vy.as_mut_slice();
        let out_surprise = self.pool.surprise.as_mut_slice();
        let out_x = self.pool.x.as_mut_slice();
        let out_y = self.pool.y.as_mut_slice();

        for i in 0..n {
            out_vx[i] = new_vx[i];
            out_vy[i] = new_vy[i];
            out_surprise[i] = new_surprise[i];
            out_x[i] = (out_x[i] + new_vx[i]).clamp(0.0, width);
            out_y[i] = (out_y[i] + new_vy[i]).clamp(0.0, height);
        }
    }

    /// Agents deposit pheromones based on their state.
    /// High-surprise agents emit danger signals.
    /// All agents leave trail markers along their path.
    fn deposit_agent_pheromones(&mut self) {
        let x = self.pool.x.as_slice();
        let y = self.pool.y.as_slice();
        let surprise = self.pool.surprise.as_slice();

        // Sample a subset for pheromone deposit (1 in 100 agents)
        // Full 100M deposits would overwhelm the pheromone field
        let stride = 100.max(1);
        for i in (0..self.pool.n_agents).step_by(stride) {
            // Trail marker (channel 2)
            self.pheromones.deposit(x[i], y[i], 2, 0.1);

            // Danger signal if surprised (channel 1)
            if surprise[i] > 0.5 {
                self.pheromones.deposit(x[i], y[i], 1, surprise[i]);
            }
        }
    }
}
