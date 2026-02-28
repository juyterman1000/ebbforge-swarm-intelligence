//! Criticality analysis: Does surprise cascade, stabilize, or collapse?
//!
//! Run: cargo test --release criticality_analysis -- --nocapture --ignored

#[cfg(test)]
mod tests {
    use crate::swarm::master_pipeline::SwarmEngineMaster;
    use std::time::Instant;

    /// Extended 30-tick surprise propagation analysis.
    ///
    /// Injects a surprise event at the center, then tracks:
    /// - Total surprised agents per tick
    /// - Delta (new infections per tick)
    /// - Growth rate (delta_n / delta_n-1)
    /// - Spatial spread: mean distance of surprised agents from origin
    /// - Peak surprise value (is the wave front weakening?)
    #[test]
    #[ignore]
    fn criticality_analysis() {
        let sep = "=".repeat(90);
        println!("\n{}", sep);
        println!("  CRITICALITY ANALYSIS — Surprise Propagation Dynamics");
        println!("  1M agents, injection at (500,500), radius 50");
        println!("{}\n", sep);

        // Initialize
        let mut engine = SwarmEngineMaster::new(1_000_000, 1000.0, 1000.0);

        // Inject surprise at center
        let origin = (500.0f32, 500.0f32);
        let inject_radius = 50.0f32;
        {
            let x = engine.pool.x.as_slice();
            let y = engine.pool.y.as_slice();
            let surprise = engine.pool.surprise.as_mut_slice();
            for i in 0..engine.pool.n_agents {
                let dx = x[i] - origin.0;
                let dy = y[i] - origin.1;
                if dx * dx + dy * dy < inject_radius * inject_radius {
                    surprise[i] = 1.0;
                }
            }
        }

        // Collect per-tick metrics
        println!("{:<6} | {:<10} | {:<8} | {:<8} | {:<8} | {:<10} | {:<10} | {}",
            "Tick", "Surprised", "Delta", "Growth", "% Pop", "Mean Dist", "Peak S", "Time");
        println!("{}", "-".repeat(90));

        let mut prev_count = 0u64;
        let mut prev_delta = 0i64;

        for tick in 0..30 {
            let t = Instant::now();

            // Snapshot BEFORE tick
            let (count, mean_dist, peak_s, spatial_std) = measure_surprise_state(&engine, origin);

            if tick == 0 {
                prev_count = count;
                println!("{:<6} | {:<10} | {:<8} | {:<8} | {:<8} | {:<10} | {:<10} | {}",
                    "init", count, "-", "-",
                    format!("{:.2}%", count as f64 / 10000.0),
                    format!("{:.1}", mean_dist),
                    format!("{:.3}", peak_s),
                    "-");
            }

            // Run one tick
            engine.tick();
            let elapsed = t.elapsed();

            // Snapshot AFTER tick
            let (new_count, new_mean_dist, new_peak_s, new_spatial_std) = measure_surprise_state(&engine, origin);

            let delta = new_count as i64 - prev_count as i64;
            let growth = if prev_delta != 0 {
                format!("{:.3}", delta as f64 / prev_delta as f64)
            } else {
                "-".to_string()
            };

            println!("{:<6} | {:<10} | {:<8} | {:<8} | {:<8} | {:<10} | {:<10} | {:.1}s",
                tick + 1,
                new_count,
                format!("{:+}", delta),
                growth,
                format!("{:.2}%", new_count as f64 / 10000.0),
                format!("{:.1}", new_mean_dist),
                format!("{:.4}", new_peak_s),
                elapsed.as_secs_f64());

            prev_delta = delta;
            prev_count = new_count;
        }

        // Final analysis
        let (final_count, final_mean_dist, final_peak, _) = measure_surprise_state(&engine, origin);

        println!("\n{}", sep);
        println!("  ANALYSIS");
        println!("{}", sep);

        let final_pct = final_count as f64 / 10000.0;
        if final_pct > 50.0 {
            println!("  VERDICT: SUPERCRITICAL CASCADE");
            println!("  Surprise infected {:.1}% of population — runaway propagation", final_pct);
        } else if final_count > 0 && final_peak > 0.05 {
            println!("  VERDICT: CRITICAL / STEADY STATE");
            println!("  Surprise stabilized at {:.1}% with peak {:.4}", final_pct, final_peak);
        } else {
            println!("  VERDICT: SUBCRITICAL COLLAPSE");
            println!("  Surprise wave died out — decay (0.95) overwhelms propagation (0.8)");
        }

        println!("  Final mean distance from origin: {:.1} units", final_mean_dist);
        println!("  Peak surprise remaining: {:.4}", final_peak);
        println!("{}\n", sep);
    }

    /// Measure surprise state: count, mean distance from origin, peak value, spatial std
    fn measure_surprise_state(engine: &SwarmEngineMaster, origin: (f32, f32)) -> (u64, f64, f32, f64) {
        let x = engine.pool.x.as_slice();
        let y = engine.pool.y.as_slice();
        let surprise = engine.pool.surprise.as_slice();

        let threshold = 0.1;
        let mut count = 0u64;
        let mut sum_dist = 0.0f64;
        let mut peak = 0.0f32;

        for i in 0..engine.pool.n_agents {
            if surprise[i] > threshold {
                count += 1;
                let dx = (x[i] - origin.0) as f64;
                let dy = (y[i] - origin.1) as f64;
                sum_dist += (dx * dx + dy * dy).sqrt();
                if surprise[i] > peak {
                    peak = surprise[i];
                }
            }
        }

        let mean_dist = if count > 0 { sum_dist / count as f64 } else { 0.0 };

        // Spatial std dev
        let mut sum_sq = 0.0f64;
        if count > 0 {
            for i in 0..engine.pool.n_agents {
                if surprise[i] > threshold {
                    let dx = (x[i] - origin.0) as f64;
                    let dy = (y[i] - origin.1) as f64;
                    let d = (dx * dx + dy * dy).sqrt();
                    sum_sq += (d - mean_dist) * (d - mean_dist);
                }
            }
        }
        let std_dev = if count > 1 { (sum_sq / (count - 1) as f64).sqrt() } else { 0.0 };

        (count, mean_dist, peak, std_dev)
    }
}
