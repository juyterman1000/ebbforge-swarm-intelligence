import openrustswarm as openrustswarm

# Initialize 1,000 agents
swarm = cogops.ProductionTensorSwarm(agent_count=1000)

# Register strategic locations
swarm.register_locations(
    villages=[(100, 200), (400, 500)],
    towns=[],
    cities=[(800, 800)],
    ambush_zones=[(300, 350)]
)

# Run 100 simulation ticks
print("Running 100 ticks...")
for i in range(100):
    swarm.tick()
    if i % 20 == 0:
        print(f"  Tick {i} complete")

# Verify health output
health = swarm.get_all_health()
mean_health = sum(health)/len(health)
print(f"\nFinal Statistics:")
print(f"  Agent Count: {len(health)}")
print(f"  Mean Health: {mean_health:.3f}")

if 0 < mean_health < 1:
    print("\n✅ Quick Start Snippet: PASSED")
else:
    print("\n❌ Quick Start Snippet: FAILED (Invalid Health Data)")
