import asyncio
import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import cogops_core as openrustswarm_internal

app = FastAPI()

# Mount the static frontend
app.mount("/demo", StaticFiles(directory="demo", html=True), name="demo")

class DemoSession:
    def __init__(self):
        self.reset()
        self.running = True

    def reset(self):
        # Initialize 1000 agents in a 1000x1000 grid
        world_config = openrustswarm_internal.WorldModelConfig(ebbinghaus_decay_rate=0.1, grid_size=(1000, 1000))
        self.swarm = cogops.ProductionTensorSwarm(agent_count=1000, world_config=world_config)
        
        self.villages = []
        self.cities = []
        self.ambush_zones = []
        self._update_locations()
        self.tick_count = 0

    def _update_locations(self):
        self.swarm.register_locations(self.villages, [], self.cities, self.ambush_zones)

    def place_village(self, x, y):
        self.villages.append((float(x), float(y)))
        self._update_locations()

    def place_city(self, x, y):
        self.cities.append((float(x), float(y)))
        self._update_locations()

    def place_ambush(self, x, y):
        self.ambush_zones.append((float(x), float(y)))
        self._update_locations()

    def apply_shock(self, x, y, radius=50.0):
        self.swarm.apply_environmental_shock((float(x), float(y)), float(radius), 1.0)

    def get_state(self):
        # Extract raw arrays from the Rust engine
        positions = self.swarm.get_all_positions()
        healths = self.swarm.get_all_health()
        share_probs = self.swarm.get_all_share_probabilities()
        metrics = self.swarm.sample_population_metrics()

        # Format agent data
        agents = []
        for i in range(len(positions)):
            x, y = positions[i]
            sp = share_probs[i]
            
            # Determine caste based on TD-RL share probability
            if sp > 0.7: caste = "broker"
            elif sp < 0.3: caste = "selfish"
            else: caste = "neutral"

            agents.append({
                "x": round(x, 1),
                "y": round(y, 1),
                "h": round(healths[i], 2),
                "c": caste
            })

        return {
            "type": "state",
            "tick": self.tick_count,
            "agents": agents,
            "villages": self.villages,
            "cities": self.cities,
            "ambushes": self.ambush_zones,
            "mean_health": round(sum(healths)/len(healths), 3) if healths else 0,
            "mean_surprise": round(metrics.get("mean_surprise_score", 0), 3)
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = DemoSession()
    
    # Task to read commands from client
    async def receive_commands():
        try:
            while True:
                data = await websocket.receive_text()
                cmd = json.loads(data)
                action = cmd.get("action")
                payload = cmd.get("payload", {})
                
                if action == "reset":
                    session.reset()
                elif action == "pause":
                    session.running = False
                elif action == "resume":
                    session.running = True
                elif action == "place_village":
                    session.place_village(payload["x"], payload["y"])
                elif action == "place_city":
                    session.place_city(payload["x"], payload["y"])
                elif action == "place_ambush":
                    session.place_ambush(payload["x"], payload["y"])
                elif action == "shock":
                    session.apply_shock(payload["x"], payload["y"], payload.get("radius", 50.0))
        except WebSocketDisconnect:
            pass

    receiver_task = asyncio.create_task(receive_commands())

    try:
        while True:
            if session.running:
                session.swarm.tick()
                session.tick_count += 1
                
                # Send state snapshot at 20Hz
                state = session.get_state()
                await websocket.send_text(json.dumps(state))
                
                await asyncio.sleep(0.05) # ~20 FPS
            else:
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        receiver_task.cancel()

if __name__ == "__main__":
    print("Starting CogOps Demo Server on http://0.0.0.0:8080/demo/")
    uvicorn.run("demo_server:app", host="0.0.0.0", port=8080, workers=1)
