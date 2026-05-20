from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sb3_contrib import MaskablePPO

# Load the trained model
print("Loading MaskablePPO RL Agent...")
try:
    model = MaskablePPO.load("ppo_changwon_routing_deep")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model, ensure it is trained first:", e)
    model = None

app = FastAPI(title="AnyLogic PPO Processing Head")

class SimulationState(BaseModel):
    current_node: int
    visited: list[int]
    traffic: list[float]  # flattened traffic matrix

@app.get("/")
def read_root():
    return {"status": "Processing Head is running!"}

@app.post("/get_action")
def get_action(state: SimulationState):
    """
    AnyLogic sends the current simulation state here.
    The MaskablePPO model calculates the best next route and returns it.
    """
    if model is None:
        return {"error": "Model not loaded."}
        
    # Convert incoming data to the observation format the model expects
    obs = np.concatenate([
        [float(state.current_node)],
        state.visited,
        state.traffic
    ]).astype(np.float32)
    
    # Build action mask: unvisited = True (valid), visited = False
    visited_arr = np.array(state.visited, dtype=np.float32)
    action_masks = (visited_arr == 0.0)
    if not action_masks.any():
        action_masks = np.ones(len(state.visited), dtype=bool)
    
    # Predict the best action (next node to visit)
    action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
    
    return {"next_node_action": int(action)}

if __name__ == "__main__":
    import uvicorn
    # Run server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
