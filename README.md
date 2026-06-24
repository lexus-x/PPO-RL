# PPO-RL

MaskablePPO for solving GIS truck routing problems. Uses action masking to prevent invalid moves and trains with 16 parallel environments.

## What it does

Models a 20-node delivery network with random traffic spikes. The agent learns to plan routes that avoid congestion, beating a greedy nearest-neighbor baseline.

Key design choices:
- **MaskablePPO** from sb3-contrib — boolean masks block previously visited nodes, so the agent never wastes time on invalid actions
- **SubprocVecEnv** with 16 workers — ~3100 fps training throughput
- **Normalized rewards** — distance penalties scaled to [-1, 0] for stable value estimation

## Results

After 2M timesteps:

| Metric | Greedy Baseline | PPO Agent |
|--------|----------------|-----------|
| Invalid actions | N/A | 0% |
| Mean reward | 9.25 | 9.30 |

The agent learns to anticipate traffic patterns and reroute accordingly.

## Usage

```bash
pip install -r requirements.txt
pip install sb3-contrib

# Train
python ppo_routing.py

# Compare against baseline
python compare_methods.py

# Serve via FastAPI (for AnyLogic integration)
python api_server.py
```

API endpoint: `POST http://localhost:8000/get_action`
```json
{
  "current_node": 0,
  "visited": [1, 0, 0, ...],
  "traffic": [1.0, 1.0, ...]
}
```

## Files

- `changwon_env.py` — Gymnasium env with GIS nodes + traffic sim
- `ppo_routing.py` — Training pipeline
- `heuristic_routing.py` — Greedy baseline
- `compare_methods.py` — Benchmarking
- `api_server.py` — FastAPI server
- `ppo_changwon_routing_deep.zip` — Pretrained weights
