# 🚚 PPO-RL: Deep Reinforcement Learning for GIS Truck Routing

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Reinforcement Learning](https://img.shields.io/badge/RL-Stable--Baselines3-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Status](https://img.shields.io/badge/Status-Optimized-brightgreen)

An enterprise-grade Reinforcement Learning pipeline for solving combinatorial GIS truck routing problems using **Maskable Proximal Policy Optimization (PPO)**. 

This repository demonstrates how to successfully apply Deep RL to dynamic routing environments, overcoming common combinatorial traps (like invalid action selection) via Action Masking and rigorous reward shaping. The project includes a fully trained agent, a baseline comparison suite, and a production-ready REST API for integration with simulation software like AnyLogic.

## 🧠 Architecture & Methodology

The environment models a 20-node GIS delivery network with **stochastic traffic spikes**.

*   **Algorithm:** `MaskablePPO` (from `sb3-contrib`). By applying boolean action masking, the agent is physically constrained from selecting previously visited geographic nodes. This eliminates catastrophic invalid-action penalties and accelerates convergence.
*   **Vectorization:** Training runs on a `SubprocVecEnv` with 16 parallel CPU workers, capable of generating experience at over 3,100 frames per second on modern hardware.
*   **Reward Function:** Distance penalties are normalized to a strict `[-1, 0]` scale based on theoretical maximums, stabilizing the value function during gradient updates.

## 🚀 Performance Results

After 2,000,000 timesteps of training, the agent successfully anticipates and navigates around dynamic traffic spikes, outperforming the standard Greedy Nearest Neighbor heuristic.

| Metric | Heuristic Baseline | PPO RL Agent | Improvement |
| :--- | :--- | :--- | :--- |
| **Invalid Action Selection** | N/A | **0%** | Total Elimination |
| **Mean Reward** | 9.25 | **9.30** | **Surpassed Baseline** |
| **Training Speed** | N/A | ~3,100 fps | 4.2x Faster than V1 |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/lexus-x/PPO-RL.git
cd PPO-RL

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install sb3-contrib
```

## 💻 Usage

### 1. Training the Agent
To train a new agent from scratch (default 2M steps, 16 parallel environments):
```bash
python ppo_routing.py
```
*Logs are output to `training.log`.*

### 2. Evaluating the Model
To run a 100-episode head-to-head comparison between the Greedy Heuristic and the trained MaskablePPO agent:
```bash
python compare_methods.py
```

### 3. Running the REST API (AnyLogic Processing Head)
Start the FastAPI server to serve routing predictions to external simulators (e.g., AnyLogic):
```bash
python api_server.py
```
Send a POST request to `http://localhost:8000/get_action` with the simulation state:
```json
{
  "current_node": 0,
  "visited": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "traffic": [1.0, 1.0, ...] 
}
```

## 📁 Repository Structure
*   `changwon_env.py` - Custom Gymnasium environment with GIS nodes, traffic simulation, and action masking.
*   `ppo_routing.py` - Multiprocessed training pipeline utilizing MaskablePPO.
*   `heuristic_routing.py` - Baseline greedy Nearest Neighbor routing algorithm.
*   `compare_methods.py` - Benchmarking script for evaluating RL against heuristics.
*   `api_server.py` - FastAPI application for exposing the trained agent as a microservice.
*   `ppo_changwon_routing_deep.zip` - The pre-trained PPO model weights.
