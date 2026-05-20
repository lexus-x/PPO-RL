from changwon_env import ChangwonRoutingEnv
from heuristic_routing import run_heuristic
from sb3_contrib import MaskablePPO
import numpy as np

def run_ppo_eval(env, model, episodes=10):
    total_rewards = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action_masks = env.action_masks()
            action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
        total_rewards.append(episode_reward)
        
    return np.mean(total_rewards)

if __name__ == "__main__":
    env = ChangwonRoutingEnv(num_nodes=20)
    
    print("--- Real Changwon GIS Truck Routing Comparison ---")
    
    # 1. Evaluate Heuristic
    heuristic_mean = run_heuristic(env, episodes=100)
    print(f"Heuristic (Greedy Nearest Neighbor) Mean Reward: {heuristic_mean:.2f}")
    
    # 2. Evaluate Deep PPO
    try:
        model = MaskablePPO.load("ppo_changwon_routing_deep")
        ppo_mean = run_ppo_eval(env, model, episodes=100)
        print(f"PPO RL Agent Mean Reward: {ppo_mean:.2f}")
        
        print("\n--- Conclusion ---")
        if ppo_mean > heuristic_mean:
            pct_better = ((ppo_mean - heuristic_mean) / abs(heuristic_mean)) * 100
            print(f"PPO RL performs better by {pct_better:.1f}%! It learned to adapt to dynamic traffic.")
        else:
            pct_diff = ((heuristic_mean - ppo_mean) / abs(heuristic_mean)) * 100
            print(f"Heuristic is performing better by {pct_diff:.1f}%. PPO might need more training or tuning.")
    except Exception as e:
        print(f"Error loading PPO model: {e}")
        print("Make sure to run 'python ppo_routing.py' first to train the model!")
