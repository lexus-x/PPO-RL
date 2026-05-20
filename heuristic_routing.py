from changwon_env import ChangwonRoutingEnv
import numpy as np

def run_heuristic(env, episodes=10):
    """
    Heuristic: Always pick the nearest unvisited neighbor (Greedy nearest neighbor).
    This represents a static routing approach that doesn't anticipate future traffic.
    """
    total_rewards = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            current_node = int(obs[0])
            visited = obs[1:env.num_nodes+1]
            
            # Extract traffic and distances
            traffic = obs[env.num_nodes+1:].reshape(env.num_nodes, env.num_nodes)
            distances = env.base_distances * traffic
            
            # Find nearest unvisited node
            best_dist = float('inf')
            best_action = -1
            
            for i in range(env.num_nodes):
                if visited[i] == 0: # Unvisited
                    dist = distances[current_node, i]
                    if dist < best_dist:
                        best_dist = dist
                        best_action = i
                        
            if best_action == -1:
                break
                
            obs, reward, done, truncated, _ = env.step(best_action)
            episode_reward += reward
            
        total_rewards.append(episode_reward)
        
    return np.mean(total_rewards)

if __name__ == "__main__":
    env = ChangwonRoutingEnv(num_nodes=20)
    mean_reward = run_heuristic(env, episodes=100)
    print(f"Heuristic Mean Reward over 100 episodes: {mean_reward:.2f}")
