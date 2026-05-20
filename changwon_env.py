import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import random
import os

class ChangwonRoutingEnv(gym.Env):
    def __init__(self, num_nodes=20):
        super().__init__()
        self.num_nodes = num_nodes
        
        self.action_space = spaces.Discrete(self.num_nodes)
        obs_size = 1 + self.num_nodes + (self.num_nodes * self.num_nodes)
        self.observation_space = spaces.Box(low=0, high=self.num_nodes, shape=(obs_size,), dtype=np.float32)
        
        if os.path.exists("data/real_distances.npy"):
            self.base_distances = np.load("data/real_distances.npy")
        else:
            self.base_distances = np.random.uniform(100.0, 5000.0, size=(self.num_nodes, self.num_nodes))
            np.fill_diagonal(self.base_distances, 0)
        
        # Precompute max distance for reward normalization
        self.max_distance = self.base_distances.max() * 5.0  # worst case: max dist * max traffic
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node = 0
        self.visited = np.zeros(self.num_nodes, dtype=np.float32)
        self.visited[0] = 1.0 
        self.traffic = np.ones((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.steps = 0
        self.total_distance = 0.0
        return self._get_obs(), {}
        
    def _get_obs(self):
        obs = np.concatenate([
            [float(self.current_node)],
            self.visited,
            self.traffic.flatten()
        ])
        return obs.astype(np.float32)

    def action_masks(self):
        """Return valid action mask for MaskablePPO.
        True = unvisited (valid), False = visited (invalid).
        If all visited, allow all (episode should terminate anyway)."""
        mask = (self.visited == 0.0)
        if not mask.any():
            mask = np.ones(self.num_nodes, dtype=bool)
        return mask

    def step(self, action):
        self.steps += 1
        
        # Traffic spikes (reduced: 20% chance, 1.5-3x multiplier)
        if random.random() < 0.2:
            u, v = random.randint(0, self.num_nodes-1), random.randint(0, self.num_nodes-1)
            self.traffic[u, v] = random.uniform(1.5, 3.0)
            
        reward = 0.0
        terminated = False
        
        if self.visited[action] == 1.0:
            # Should never happen with action masking, but safety fallback
            reward = -10.0
        else:
            # Distance * Traffic, normalized to [-1, 0]
            dist = self.base_distances[self.current_node, action] * self.traffic[self.current_node, action]
            self.total_distance += dist
            reward = -(dist / self.max_distance)  # normalized to roughly [-1, 0]
            
            self.current_node = action
            self.visited[action] = 1.0
            
        if np.all(self.visited == 1.0):
            reward += 10.0  # completion bonus (proportional to normalized rewards)
            terminated = True
            
        truncated = self.steps >= self.num_nodes * 2
        return self._get_obs(), reward, terminated, truncated, {}
