import osmnx as ox
import networkx as nx
import numpy as np
import random

def fetch_and_save_distances(num_nodes=20):
    print("Downloading real Changwon road network...")
    # Get the drive network for Changwon
    G = ox.graph_from_place("Changwon, South Korea", network_type="drive")
    
    print("Network downloaded. Nodes:", len(G.nodes))
    
    nodes = list(G.nodes)
    random.seed(42)
    hotspots = random.sample(nodes, num_nodes)
    
    print(f"Calculating real driving distances between {num_nodes} hotspots...")
    dist_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                try:
                    # shortest path length in meters
                    path_len = nx.shortest_path_length(G, source=hotspots[i], target=hotspots[j], weight='length')
                    dist_matrix[i, j] = path_len
                except nx.NetworkXNoPath:
                    dist_matrix[i, j] = 999999.0
                    
    np.save("real_distances.npy", dist_matrix)
    print("Real distance matrix saved to real_distances.npy")

if __name__ == "__main__":
    fetch_and_save_distances(num_nodes=20)
