import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import random
import json
import os
import numpy as np
from sb3_contrib import MaskablePPO
from changwon_env import ChangwonRoutingEnv

st.set_page_config(page_title="PPO GIS Routing", layout="wide")
st.title("🚚 PPO vs Traffic: Live Changwon Routing")

@st.cache_data
def get_node_coordinates(num_nodes=20):
    coords_file = "data/node_coords.json"
    if os.path.exists(coords_file):
        with open(coords_file, "r") as f:
            return json.load(f)
            
    st.info("Downloading Changwon GIS network to extract coordinates... (This only happens once)")
    G = ox.graph_from_place("Changwon, South Korea", network_type="drive")
    nodes = list(G.nodes)
    
    # Must use same seed as fetch_gis_data.py to match real_distances.npy
    random.seed(42)
    hotspots = random.sample(nodes, num_nodes)
    
    coords = []
    for node_id in hotspots:
        node_data = G.nodes[node_id]
        coords.append({"lat": node_data['y'], "lon": node_data['x']})
        
    with open(coords_file, "w") as f:
        json.dump(coords, f)
        
    return coords

@st.cache_resource
def load_agent():
    return MaskablePPO.load("models/ppo_changwon_routing_deep")

coords = get_node_coordinates(20)
agent = load_agent()

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Controls")
    run_sim = st.button("▶️ Run PPO Simulation", use_container_width=True)
    
    if run_sim:
        st.success("Simulation complete! Route mapped.")

# Base map centered on Changwon
center_lat = sum([c['lat'] for c in coords]) / len(coords)
center_lon = sum([c['lon'] for c in coords]) / len(coords)
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter")

# Add markers for nodes
for i, coord in enumerate(coords):
    folium.CircleMarker(
        location=[coord['lat'], coord['lon']],
        radius=6 if i == 0 else 4,
        color='green' if i == 0 else 'blue',
        fill=True,
        popup=f"Node {i}"
    ).add_to(m)

if run_sim:
    env = ChangwonRoutingEnv(num_nodes=20)
    
    with st.spinner("Running 100 episodes for comparison..."):
        from heuristic_routing import run_heuristic
        heuristic_mean = run_heuristic(env, episodes=100)
        
        total_ppo_rewards = []
        for _ in range(100):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            while not done:
                action_masks = env.action_masks()
                action, _ = agent.predict(obs, deterministic=True, action_masks=action_masks)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            total_ppo_rewards.append(ep_reward)
        ppo_mean = np.mean(total_ppo_rewards)
        
    pct_diff = ((ppo_mean - heuristic_mean) / abs(heuristic_mean)) * 100

    st.sidebar.markdown("### 🏆 Performance (100 Episodes)")
    st.sidebar.metric("PPO Agent Mean", f"{ppo_mean:.2f}", f"{pct_diff:.2f}% vs Baseline")
    st.sidebar.metric("Greedy Heuristic Mean", f"{heuristic_mean:.2f}")
    st.sidebar.markdown("---")
    
    # Run one specific episode to map
    obs, _ = env.reset(seed=42)
    done = False
    route_sequence = [0]
    total_reward = 0
    while not done:
        action_masks = env.action_masks()
        action, _ = agent.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, terminated, truncated, _ = env.step(action)
        route_sequence.append(int(action))
        total_reward += reward
        done = terminated or truncated
        
    st.sidebar.markdown("### 🗺️ Mapped Route Demo")
    st.sidebar.metric("Route Nodes Visited", len(route_sequence))
    st.sidebar.metric("Demo Route Reward", f"{total_reward:.2f}")
    
    # Draw route
    route_coords = [[coords[node]['lat'], coords[node]['lon']] for node in route_sequence]
    
    # Draw path
    folium.PolyLine(
        route_coords,
        weight=3,
        color='cyan',
        opacity=0.8
    ).add_to(m)
    
    # Add animated ant path if folium plugins are available
    from folium import plugins
    plugins.AntPath(locations=route_coords, color="cyan", weight=5, delay=800).add_to(m)

with col2:
    st_folium(m, width=1000, height=700)
