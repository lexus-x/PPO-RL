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

# 1. Page Configuration & Custom CSS Injection
st.set_page_config(
    page_title="PPO Truck Routing Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Cyber-Tech Custom Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Main Page Container Overrides */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #0b0c10 !important;
    font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif !important;
    color: #c5c6c7 !important;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #1f2833 !important;
    border-right: 1px solid rgba(102, 252, 241, 0.1) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #c5c6c7 !important;
}

/* Title Gradient Styling */
h1 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #66fcf1 0%, #45f3ff 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-shadow: 0 0 35px rgba(102, 252, 241, 0.15);
    font-size: 2.6rem !important;
    margin-bottom: 0.2rem !important;
    letter-spacing: -0.8px !important;
}

.dashboard-desc {
    font-size: 1.05rem;
    color: #8b949e;
    margin-bottom: 2rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 400;
}

h3 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: #66fcf1 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.75rem !important;
    border-bottom: 1px solid rgba(102, 252, 241, 0.1);
    padding-bottom: 0.4rem;
}

/* Sidebar status indicator */
.sidebar-status {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(11, 12, 16, 0.85);
    border: 1.5px solid rgba(102, 252, 241, 0.2);
    padding: 0.9rem 1.1rem;
    border-radius: 12px;
    margin-bottom: 1.8rem;
    font-size: 0.88rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.pulse-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: #66fcf1;
    box-shadow: 0 0 0 0 rgba(102, 252, 241, 0.7);
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(102, 252, 241, 0.7);
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 8px rgba(102, 252, 241, 0);
    }
    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(102, 252, 241, 0);
    }
}

/* Customized Premium Button */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%) !important;
    color: #66fcf1 !important;
    border: 1.5px solid #66fcf1 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.4rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #66fcf1 0%, #45f3ff 100%) !important;
    color: #0b0c10 !important;
    border-color: transparent !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 25px rgba(102, 252, 241, 0.45) !important;
}

div[data-testid="stButton"] button:active {
    transform: translateY(1px) !important;
}

/* Custom KPI card container */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2rem;
    width: 100%;
}

.kpi-card {
    background: rgba(31, 40, 51, 0.5);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    position: relative;
    overflow: hidden;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

.kpi-card:hover {
    transform: translateY(-5px);
    border-color: rgba(102, 252, 241, 0.35);
    box-shadow: 0 12px 30px rgba(102, 252, 241, 0.18);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: var(--card-gradient, linear-gradient(90deg, #66fcf1, #45f3ff));
}

.kpi-card.ppo::before {
    --card-gradient: linear-gradient(90deg, #66fcf1, #4facfe);
}

.kpi-card.heuristic::before {
    --card-gradient: linear-gradient(90deg, #8b949e, #c5c6c7);
}

.kpi-card.advantage::before {
    --card-gradient: linear-gradient(90deg, #00ff87, #60efff);
}

.kpi-card.demo::before {
    --card-gradient: linear-gradient(90deg, #a18cd1, #fbc2eb);
}

.kpi-title {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #8b949e;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.kpi-value {
    font-size: 1.95rem;
    font-weight: 800;
    font-family: 'Outfit', sans-serif;
    color: #ffffff;
    margin-bottom: 0.2rem;
}

.kpi-subtitle {
    font-size: 0.78rem;
    font-weight: 500;
    color: #66fcf1;
}

.kpi-card.heuristic .kpi-subtitle {
    color: #8b949e;
}

.kpi-card.advantage .kpi-subtitle {
    color: #00ff87;
}

.kpi-card.demo .kpi-subtitle {
    color: #a18cd1;
}

/* Styled Alert notifications */
div[data-testid="stAlert"] {
    background-color: rgba(31, 40, 51, 0.75) !important;
    border: 1px solid rgba(102, 252, 241, 0.2) !important;
    color: #c5c6c7 !important;
    border-radius: 12px !important;
}

/* Map frame glow styling for iframes */
iframe {
    border-radius: 16px !important;
    border: 1px solid rgba(102, 252, 241, 0.15) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45) !important;
    transition: all 0.3s ease !important;
}

iframe:hover {
    border-color: rgba(102, 252, 241, 0.35) !important;
    box-shadow: 0 15px 40px rgba(102, 252, 241, 0.1) !important;
}
</style>
""", unsafe_allow_html=True)


# 2. Session State Initialization
if "sim_results" not in st.session_state:
    st.session_state["sim_results"] = None


# 3. Load Resources (Cached)
@st.cache_data
def get_node_coordinates(num_nodes=20):
    coords_file = "data/node_coords.json"
    if os.path.exists(coords_file):
        with open(coords_file, "r") as f:
            return json.load(f)
            
    st.info("🌐 Fetching Changwon driving network metadata... (One-time setup)")
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


# Initialize coordinates and agent
coords = get_node_coordinates(20)
agent = load_agent()

# Center latitude and longitude for Changwon Map
center_lat = sum([c['lat'] for c in coords]) / len(coords)
center_lon = sum([c['lon'] for c in coords]) / len(coords)


# 4. Sidebar Panel Configuration
with st.sidebar:
    st.markdown("### 🎛️ Control Hub")
    
    # Elegant custom model loaded status
    st.markdown("""
    <div class="sidebar-status">
        <div class="pulse-indicator"></div>
        <span>Network Engine: <b>MaskablePPO Active</b></span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Use the controls below to run dynamic routing comparisons using our deep reinforcement learning model.")
    
    run_sim = st.button("🚀 Run PPO Simulation", use_container_width=True)
    
    # Conditionally display clear button if results are active
    if st.session_state["sim_results"] is not None:
        st.markdown("---")
        clear_sim = st.button("🧹 Reset Dashboard", use_container_width=True)
        if clear_sim:
            st.session_state["sim_results"] = None
            st.rerun()


# 5. Main Title Panel
st.markdown("<h1>🚚 PPO vs Traffic: Live Changwon Routing</h1>", unsafe_allow_html=True)
st.markdown("<p class='dashboard-desc'>Comparing Deep MaskablePPO Reinforcement Learning vs Greedy Nearest Neighbor Baselines on real Changwon GIS road networks under dynamic traffic spikes.</p>", unsafe_allow_html=True)


# 6. Execute Simulation Logic
if run_sim:
    env = ChangwonRoutingEnv(num_nodes=20)
    
    with st.spinner("🤖 Simulating PPO RL Agent & Greedy Baseline over 100 dynamic episodes..."):
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
        
        # Run one specific demo episode for visualization
        obs, _ = env.reset(seed=42)
        done = False
        route_sequence = [0]
        demo_reward = 0
        while not done:
            action_masks = env.action_masks()
            action, _ = agent.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, terminated, truncated, _ = env.step(action)
            route_sequence.append(int(action))
            demo_reward += reward
            done = terminated or truncated
            
        # Store all simulation variables safely in Session State
        st.session_state["sim_results"] = {
            "ppo_mean": ppo_mean,
            "heuristic_mean": heuristic_mean,
            "pct_diff": pct_diff,
            "route_sequence": route_sequence,
            "demo_reward": demo_reward
        }
        
    st.toast("⚡ Simulation executed successfully! Mapped routes and metrics are updated.", icon="🎉")


# 7. Render KPI Metrics Cards
sim_results = st.session_state["sim_results"]

if sim_results is not None:
    ppo_mean = sim_results["ppo_mean"]
    heuristic_mean = sim_results["heuristic_mean"]
    pct_diff = sim_results["pct_diff"]
    route_sequence = sim_results["route_sequence"]
    demo_reward = sim_results["demo_reward"]
    
    # Format Advantage Indicator Color
    advantage_prefix = "+" if pct_diff >= 0 else ""
    
    kpi_html = f"""
    <div class="kpi-container">
        <div class="kpi-card ppo">
            <div class="kpi-title">🤖 PPO Agent Mean</div>
            <div class="kpi-value">{ppo_mean:.2f}</div>
            <div class="kpi-subtitle">100-Episode Average Reward</div>
        </div>
        <div class="kpi-card heuristic">
            <div class="kpi-title">📏 Greedy Heuristic</div>
            <div class="kpi-value">{heuristic_mean:.2f}</div>
            <div class="kpi-subtitle">Nearest Neighbor Baseline</div>
        </div>
        <div class="kpi-card advantage">
            <div class="kpi-title">🚀 Efficiency Gain</div>
            <div class="kpi-value">{advantage_prefix}{pct_diff:.2f}%</div>
            <div class="kpi-subtitle">PPO vs Baseline Heuristic</div>
        </div>
        <div class="kpi-card demo">
            <div class="kpi-title">🗺️ Demo Route Reward</div>
            <div class="kpi-value">{demo_reward:.2f}</div>
            <div class="kpi-subtitle">Visited {len(route_sequence)} Nodes (Seed 42)</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)
else:
    # Beautiful placeholder card indicating instructions
    st.markdown("""
    <div class="kpi-container">
        <div class="kpi-card demo" style="grid-column: 1 / -1; text-align: center; padding: 2.5rem;">
            <div class="kpi-title">Ready for execution</div>
            <div class="kpi-value" style="color: #66fcf1; font-size: 1.6rem;">No Simulation Active</div>
            <div class="kpi-subtitle" style="color: #8b949e; font-size: 0.9rem; margin-top: 0.5rem;">
                Click "🚀 Run PPO Simulation" in the Control Hub to compute routing paths and render telemetry.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# 8. Construct Folium Map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles="CartoDB dark_matter",
    control_scale=True
)

# Highlight Delivery Coordinates on the Map
for i, coord in enumerate(coords):
    color = '#66fcf1' if i == 0 else '#4facfe'
    radius = 8 if i == 0 else 5
    popup_text = f"📍 <b>Depot</b> (Start Node)" if i == 0 else f"📦 <b>Node {i}</b>"
    
    folium.CircleMarker(
        location=[coord['lat'], coord['lon']],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=popup_text
    ).add_to(m)

# Draw simulated path if results are active in Session State
if sim_results is not None:
    route_sequence = sim_results["route_sequence"]
    route_coords = [[coords[node]['lat'], coords[node]['lon']] for node in route_sequence]
    
    # Beautiful translucent cyan baseline line
    folium.PolyLine(
        route_coords,
        weight=3.5,
        color='#66fcf1',
        opacity=0.7,
        popup="PPO Planned Route Path"
    ).add_to(m)
    
    # Dynamic neon animated crawling dots along the path
    from folium import plugins
    plugins.AntPath(
        locations=route_coords,
        color="#a18cd1",
        pulse_color="#66fcf1",
        weight=5.5,
        delay=700,
        opacity=0.9
    ).add_to(m)

# Render map in layout
st_folium(m, width=1500, height=650)
