import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# 1. Page Configuration
st.set_page_config(
    page_title="Emergent Marketplace // Attractor Model",
    layout="wide",
    page_icon="⚡"
)

# 2. Sidebar Controls
with st.sidebar:
    st.header("Simulation Parameters")
    st.write("Tweak these to alter the attractor field.")
    
    volatility = st.slider("Market Volatility (Sigma)", 0.01, 1.0, 0.15)
    recursion_depth = st.slider("Recursion Depth", 10, 500, 100)
    bias = st.slider("Directional Bias", -0.5, 0.5, 0.0)
    
    st.markdown("---")
    st.caption("Status: HOSTED & LIVE")

# 3. Main Interface
st.title("⚡ Emergent Marketplace Dashboard")
st.markdown("### Universal Attractor Simulation")

col1, col2 = st.columns([3, 1])

# 4. The "Math" (Simulating your logic)
# In the future, this is where we paste your actual notebook code.
def generate_attractor(steps, vol, drift):
    # Generating a "Random Walk" that mimics market physics
    returns = np.random.normal(loc=drift, scale=vol, size=steps)
    price_path = 100 * np.cumprod(1 + returns)
    
    # Adding a "Strange Attractor" visualization (Phase Space)
    phase_x = price_path[:-1]
    phase_y = price_path[1:]
    return price_path, phase_x, phase_y

# Generate data based on slider inputs
prices, px, py = generate_attractor(recursion_depth, volatility, bias)

# 5. Visuals
with col1:
    # Main Time Series Plot
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(y=prices, mode='lines', name='Asset Value', line=dict(color='#00ffcc')))
    fig_main.update_layout(
        title="Time-Domain Trajectory",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_main, use_container_width=True)

with col2:
    # Phase Space Plot (Attractor View)
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=px, y=py, mode='markers', marker=dict(size=5, color=py, colorscale='Viridis')))
    fig_phase.update_layout(
        title="Phase Space Attractor",
        template="plotly_dark",
        height=400,
        xaxis_title="Price (t)",
        yaxis_title="Price (t+1)",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_phase, use_container_width=True)

# 6. Data Table
st.markdown("### Raw Simulation Data")
st.dataframe(pd.DataFrame({"Price": prices, "Delta": np.append([0], np.diff(prices))}).style.highlight_max(axis=0))
