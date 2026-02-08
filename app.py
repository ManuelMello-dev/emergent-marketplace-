import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import yfinance as yf
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Emergent Marketplace", layout="wide", page_icon="⚡")

# --- CONSTANTS & CONFIG ---
DEVICE = "cpu"  # Force CPU for web deployment stability
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Default Tickers (Cluster approach from your notebook)
DEFAULT_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 
    'GLD', 'SLV', 'PBR', 'EC', 'VALE', 'AMD', 'NFLX'
]

# --- PHYSICS ENGINE (From your Notebook) ---
def apply_basel_constraints(prices, volumes, capital, enabled=True, max_risk=0.3):
    if not enabled: return prices
    risk_exposure = prices * volumes
    max_allowed = max_risk * capital
    excess = (risk_exposure - max_allowed).clamp(min=0)
    overexposed = excess > 0
    if overexposed.any():
        # Deleveraging penalty
        penalty = 0.02 * (excess / (capital + 1e-9))
        prices = prices * (1 - penalty)
    return prices

def apply_regime_shock(prices, baseline, step, enabled=True):
    if not enabled: return prices
    # Simple shock logic: if divergence is too high, crash it
    divergence = (prices - baseline).abs()
    shock_mask = divergence > (baseline * 0.15) # 15% divergence threshold
    if shock_mask.any():
        shock = torch.randn_like(prices) * -0.10 # 10% drop
        prices = torch.where(shock_mask, prices * (1 + shock), prices)
    return prices

# --- DATA LOADING ---
@st.cache_resource
def initialize_market_data(tickers):
    """Fetches initial data once and caches it."""
    try:
        data = yf.download(tickers, period="5d", interval="1d", progress=False)['Close']
        if data.empty:
            raise ValueError("No data fetched")
        
        # Get last valid price
        current_prices = data.iloc[-1].fillna(100.0).values
        
        # Create Tensors
        prices_tensor = torch.tensor(current_prices, dtype=torch.float32, device=DEVICE)
        
        # Create Derivatives (Synthetic agents from your notebook)
        # We assume 2 derivatives per ticker for density
        deriv_prices = prices_tensor.repeat(2) * torch.FloatTensor(len(prices_tensor)*2).uniform_(0.9, 1.1)
        
        combined_prices = torch.cat([prices_tensor, deriv_prices])
        
        # Names
        names = list(data.columns) + [f"{t}_D1" for t in data.columns] + [f"{t}_D2" for t in data.columns]
        
        return combined_prices, names
    except Exception as e:
        st.error(f"Data Fetch Failed: {e}. Using Synthetic Data.")
        # Fallback to synthetic data
        dummy_prices = torch.FloatTensor(30).uniform_(50, 200)
        dummy_names = [f"ASSET_{i}" for i in range(30)]
        return dummy_prices, dummy_names

# --- SESSION STATE SETUP ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.history_global = []
    st.session_state.step_count = 0

# --- UI LAYOUT ---
st.title("⚡ Emergent Marketplace // Consensus Node")

# Sidebar Controls
with st.sidebar:
    st.header("Model Parameters")
    
    # Simulation Tweaks
    enable_basel = st.checkbox("Basel III Constraints", value=True)
    enable_shocks = st.checkbox("Regime Shocks", value=True)
    enable_memory = st.checkbox("Memory Feedback", value=True)
    
    st.markdown("---")
    volatility_param = st.slider("System Volatility (Eta)", 0.01, 1.0, 0.4)
    steps_per_click = st.slider("Steps per Cycle", 1, 50, 10)
    
    if st.button("RESET SIMULATION", type="primary"):
        st.session_state.initialized = False
        st.rerun()

# --- INITIALIZATION LOGIC ---
if not st.session_state.initialized:
    with st.spinner("Initializing Quantum Tensor Field..."):
        prices, names = initialize_market_data(DEFAULT_TICKERS)
        
        # Store in Session State
        st.session_state.agent_prices = prices
        st.session_state.baseline_prices = prices.clone()
        st.session_state.agent_volumes = torch.ones_like(prices) * 1000  # Simplified volume
        st.session_state.agent_capital = prices * 1000 * 1.5 # Initial capital buffer
        st.session_state.agent_names = names
        st.session_state.history_global = []
        st.session_state.step_count = 0
        
        st.session_state.initialized = True
        st.success(f"System Online. Agents: {len(names)}")

# --- SIMULATION ENGINE ---
def run_simulation_step():
    prices = st.session_state.agent_prices
    baseline = st.session_state.baseline_prices
    volumes = st.session_state.agent_volumes
    capital = st.session_state.agent_capital
    
    # 1. Random Walk / Noise (The "Eta" param)
    noise = torch.randn_like(prices) * volatility_param
    new_prices = prices + noise
    
    # 2. Apply Notebook Logic (Tweaks)
    if enable_basel:
        new_prices = apply_basel_constraints(new_prices, volumes, capital)
    
    if enable_shocks:
        new_prices = apply_regime_shock(new_prices, baseline, st.session_state.step_count)
        
    # 3. Global Consensus (Z3 Metric)
    global_z3 = new_prices.mean().item()
    
    # 4. Update State
    st.session_state.agent_prices = new_prices
    st.session_state.history_global.append(global_z3)
    st.session_state.step_count += 1

# --- MAIN DASHBOARD ---
col1, col2 = st.columns([3, 1])

with col1:
    # Button to Advance Time
    if st.button(f"⚡ RUN {steps_per_click} CYCLES"):
        progress_bar = st.progress(0)
        for i in range(steps_per_click):
            run_simulation_step()
            progress_bar.progress((i + 1) / steps_per_click)
            time.sleep(0.05) # Tiny pause for visual effect
        st.rerun()

    # Main Plot: Global Consensus Z3
    if len(st.session_state.history_global) > 0:
        fig_z3 = go.Figure()
        fig_z3.add_trace(go.Scatter(
            y=st.session_state.history_global, 
            mode='lines', 
            name='Global Z3',
            line=dict(color='#00ffcc', width=2)
        ))
        fig_z3.update_layout(
            title="Global Z3 Consensus State",
            template="plotly_dark",
            height=400,
            yaxis_title="Price ($)"
        )
        st.plotly_chart(fig_z3, use_container_width=True)

with col2:
    st.metric("Simulation Step", st.session_state.step_count)
    
    # Current Top Movers
    if st.session_state.initialized:
        prices = st.session_state.agent_prices.numpy()
        names = st.session_state.agent_names
        
        # Calculate % Change from Baseline
        baseline = st.session_state.baseline_prices.numpy()
        delta = (prices - baseline) / baseline
        
        # Create DataFrame
        df = pd.DataFrame({"Ticker": names, "Price": prices, "Change": delta})
        df = df.sort_values("Change", ascending=False).head(10)
        
        st.markdown("### Top Gainers")
        st.dataframe(
            df.style.format({"Price": "${:.2f}", "Change": "{:+.2%}"}),
            hide_index=True
        )

# --- DEBUG / RAW DATA ---
with st.expander("View Tensor State"):
    st.write(st.session_state.agent_prices)
    
