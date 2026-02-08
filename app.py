import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

# --- 1. CONFIGURATION (Matching Notebook Source 32-34) ---
st.set_page_config(page_title="Emergent Marketplace", layout="wide", page_icon="⚡")
DEVICE = "cpu"
torch.manual_seed(42)
np.random.seed(42)

# Physics Constants
MAX_RISK_RATIO = 0.3
DELEVERAGING_SEVERITY = 0.02
SLOW_AGENT_RATIO = 0.3
SLOW_REACTION_RATE = 0.05
SHOCK_THRESHOLD = 0.10
SHOCK_MAGNITUDE = 0.15
MEMORY_ALPHA = 0.1
MEMORY_INFLUENCE = 0.05
SCARCITY_PREMIUM_FACTOR = 0.1
MAX_DELIVERY_RATE = 0.02
DERIVATIVES_PER_TICKER = 3  # From Source 32

# Ticker Clusters
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'GLD', 'SLV', 'PBR', 'VALE', 'AMD', 'NFLX']
COMMODITY_TICKERS = ['GLD', 'SLV', 'GC=F', 'SI=F']
LATAM_TICKERS = ['PBR', 'EC', 'VALE']

# --- 2. PHYSICS ENGINE (Exact Port from Sources 49-54) ---

def apply_basel_constraints(prices, volumes, capital):
    """Source 49: Basel III Capital Constraints"""
    risk_exposure = prices * volumes
    max_allowed = MAX_RISK_RATIO * capital
    excess = (risk_exposure - max_allowed).clamp(min=0)
    
    if excess.any():
        # Deleveraging penalty
        penalty = DELEVERAGING_SEVERITY * (excess / (capital + 1e-9))
        prices = prices * (1 - penalty)
    return prices

def apply_tiered_liquidity(prices, target_prices, is_slow):
    """Source 50: Slow agents (Institutions) respond with delay"""
    delta = target_prices - prices
    # Fast agents move full delta, Slow agents move partial
    update = torch.where(is_slow, delta * SLOW_REACTION_RATE, delta)
    return prices + update

def apply_delivery_friction(prices, volumes, inventory, is_commodity):
    """Source 50-51: Physical Delivery Constraints"""
    delivery = volumes * MAX_DELIVERY_RATE
    # Deplete inventory
    inventory = (inventory - delivery).clamp(min=0)
    
    # Scarcity Premium
    scarcity = 1.0 / (inventory + 1.0)
    premium = SCARCITY_PREMIUM_FACTOR * scarcity
    
    # Apply only to commodity agents
    prices = torch.where(is_commodity, prices * (1 + premium), prices)
    return prices, inventory

def apply_regime_shock(prices, baseline, latam_mask):
    """Source 51-52: Endogenous Shocks (LatAm Cluster)"""
    # Check divergence for LATAM cluster
    if latam_mask.any():
        cluster_prices = prices[latam_mask]
        cluster_base = baseline[latam_mask]
        divergence = (cluster_prices - cluster_base).abs().mean()
        
        if divergence > SHOCK_THRESHOLD:
            # Apply shock
            shock = torch.randn(latam_mask.sum()) * SHOCK_MAGNITUDE
            prices[latam_mask] = prices[latam_mask] * (1 - shock.abs()) # Force drop
            return prices, True
    return prices, False

def apply_memory_feedback(prices, memory):
    """Source 52-53: Memory Support/Resistance"""
    # Update memory (EMA)
    memory = MEMORY_ALPHA * prices + (1 - MEMORY_ALPHA) * memory
    # Pull prices toward memory
    prices = prices + MEMORY_INFLUENCE * (memory - prices)
    return prices, memory

def compute_influence_consensus(prices, volumes, is_slow):
    """Source 54: Dollar-Weighted Consensus (Z3)"""
    power = (volumes * prices)
    # Slow agents get 2.0x weight (Source 33)
    weights = torch.where(is_slow, power * 2.0, power)
    z3 = (prices * weights).sum() / (weights.sum() + 1e-9)
    return z3.item()

# --- 3. INITIALIZATION (Adapting Source 43-48 for Web) ---
def init_simulation():
    # 1. Fetch Data
    try:
        data = yf.download(TICKERS, period="5d", interval="1d", progress=False)['Close']
        base_prices = data.iloc[-1].fillna(150.0)
    except:
        # Fallback if API fails
        base_prices = pd.Series(np.random.uniform(100, 200, len(TICKERS)), index=TICKERS)

    # 2. Create Derivatives (Source 47)
    agent_rows = []
    for ticker, price in base_prices.items():
        # Underlying Agent
        agent_rows.append({'id': ticker, 'price': price, 'parent': ticker})
        # Derivative Agents
        for d in range(DERIVATIVES_PER_TICKER):
            d_price = price * np.random.uniform(0.95, 1.05)
            agent_rows.append({'id': f"{ticker}_D{d+1}", 'price': d_price, 'parent': ticker})
            
    df = pd.DataFrame(agent_rows)
    num_agents = len(df)
    
    # 3. Create Tensors
    prices = torch.tensor(df['price'].values, dtype=torch.float32)
    
    state = {
        'prices': prices,
        'baseline': prices.clone(),
        'volumes': torch.ones(num_agents) * 10000, 
        'capital': prices * 10000 * 1.5,
        'memory': prices.clone(),
        'inventory': torch.ones(num_agents) * 1000,
        'history_z3': [],
        'step': 0,
        'ids': df['id'].tolist(),
        'parents': df['parent'].tolist()
    }
    
    # 4. Classifications (Source 48-49)
    # Slow Agents (Institutions) - Top 30%
    state['is_slow'] = torch.zeros(num_agents, dtype=torch.bool)
    # Simple logic: assign random 30% as slow for diversity in derivatives
    indices = torch.randperm(num_agents)[:int(num_agents * SLOW_AGENT_RATIO)]
    state['is_slow'][indices] = True
    
    # Commodity Agents
    state['is_commodity'] = torch.tensor([t in COMMODITY_TICKERS for t in state['parents']], dtype=torch.bool)
    
    # LatAm Agents
    state['is_latam'] = torch.tensor([t in LATAM_TICKERS for t in state['parents']], dtype=torch.bool)
    
    return state

# Initialize Session State
if 'sim' not in st.session_state:
    st.session_state.sim = init_simulation()

# --- 4. MAIN LOOP (Source 56-57 - Exact Order) ---
def step_simulation(steps=1, volatility=0.2):
    sim = st.session_state.sim
    prices = sim['prices']
    
    for _ in range(steps):
        # Base Dynamics
        noise = torch.randn_like(prices) * volatility
        target = prices + noise
        
        # --- EXECUTE TWEAKS IN ORDER (Source 56-57) ---
        
        # 1. Tiered Liquidity
        prices = apply_tiered_liquidity(prices, target, sim['is_slow'])
        
        # 2. Memory Feedback
        prices, sim['memory'] = apply_memory_feedback(prices, sim['memory'])
        
        # 3. Basel Constraints
        prices = apply_basel_constraints(prices, sim['volumes'], sim['capital'])
        
        # 4. Delivery Friction
        prices, sim['inventory'] = apply_delivery_friction(prices, sim['volumes'], sim['inventory'], sim['is_commodity'])
        
        # 5. Regime Shocks
        prices, shock = apply_regime_shock(prices, sim['baseline'], sim['is_latam'])
        
        # Metrics
        z3 = compute_influence_consensus(prices, sim['volumes'], sim['is_slow'])
        sim['history_z3'].append(z3)
        sim['step'] += 1
        
    sim['prices'] = prices

# --- 5. DASHBOARD ---
st.title("⚡ Emergent Marketplace // Kernel v1.0")
sim = st.session_state.sim

with st.sidebar:
    st.header("Physics Controls")
    vol = st.slider("Volatility (Phi)", 0.01, 1.0, 0.4)
    cycles = st.slider("Cycles per Click", 1, 20, 5)
    
    if st.button("RUN SIMULATION", type="primary"):
        step_simulation(cycles, vol)
        st.rerun()
        
    if st.button("RESET SYSTEM"):
        st.session_state.sim = init_simulation()
        st.rerun()

# Layout
col1, col2 = st.columns([3, 1])

with col1:
    # Z3 Chart
    if sim['history_z3']:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=sim['history_z3'], mode='lines', name='Global Z3', line=dict(color='#00ffcc')))
        fig.update_layout(
            title=f"Global Z3 Consensus: ${sim['history_z3'][-1]:.2f}", 
            template="plotly_dark", 
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Agent Ledger
    df = pd.DataFrame({
        "Agent": sim['ids'],
        "Price": sim['prices'].numpy(),
        "Role": ["INSTITUTIONAL" if s else "HFT" for s in sim['is_slow'].numpy()]
    })
    st.markdown(f"### Active Agents: {len(df)}")
    st.dataframe(
        df.sort_values("Price", ascending=False).head(15).style.format({"Price": "${:.2f}"}), 
        height=450, 
        hide_index=True
    )
