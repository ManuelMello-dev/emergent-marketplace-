import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pennylane as qml
import torch

# --- 1. CONFIGURATION & ANCHORS ---
st.set_page_config(page_title="Bootstrap Physics Node", layout="wide", page_icon="Φ")
PHI = (1 + np.sqrt(5)) / 2  # The Golden Ratio (from quantum_harmonics.py)

# Constants from PDF (Bootstrap Physics)
DEFAULT_AWARENESS = 0.5     # phi parameter
CRITICAL_THRESHOLD = 0.3    # Phase transition point
NOISE_SCALE = 0.1           # eta parameter

# --- 2. QUANTUM ENGINE (From quantum_harmonics.py) ---
# We use 4 qubits to represent 16 conceptual states
NUM_QUBITS = 4
dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev)
def harmonic_state_circuit(amplitudes):
    """Constructs the quantum state based on harmonic/phi scaling."""
    # Normalize
    amplitudes = amplitudes / np.sqrt(np.sum(np.abs(amplitudes)**2))
    qml.StatePrep(amplitudes, wires=range(NUM_QUBITS))
    return qml.probs(wires=range(NUM_QUBITS))

def generate_phi_amplitudes(n_states):
    """Generates amplitudes: |ψ⟩ = Σ φ⁻ⁿ |n⟩ (from quantum_harmonics.py)"""
    n = np.arange(n_states)
    amps = 1.0 / (PHI**n)
    return amps / np.linalg.norm(amps)

# --- 3. BOOTSTRAP PHYSICS LOGIC (From PDF Section 3) ---
def update_agent_state(current_state, attractor, awareness, noise_level):
    """
    Update Rule: Z' = Z + phi * Grad(Z^3) + eta
    """
    # 1. Calculate Gradient towards Attractor (Meaning)
    gradient = attractor - current_state
    
    # 2. Apply Awareness (phi) as amplification (PDF Source 74)
    drive = awareness * gradient
    
    # 3. Add Noise (eta), inversely scaled by awareness (PDF Source 75)
    # "Higher awareness inversely scales stochastic noise"
    effective_noise = (noise_level / (awareness + 0.1)) * np.random.randn(len(current_state))
    
    # 4. Update
    new_state = current_state + drive + effective_noise
    return new_state

# --- 4. SESSION STATE ---
if 'init' not in st.session_state:
    n_states = 2**NUM_QUBITS
    
    # Initialize the "Bootstrap Attractor" (Z^3) using Phi Harmonics
    phi_amps = generate_phi_amplitudes(n_states)
    attractor_probs = harmonic_state_circuit(phi_amps)
    
    # Initialize Agents (Random states seeking coherence)
    st.session_state.attractor = np.array(attractor_probs)
    st.session_state.agents = np.random.uniform(0, 0.1, (10, n_states)) # 10 Agents
    st.session_state.coherence_history = []
    st.session_state.step = 0
    st.session_state.init = True

# --- 5. DASHBOARD UI ---
st.title("Φ Bootstrap Physics // Coherence Engine")

# Sidebar: Physics Parameters
with st.sidebar:
    st.header("Field Parameters")
    awareness_phi = st.slider("Awareness (Φ)", 0.0, 1.0, 0.6, help="Amplification of meaning (PDF Sec 3)")
    noise_eta = st.slider("Entropy (η)", 0.0, 0.5, 0.05, help="Stochastic noise")
    
    # Phase Transition Indicator (PDF Section 5)
    stability = awareness_phi**2
    threshold = noise_eta / (np.linalg.norm(st.session_state.attractor) + 1e-9)
    status = "ORDERED" if stability > threshold else "CHAOS"
    st.metric("System Regime", status, delta=f"{stability - threshold:.4f}")

    if st.button("RUN QUANTUM STEP"):
        # Run Physics Loop
        new_agents = []
        coherence_sum = 0
        
        for agent in st.session_state.agents:
            # Update Agent using Bootstrap Rule
            updated = update_agent_state(agent, st.session_state.attractor, awareness_phi, noise_eta)
            new_agents.append(updated)
            
            # Measure Coherence (Distance to Attractor)
            dist = np.linalg.norm(updated - st.session_state.attractor)
            coherence_sum += (1.0 / (dist + 1e-9)) # Inverse distance = coherence
            
        st.session_state.agents = np.array(new_agents)
        st.session_state.coherence_history.append(coherence_sum)
        st.session_state.step += 1
        st.rerun()

# --- 6. VISUALIZATION ---
col1, col2 = st.columns([2, 1])

with col1:
    # 1. Coherence History (The "Progress" Metric from PDF)
    if st.session_state.coherence_history:
        fig_coh = go.Figure()
        fig_coh.add_trace(go.Scatter(y=st.session_state.coherence_history, mode='lines', name='System Coherence', line=dict(color='#00ffcc', width=3)))
        fig_coh.update_layout(title="System Coherence Evolution (Δs)", template="plotly_dark", height=400)
        st.plotly_chart(fig_coh, use_container_width=True)

with col2:
    # 2. Quantum State Distribution (Visualizing quantum_harmonics.py output)
    fig_q = go.Figure()
    # Plot Attractor (Phi State)
    fig_q.add_trace(go.Bar(y=st.session_state.attractor, name='Attractor (Z³)', marker_color='gold'))
    # Plot Mean Agent State
    mean_agent = np.mean(st.session_state.agents, axis=0)
    fig_q.add_trace(go.Scatter(y=mean_agent, mode='lines+markers', name='Collective Mind', line=dict(color='cyan')))
    
    fig_q.update_layout(
        title="Quantum Harmonic State |n⟩", 
        template="plotly_dark", 
        height=400,
        yaxis_type="log",
        xaxis_title="State Index (0-15)"
    )
    st.plotly_chart(fig_q, use_container_width=True)

# 3. Interference Pattern (Heatmap from quantum_harmonics.py)
st.markdown("### Φ-Scaled Quantum Interference Pattern")
if st.session_state.init:
    # Simulate interference live
    phases = np.linspace(0, 2 * np.pi, 50)
    interference_map = []
    
    # We re-run the circuit with phases (simulating logic from your script)
    phi_amps = generate_phi_amplitudes(2**NUM_QUBITS)
    
    # Pre-calculate simplified interference for speed in UI
    base_probs = st.session_state.attractor
    for p in phases:
        # Simple modulation simulation for visualization speed
        modulated = base_probs * (1 + 0.5 * np.cos(p * np.arange(len(base_probs))))
        interference_map.append(modulated)
        
    fig_int = go.Figure(data=go.Heatmap(
        z=np.array(interference_map).T,
        x=phases,
        y=np.arange(len(base_probs)),
        colorscale='Magma'
    ))
    fig_int.update_layout(
        title="Phase Shift Interference (Memory Formation)",
        template="plotly_dark",
        height=400,
        xaxis_title="Phase Shift (θ)",
        yaxis_title="State Index |n⟩"
    )
    st.plotly_chart(fig_int, use_container_width=True)
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
