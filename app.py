import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pennylane as qml

# --- 1. CONFIGURATION & ANCHORS ---
st.set_page_config(page_title="Bootstrap Physics Node", layout="wide", page_icon="Φ")
PHI = (1 + np.sqrt(5)) / 2  # The Golden Ratio

# --- 2. QUANTUM ENGINE ---
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
    """Generates amplitudes: |ψ⟩ = Σ φ⁻ⁿ |n⟩"""
    n = np.arange(n_states)
    amps = 1.0 / (PHI**n)
    return amps / np.linalg.norm(amps)

# --- 3. BOOTSTRAP PHYSICS LOGIC ---
def update_agent_state(current_state, attractor, awareness, noise_level):
    """
    Update Rule: Z' = Z + phi * Grad(Z^3) + eta
    """
    # 1. Calculate Gradient towards Attractor (Meaning)
    gradient = attractor - current_state
    
    # 2. Apply Awareness (phi) as amplification
    drive = awareness * gradient
    
    # 3. Add Noise (eta), inversely scaled by awareness
    effective_noise = (noise_level / (awareness + 0.1)) * np.random.randn(len(current_state))
    
    # 4. Update
    new_state = current_state + drive + effective_noise
    return new_state

# --- 4. SESSION STATE INITIALIZATION ---
if 'init' not in st.session_state:
    n_states = 2**NUM_QUBITS
    
    # Initialize the "Bootstrap Attractor" (Z^3) using Phi Harmonics
    phi_amps = generate_phi_amplitudes(n_states)
    # We convert the QNode output to a numpy array immediately
    attractor_probs = np.array(harmonic_state_circuit(phi_amps))
    
    # Initialize Agents (Random states seeking coherence)
    st.session_state.attractor = attractor_probs
    st.session_state.agents = np.random.uniform(0, 0.1, (10, n_states)) # 10 Agents
    st.session_state.coherence_history = []
    st.session_state.step = 0
    st.session_state.init = True

# --- 5. DASHBOARD UI ---
st.title("Φ Bootstrap Physics // Coherence Engine")

# Sidebar: Physics Parameters
with st.sidebar:
    st.header("Field Parameters")
    awareness_phi = st.slider("Awareness (Φ)", 0.0, 1.0, 0.6, help="Amplification of meaning")
    noise_eta = st.slider("Entropy (η)", 0.0, 0.5, 0.05, help="Stochastic noise")
    
    # Phase Transition Indicator
    stability = awareness_phi**2
    # Avoid division by zero
    norm = np.linalg.norm(st.session_state.attractor)
    threshold = noise_eta / (norm + 1e-9)
    
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
    # 1. Coherence History
    if st.session_state.coherence_history:
        fig_coh = go.Figure()
        fig_coh.add_trace(go.Scatter(y=st.session_state.coherence_history, mode='lines', name='System Coherence', line=dict(color='#00ffcc', width=3)))
        fig_coh.update_layout(title="System Coherence Evolution (Δs)", template="plotly_dark", height=400)
        st.plotly_chart(fig_coh, use_container_width=True)

with col2:
    # 2. Quantum State Distribution
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

# 3. Interference Pattern
st.markdown("### Φ-Scaled Quantum Interference Pattern")
if st.session_state.init:
    # Simulate interference live
    phases = np.linspace(0, 2 * np.pi, 50)
    interference_map = []
    
    # Pre-calculate simplified interference for visualization speed
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
    
