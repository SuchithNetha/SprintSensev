import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import os
import logging
from datetime import datetime

# --- 1. CONFIG & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SprintSense-Frontend")

_raw_url = os.getenv("API_URL", "http://localhost:8000/predict")
if not _raw_url.startswith(("http://", "https://")):
    _raw_url = f"https://{_raw_url}"

if "/" not in _raw_url.split("//")[-1]:
    _raw_url = _raw_url.rstrip("/") + "/predict"

API_URL = _raw_url
HEALTH_URL = API_URL.replace("/predict", "/health")

st.set_page_config(
    page_title="SprintSense AI | Command Center",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# --- 2. PREMIUM THEME & STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top right, #1a1c2c, #0e1117);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 2px;
        color: #00f2ff !important;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.3);
    }

    /* Glassmorphism Containers */
    .stMetric, div[data-testid="stExpander"], .stForm {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    .stMetric:hover {
        border-color: #00f2ff !important;
        transform: translateY(-5px);
        transition: all 0.3s ease;
    }

    /* Gradient Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00f2ff, #0066ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.5) !important;
        transform: scale(1.02);
    }

    /* Status Pulse Animation */
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.5; }
        50% { transform: scale(1.05); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.5; }
    }
    .pulse-dot {
        height: 10px;
        width: 10px;
        background-color: #00f2ff;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }

    /* Custom Table Styling */
    .stTable {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Advice Box */
    .advice-card {
        background: linear-gradient(135deg, rgba(0, 242, 255, 0.1), rgba(0, 102, 255, 0.1));
        border-left: 4px solid #00f2ff;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
def init_session():
    defaults = {
        'team_data': {"name": "Delta Station", "members": []},
        'current_assessment_idx': 0,
        'temp_assessments': [],
        'api_status': "Unknown"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# --- 4. API HELPERS ---
@st.cache_data(ttl=60)
def check_api_health():
    try:
        resp = requests.get(HEALTH_URL, timeout=5)
        if resp.status_code == 200:
            return "Online"
        return "Degraded"
    except:
        return "Offline"

# --- 5. UI COMPONENTS ---

def map_response_to_int(response_text):
    text = response_text.lower()
    if any(x in text for x in ["overload", "today", "constant", "diverging", "chaos", "<4h"]): return 3
    if any(x in text for x in ["high", "3-4 days", "frequent", "stalled", "noisy", "4-5h"]): return 2
    if any(x in text for x in ["normal", "next week", "few", "converging", "quiet", "6-7h"]): return 1
    return 0

@st.dialog("Cognitive Assessment Console")
def run_assessment_dialog(member_name, role, total_count):
    st.markdown(f"### 📡 Scanning: {member_name}")
    st.caption(f"Role: {role}")
    
    progress = (st.session_state.current_assessment_idx + 1) / total_count
    st.progress(progress, text=f"Syncing Profile {st.session_state.current_assessment_idx + 1}/{total_count}")

    with st.form("assessment_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            ticket_vol = st.select_slider("Current Workload", options=["Low (1-2)", "Normal (3-4)", "High (5-6)", "Overload (7+)"])
            deadline = st.select_slider("Time Sensitivity", options=["Next Week", "3-4 Days", "Tomorrow", "Today"])
        with c2:
            sleep = st.select_slider("Biological Recovery (Sleep)", options=["Excellent (8h+)", "Good (6-7h)", "Fair (4-5h)", "Poor (<4h)"])
            interruptions = st.select_slider("System Noise (Interruptions)", options=["None", "Few", "Frequent", "Constant"])

        st.divider()
        st.markdown(f"**Field Specifics:** {role}")
        if role == "AI/ML Engineer":
            complexity = st.radio("Neural Network Convergence?", ["Converging (Stable)", "Slow Convergence", "Stalled/Plateau", "Diverging/NaN (Critical)"])
        else:
            complexity = st.radio("Infrastructure Performance?", ["Stable", "High Latency", "Connection Errors", "Downtime/Locks"])

        submitted = st.form_submit_button("UNLEASH NEURAL ANALYSIS")

        if submitted:
            payload = {
                "name": member_name,
                "role": role,
                "ticket_volume": map_response_to_int(ticket_vol),
                "deadline_proximity": map_response_to_int(deadline),
                "sleep_quality": map_response_to_int(sleep),
                "interruptions": map_response_to_int(interruptions),
                "complexity": map_response_to_int(complexity)
            }

            try:
                with st.spinner("🧠 GENERATING SYNTHETIC EEG WAVEFORMS..."):
                    response = requests.post(API_URL, json=payload, timeout=12)
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.temp_assessments.append({"name": member_name, "role": role, **result, "timestamp": datetime.now().strftime("%H:%M:%S")})
                        st.session_state.current_assessment_idx += 1
                        st.rerun()
                    else:
                        st.error(f"NEURAL SYNC FAILED: ERROR {response.status_code}")
            except Exception as e:
                st.error(f"LINK DISCONNECTED: {e}")

def render_history_view():
    st.title("📈 HISTORICAL ANALYTICS")
    st.caption("DEEP-TIME COGNITIVE LOAD TRENDS")
    
    try:
        history_url = API_URL.replace("/predict", "/history")
        response = requests.get(history_url, timeout=10)
        if response.status_code == 200:
            history_data = response.json()
            if not history_data:
                st.info("ARCHIVES EMPTY. INITIALIZE FIRST SCANS.")
                return
            
            df = pd.DataFrame(history_data, columns=["ID", "Timestamp", "Name", "Role", "Status", "Alpha", "Beta", "Delta", "Theta"])
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Team Biome State")
                fig_pie = px.pie(df, names="Status", color="Status", hole=0.4,
                                color_discrete_map={"Stressed": "#ff4b4b", "Fatigued": "#636efa", "Focused": "#00cc96", "Distracted": "#fec032"})
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#fff")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                st.subheader("Beta Intensity (Stress Monitor)")
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                fig_line = px.line(df, x="Timestamp", y="Beta", color="Name", markers=True)
                fig_line.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_line, use_container_width=True)

            # Data Utilities
            st.divider()
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.subheader("📥 Data Export")
            with col_b:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("DOWNLOAD AUDIT LOG (CSV)", data=csv, file_name=f"sprintsense_logs_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
                
            with st.expander("VIEW RAW NEURAL DATASTREAM"):
                st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True)
    except Exception as e:
        st.error(f"ARCHIVE ACCESS FAILED: {e}")

def render_dashboard():
    members = st.session_state.team_data['members']
    if st.session_state.current_assessment_idx < len(members):
        curr_member = members[st.session_state.current_assessment_idx]
        run_assessment_dialog(curr_member['name'], curr_member['role'], len(members))

    st.title(f"🛰️ {st.session_state.team_data['name'].upper()} MONITOR")

    if st.session_state.current_assessment_idx >= len(members):
        results = st.session_state.temp_assessments
        c1, c2, c3, c4 = st.columns(4)
        states = [r['state'] for r in results]
        stressed_count = states.count("Stressed") + states.count("Fatigued")

        c1.metric("THREAT LEVEL", "CRITICAL" if stressed_count > 0 else "NOMINAL", f"{stressed_count} AT RISK")
        c2.metric("SPRINT MOMENTUM", "94%", "-12%" if stressed_count > 0 else "+4%")
        c3.metric("DOMINANT FREQUENCY", max(set(states), key=states.count))
        c4.metric("ACTIVE NODES", len(members))

        st.divider()
        for res in results:
            with st.expander(f"👤 NODE: {res['name'].upper()} | STATE: {res['state'].upper()}", expanded=True):
                col_graph, col_advice = st.columns([2, 1])
                with col_graph:
                    x = np.linspace(0, 10, 60)
                    fig = go.Figure()
                    # Synthetic wave representation
                    fig.add_trace(go.Scatter(x=x, y=np.sin(x*3)*res['eeg_data']['beta'] + 2, name='Beta (High)', line=dict(color='#FF4B4B', width=3)))
                    fig.add_trace(go.Scatter(x=x, y=np.sin(x)*res['eeg_data']['alpha'] - 2, name='Alpha (Base)', line=dict(color='#00CC96', width=2)))
                    fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", 
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    st.plotly_chart(fig, use_container_width=True)
                with col_advice:
                    st.markdown(f"""
                    <div class="advice-card">
                        <small style="color:#00f2ff">NEURAL ADVISOR [v2.1]</small><br>
                        <strong>PROTOCOL:</strong><br>
                        {res['advice']}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("SCANNING TEAM SIGNATURES... PLEASE WAIT.")

# --- 6. MAIN ROUTER ---
def main():
    with st.sidebar:
        st.markdown("<h1 style='font-size:1.5rem'>SPRINTSENSE AI</h1>", unsafe_allow_html=True)
        st.caption("CYBERNETIC AGILE MANAGEMENT")
        
        # Health Monitor
        status = check_api_health()
        color = "#00f2ff" if status == "Online" else "#ff4b4b"
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.05); padding:15px; border-radius:12px; border:1px solid rgba(255,255,255,0.1)'>
            <div class="pulse-dot" style="background-color:{color}"></div>
            <span style="color:{color}; font-weight:bold; letter-spacing:1px">SYSTEM {status.upper()}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        nav = st.radio("SATELLITE SYSTEMS", ["CONFIG", "DASHBOARD", "ARCHIVES"])
        
        st.divider()
        if st.button("HARD RESET SESSION"):
            st.session_state.clear()
            st.rerun()

    if nav == "ARCHIVES":
        render_history_view()
    elif nav == "DASHBOARD":
        if not st.session_state.team_data['members']:
            st.warning("ERROR: NO NODES CONFIGURED. ACCESS SETUP SYSTEM.")
        else:
            render_dashboard()
    else:
        st.title("🛰️ SYSTEM CONFIG")
        with st.container(border=True):
            st.subheader("Network Identifier")
            team_name = st.text_input("SET SQUADRON NAME", value=st.session_state.team_data['name'])
            st.session_state.team_data['name'] = team_name
            
            st.divider()
            st.subheader("Add Synthetic Node")
            col1, col2 = st.columns(2)
            name = col1.text_input("OPERATIVE NAME")
            role = col2.selectbox("FUNCTIONAL CLASS", ["AI/ML Engineer", "Backend Developer"])
            
            if st.button("ENROLL NODE", use_container_width=True):
                if name:
                    st.session_state.team_data['members'].append({"name": name, "role": role})
                    st.success(f"NODE {name.upper()} ACTIVE")

        if st.session_state.team_data['members']:
            st.subheader("Active Roster")
            st.table(pd.DataFrame(st.session_state.team_state['members'] if 'members' in st.session_state.team_data else st.session_state.team_data['members']))
            if st.button("INITIALIZE COMMAND CENTER", type="primary"):
                st.rerun()

if __name__ == "__main__":
    main()
