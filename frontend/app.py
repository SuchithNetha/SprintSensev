import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import os
import logging

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
    page_title="SprintSense AI | Production",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# --- 2. THEME & CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    .stMetric:hover { border-color: #ff4b4b; transform: translateY(-2px); }
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        border-color: #ff4b4b;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.2);
    }
    /* Status Badge */
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
def init_session():
    defaults = {
        'team_data': {"name": "", "members": []},
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
    if any(x in text for x in ["critical", "today", "constant", "diverging", "chaos", "<4h"]): return 3
    if any(x in text for x in ["high", "tomorrow", "frequent", "stalled", "noisy", "4-5h"]): return 2
    if any(x in text for x in ["moderate", "3 days", "few", "converging", "quiet", "6-7h"]): return 1
    return 0

@st.dialog("Cognitive Assessment Console")
def run_assessment_dialog(member_name, role, total_count):
    st.caption(f"Member: **{member_name}** | Role: **{role}**")
    progress = (st.session_state.current_assessment_idx + 1) / total_count
    st.progress(progress, text=f"Assessment {st.session_state.current_assessment_idx + 1} of {total_count}")

    with st.form("assessment_form"):
        c1, c2 = st.columns(2)
        with c1:
            ticket_vol = st.select_slider("Current Ticket Volume", options=["Low (1-2)", "Normal (3-4)", "High (5-6)", "Overload (7+)"])
            deadline = st.select_slider("Nearest Deadline", options=["Next Week", "3-4 Days", "Tomorrow", "Today"])
        with c2:
            sleep = st.select_slider("Sleep Quality (Last Night)", options=["Excellent (8h+)", "Good (6-7h)", "Fair (4-5h)", "Poor (<4h)"])
            interruptions = st.select_slider("Context Switching", options=["None", "Few", "Frequent", "Constant"])

        st.divider()
        st.caption(f"Department Specific: {role}")
        if role == "AI/ML Engineer":
            complexity = st.radio("Model Training Status?", ["Converging (Stable)", "Slow Convergence", "Stalled/Plateau", "Diverging/NaN (Critical)"])
        else:
            complexity = st.radio("API/Database Status?", ["Stable", "High Latency", "Connection Errors", "Downtime/Locks"])

        submitted = st.form_submit_button("Submit Assessment")

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
                with st.spinner("Connecting to Neural Engine..."):
                    response = requests.post(API_URL, json=payload, timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.temp_assessments.append({"name": member_name, "role": role, **result})
                        st.session_state.current_assessment_idx += 1
                        st.rerun()
                    else:
                        st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

def render_history_view():
    st.title("📈 Team Stress Trends")
    st.caption("Historical insights from the SprintSense Database")
    
    try:
        history_url = API_URL.replace("/predict", "/history")
        response = requests.get(history_url, timeout=10)
        if response.status_code == 200:
            history_data = response.json()
            if not history_data:
                st.info("No historical data found. Run some assessments first!")
                return
            
            df = pd.DataFrame(history_data, columns=["ID", "Timestamp", "Name", "Role", "Status", "Alpha", "Beta", "Delta", "Theta"])
            
            # Distribution
            st.subheader("Mental State Distribution")
            fig_pie = px.pie(df, names="Status", color="Status", 
                            color_discrete_map={"Stressed": "#ff4b4b", "Fatigued": "#636efa", "Focused": "#00cc96", "Distracted": "#fec032"})
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Timeline
            st.subheader("Cognitive Load Timeline")
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            fig_line = px.line(df, x="Timestamp", y="Beta", color="Name", markers=True)
            fig_line.update_layout(template="plotly_dark")
            st.plotly_chart(fig_line, use_container_width=True)
            
            with st.expander("View Raw Audit Logs"):
                st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True)
        else:
            st.error(f"Failed to fetch history: {response.status_code}")
    except Exception as e:
        st.error(f"History Engine Offline: {e}")

def render_dashboard():
    members = st.session_state.team_data['members']
    if st.session_state.current_assessment_idx < len(members):
        curr_member = members[st.session_state.current_assessment_idx]
        run_assessment_dialog(curr_member['name'], curr_member['role'], len(members))

    st.title(f"📊 {st.session_state.team_data['name']} Monitor")

    if st.session_state.current_assessment_idx >= len(members):
        results = st.session_state.temp_assessments
        c1, c2, c3, c4 = st.columns(4)
        states = [r['state'] for r in results]
        stressed_count = states.count("Stressed") + states.count("Fatigued")

        c1.metric("Risk Level", "Critical" if stressed_count > 0 else "Optimal", f"{stressed_count} At-Risk")
        c2.metric("Velocity Impact", "-12pts" if stressed_count > 0 else "+5pts")
        c3.metric("Team Mood", max(set(states), key=states.count))
        c4.metric("Active Size", len(members))

        st.divider()
        for res in results:
            with st.expander(f"👤 {res['name']} ({res['role']}) - {res['state']}", expanded=True):
                col_graph, col_advice = st.columns([2, 1])
                with col_graph:
                    x = np.linspace(0, 10, 50)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=np.sin(x*3)*res['eeg_data']['beta'], name='Beta', line=dict(color='#FF4B4B')))
                    fig.add_trace(go.Scatter(x=x, y=np.sin(x)*res['eeg_data']['alpha'], name='Alpha', line=dict(color='#00CC96')))
                    fig.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                with col_advice:
                    st.caption("AI Guidance")
                    st.write(res['advice'])
    else:
        st.info("Assessments in progress...")

# --- 6. MAIN ROUTER ---
def main():
    with st.sidebar:
        st.title("SprintSense 2.1")
        st.caption("Neuro-Agile Production")
        nav = st.radio("Navigation", ["Setup", "Dashboard", "Trends"])
        st.divider()
        
        # Health Monitor
        status = check_api_health()
        color = "green" if status == "Online" else "red"
        st.markdown(f"**System Status:** <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
        
        if st.button("Reset Session"):
            st.session_state.clear()
            st.rerun()

    if nav == "Trends":
        render_history_view()
    elif nav == "Dashboard":
        if not st.session_state.team_data['members']:
            st.warning("Create a team first!")
        else:
            render_dashboard()
    else:
        st.title("🚀 Team Configuration")
        with st.container(border=True):
            team_name = st.text_input("Team Identifier", value=st.session_state.team_data['name'])
            st.session_state.team_data['name'] = team_name
            
            col1, col2 = st.columns(2)
            name = col1.text_input("Member Name")
            role = col2.selectbox("Role", ["AI/ML Engineer", "Backend Developer"])
            
            if st.button("Add Member", use_container_width=True):
                if name:
                    st.session_state.team_data['members'].append({"name": name, "role": role})
                    st.success(f"Added {name}")

        if st.session_state.team_data['members']:
            st.subheader("Roster")
            st.table(pd.DataFrame(st.session_state.team_data['members']))
            if st.button("Initialize Dashboard", type="primary"):
                st.rerun()

if __name__ == "__main__":
    main()
