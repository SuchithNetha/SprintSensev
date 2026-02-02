import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import os

# --- CONFIG ---
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
st.set_page_config(page_title="SprintSense AI", layout="wide", page_icon="🧠")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    .stMetric:hover {
        border-color: #ff4b4b;
        transform: translateY(-2px);
    }
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
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'team_data' not in st.session_state:
    st.session_state.team_data = {"name": "", "members": []}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'current_assessment_idx' not in st.session_state:
    st.session_state.current_assessment_idx = 0
if 'temp_assessments' not in st.session_state:
    st.session_state.temp_assessments = []


# --- HELPER FUNCTIONS ---

def map_response_to_int(response_text):
    """Maps UI text options to 0-3 integers for the ML model."""
    # Simple logic: Higher intensity keywords = Higher score
    # This is a heuristic mapping for the MVP
    text = response_text.lower()
    if any(x in text for x in ["critical", "today", "constant", "diverging", "chaos", "<4h"]): return 3
    if any(x in text for x in ["high", "tomorrow", "frequent", "stalled", "noisy", "4-5h"]): return 2
    if any(x in text for x in ["moderate", "3 days", "few", "converging", "quiet", "6-7h"]): return 1
    return 0  # Default (Safe/Good)


@st.dialog("Cognitive Assessment Console")
def run_assessment_dialog(member_name, role, total_count):
    st.caption(f"Member: **{member_name}** | Role: **{role}**")
    progress = (st.session_state.current_assessment_idx + 1) / total_count
    st.progress(progress, text=f"Assessment {st.session_state.current_assessment_idx + 1} of {total_count}")

    with st.form("assessment_form"):
        # Common Inputs
        c1, c2 = st.columns(2)
        with c1:
            ticket_vol = st.select_slider("Current Ticket Volume",
                                          options=["Low (1-2)", "Normal (3-4)", "High (5-6)", "Overload (7+)"])
            deadline = st.select_slider("Nearest Deadline", options=["Next Week", "3-4 Days", "Tomorrow", "Today"])
        with c2:
            sleep = st.select_slider("Sleep Quality (Last Night)",
                                     options=["Excellent (8h+)", "Good (6-7h)", "Fair (4-5h)", "Poor (<4h)"])
            interruptions = st.select_slider("Context Switching", options=["None", "Few", "Frequent", "Constant"])

        # Role Specific Input (Adds flavor to the demo)
        st.divider()
        st.caption(f"Department Specific: {role}")
        if role == "AI/ML Engineer":
            complexity = st.radio("Model Training Status?",
                                  ["Converging (Stable)", "Slow Convergence", "Stalled/Plateau",
                                   "Diverging/NaN (Critical)"])
        else:  # Backend
            complexity = st.radio("API/Database Status?",
                                  ["Stable", "High Latency", "Connection Errors", "Downtime/Locks"])

        submitted = st.form_submit_button("Submit Assessment")

        if submitted:
            # 1. Map Data
            payload = {
                "name": member_name,
                "role": role,
                "ticket_volume": map_response_to_int(ticket_vol),
                "deadline_proximity": map_response_to_int(deadline),
                "sleep_quality": map_response_to_int(sleep),
                "interruptions": map_response_to_int(interruptions),
                "complexity": map_response_to_int(complexity)
            }

            # 2. Call API (The "Analysis" Step)
            try:
                with st.spinner("Connecting to Neural Engine..."):
                    response = requests.post(API_URL, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        # Combine Member Info with Analysis Result
                        full_profile = {
                            "name": member_name,
                            "role": role,
                            **result  # Unpacks state, eeg_data, advice
                        }
                        st.session_state.temp_assessments.append(full_profile)
                        st.session_state.current_assessment_idx += 1
                        st.rerun()  # Closes dialog and triggers next step
                    else:
                        st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")


# --- MAIN PAGE LAYOUT ---

# --- MAIN PAGE LAYOUT ---

def render_history_view():
    st.title("📈 Team Stress Trends")
    st.caption("Historical insights from the SprintSense Database")
    
    try:
        # Construct the history URL by replacing 'predict' with 'history'
        history_url = API_URL.replace("/predict", "/history")
        response = requests.get(history_url)
        if response.status_code == 200:
            history_data = response.json()
            if not history_data:
                st.info("No historical data found. Run some assessments first!")
                return
            
            df = pd.DataFrame(history_data, columns=["ID", "Timestamp", "Name", "Role", "Status", "Alpha", "Beta", "Delta", "Theta"])
            
            # 1. Stress Prevalence Chart
            st.subheader("Mental State Distribution")
            fig_pie = px.pie(df, names="Status", color="Status", 
                            color_discrete_map={"Stressed": "#ff4b4b", "Fatigued": "#636efa", "Focused": "#00cc96", "Distracted": "#fec032"})
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # 2. Timeline of Stress
            st.subheader("Cognitive Load Timeline")
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            fig_line = px.line(df, x="Timestamp", y="Beta", color="Name", markers=True, title="Beta Wave Intensity (Stress Proxy)")
            fig_line.update_layout(template="plotly_dark")
            st.plotly_chart(fig_line, use_container_width=True)
            
            # 3. Raw Logs
            with st.expander("View Raw Audit Logs"):
                st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True)
                
        else:
            st.error(f"Failed to fetch history: {response.status_code}")
    except Exception as e:
        st.error(f"History Engine Offline: {e}")


def render_manager_view():
    st.title("🚀 SprintSense Setup")
    
    # Sidebar Navigation for Pro feel
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1693/1693746.png", width=100)
        st.title("SprintSense 2.0")
        nav = st.radio("Navigation", ["Setup Team", "Live Dashboard", "History & Trends"])
        st.divider()
        st.info("Railway Deployment Active")

    if nav == "History & Trends":
        render_history_view()
        return
    elif nav == "Live Dashboard":
        if not st.session_state.team_data['members']:
            st.warning("Please setup a team first!")
            return
        st.session_state.page = "dashboard"
        render_dashboard()
        return

    # Original Setup View
    c1, c2 = st.columns([1, 2])
    with c1:
        with st.container(border=True):
            st.subheader("Create Team")
            team_name = st.text_input("Team Name", placeholder="e.g. Delta Force", key="team_name_input")

            st.divider()
            st.caption("Add Member")
            new_name = st.text_input("Name")
            new_role = st.selectbox("Role", ["AI/ML Engineer", "Backend Developer"])

            if st.button("Add to Roster", use_container_width=True):
                if new_name:
                    st.session_state.team_data['members'].append({"name": new_name, "role": new_role})
                    st.success(f"Added {new_name}")

    with c2:
        st.subheader("Team Roster")
        if st.session_state.team_data['members']:
            df_members = pd.DataFrame(st.session_state.team_data['members'])
            st.dataframe(df_members, use_container_width=True, hide_index=True)

            if len(st.session_state.team_data['members']) > 0 and team_name:
                if st.button("✅ Finalize Team & Open Dashboard", type="primary"):
                    st.session_state.team_data['name'] = team_name
                    st.session_state.page = "dashboard"
                    st.rerun()
        else:
            st.info("Add team members to proceed.")


def render_dashboard():
    # Check if we need to run assessments
    members = st.session_state.team_data['members']

    if st.session_state.current_assessment_idx < len(members):
        curr_member = members[st.session_state.current_assessment_idx]
        run_assessment_dialog(curr_member['name'], curr_member['role'], len(members))

    st.title(f"📊 {st.session_state.team_data['name']} Health Monitor")

    if st.session_state.current_assessment_idx >= len(members):
        results = st.session_state.temp_assessments

        c1, c2, c3, c4 = st.columns(4)
        states = [r['state'] for r in results]
        stressed_count = states.count("Stressed") + states.count("Fatigued")

        c1.metric("Team Risk Level", "Critical" if stressed_count > 0 else "Optimal", f"{stressed_count} At-Risk")
        c2.metric("Sprint Velocity (Proj.)", "42 pts", "-12 pts" if stressed_count > 0 else "+5 pts")
        c3.metric("Dominant State", max(set(states), key=states.count))
        c4.metric("Active Members", len(members))

        st.divider()

        for res in results:
            with st.expander(f"👤 **{res['name']}** ({res['role']}) - Status: **{res['state']}**", expanded=True):
                col_graph, col_advice = st.columns([2, 1])

                with col_graph:
                    x_axis = np.linspace(0, 10, 100)
                    y_alpha = np.sin(x_axis) * res['eeg_data']['alpha']
                    y_beta = np.sin(x_axis * 3) * res['eeg_data']['beta'] + 2 
                    y_delta = np.sin(x_axis / 2) * res['eeg_data']['delta'] - 2

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_axis, y=y_beta, name='Beta (Stress)', line=dict(color='#FF4B4B')))
                    fig.add_trace(go.Scatter(x=x_axis, y=y_alpha, name='Alpha (Relax)', line=dict(color='#00CC96')))
                    fig.add_trace(go.Scatter(x=x_axis, y=y_delta, name='Delta (Fatigue)', line=dict(color='#636EFA')))

                    fig.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0), template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

                with col_advice:
                    st.info(f"**🤖 AI Agent Advice:**\n\n{res['advice']}")
                    if res['state'] in ["Stressed", "Fatigued"]:
                        st.error("Action Required: High Cognitive Load")
                    else:
                        st.success("Cognitive Load Optimal")
    else:
        st.info("Waiting for assessments to complete...")


# --- APP ROUTER ---
if 'page' not in st.session_state:
    st.session_state.page = "setup"

# Always start with the manager view (which now includes sidebar nav)
render_manager_view()
