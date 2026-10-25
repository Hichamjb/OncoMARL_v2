import os
import ast
import html
import pandas as pd
import streamlit as st
import plotly.express as px

from demo import OncoMARLInferenceEngine


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="OncoMARL v2 | Clinical AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL CSS
# =========================================================

st.markdown("""
<style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F4F7FB;
    }

    [data-testid="stSidebar"] {
        background-color: #10243A;
    }

    [data-testid="stSidebar"] * {
        color: #F1F5F9;
    }

    [data-testid="stSidebar"] input {
        color: #10243A;
        background-color: #FFFFFF;
    }

    .hero {
        background: linear-gradient(120deg, #10243A, #174A68);
        padding: 30px 34px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 30px;
        font-weight: 800;
        margin: 0 0 10px 0;
        color: white;
    }

    .hero p {
        font-size: 14px;
        color: #D4E5F2;
        margin: 0;
        line-height: 1.7;
    }

    .section-title {
        color: #10243A;
        font-size: 21px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 16px;
    }

    .metric-card {
        background: white;
        border: 1px solid #E4EAF1;
        border-radius: 14px;
        padding: 20px;
        min-height: 120px;
        box-shadow: 0 3px 12px rgba(16, 36, 58, 0.035);
    }

    .metric-label {
        color: #64748B;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .metric-value {
        color: #10243A;
        font-size: 27px;
        font-weight: 800;
    }

    .metric-caption {
        color: #64748B;
        font-size: 11px;
        margin-top: 8px;
    }

    .agent-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        min-height: 150px;
    }

    .agent-label {
        font-size: 12px;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .agent-vote {
        font-size: 16px;
        font-weight: 700;
        color: #174A68;
        line-height: 1.6;
    }

    .decision-card {
        background: white;
        border: 1px solid #D9E6F0;
        border-left: 5px solid #0D9488;
        border-radius: 14px;
        padding: 24px;
        margin-top: 10px;
    }

    .decision-label {
        color: #64748B;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .decision-value {
        color: #10243A;
        font-size: 23px;
        font-weight: 800;
        line-height: 1.5;
    }

    .status-pill {
        display: inline-block;
        background: #DDF7EF;
        color: #087F5B;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin-top: 12px;
    }

    .footer {
        text-align: center;
        color: #94A3B8;
        font-size: 11px;
        padding: 30px 0 10px 0;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 15px;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
    }

    div[data-testid="stMetricValue"] {
        color: #10243A;
    }

    .stButton button {
        border-radius: 9px;
        font-weight: 600;
    }

    hr {
        border-color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL AND DATA LOADING
# =========================================================

@st.cache_resource
def load_model():
    return OncoMARLInferenceEngine(
        "checkpoints/marl_model.pkl"
    )


@st.cache_data
def load_data():
    path = "data/processed/test_patients.csv"

    if os.path.exists(path):
        return pd.read_csv(path)

    return None


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_patient_state(df, patient_idx):
    """Extract a six-dimensional state vector."""

    raw_state = df.iloc[patient_idx]["state"]

    if isinstance(raw_state, str):
        raw_state = ast.literal_eval(raw_state)

    state = [float(value) for value in raw_state]

    if len(state) != 6:
        raise ValueError(
            f"Expected 6 state variables, got {len(state)}."
        )

    return state


def get_action_name(engine, action_id):
    """Return a safe, readable action label."""

    action_id = int(action_id)

    return engine.action_names.get(
        action_id,
        f"Unknown action ({action_id})"
    )


def metric_card(label, value, caption):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {html.escape(str(label))}
            </div>
            <div class="metric-value">
                {html.escape(str(value))}
            </div>
            <div class="metric-caption">
                {html.escape(str(caption))}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>OncoMARL v2</h1>
    <p>
        Multi-Agent Reinforcement Learning for Oncology Research
        <br>
        Interactive Clinical State Analysis & Decision Support
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD RESOURCES
# =========================================================

try:
    engine = load_model()
except Exception as exc:
    st.error(f"Unable to initialize the inference engine: {exc}")
    st.stop()

if not engine.is_loaded:
    st.error(
        "Model not found. Check checkpoints/marl_model.pkl."
    )
    st.stop()

df = load_data()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## OncoMARL")
st.sidebar.caption("Clinical AI Research Platform")
st.sidebar.divider()

st.sidebar.markdown("### Data Configuration")

modes = ["Custom Clinical State"]

if df is not None and not df.empty:
    modes.insert(0, "Test Patient")

mode = st.sidebar.radio(
    "Input source",
    modes
)

default_state = [
    0.46995,
    0.29196,
    0.35691,
    0.0,
    0.0,
    -1.0
]

patient_idx = 0

if mode == "Test Patient":
    patient_idx = st.sidebar.number_input(
        "Test patient index",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    try:
        default_state = get_patient_state(
            df, int(patient_idx)
        )
    except Exception as exc:
        st.sidebar.error(f"Invalid patient state: {exc}")
        st.stop()

st.sidebar.divider()

st.sidebar.caption(
    "Research prototype — not a substitute for "
    "professional clinical judgment."
)


# =========================================================
# CLINICAL STATE INPUTS
# =========================================================

st.markdown(
    '<div class="section-title">Clinical State Parameters</div>',
    unsafe_allow_html=True
)

st.caption(
    "Configure the six state variables used by the "
    "multi-agent inference engine."
)

col1, col2, col3 = st.columns(3)

with col1:
    v1_tumor = st.number_input(
        "Tumor Burden",
        min_value=0.0,
        max_value=1.0,
        value=float(default_state[0]),
        step=0.01,
        format="%.4f",
        help="Normalized tumor burden."
    )

    v2_tox = st.number_input(
        "Treatment Toxicity",
        min_value=0.0,
        max_value=1.0,
        value=float(default_state[1]),
        step=0.01,
        format="%.4f",
        help="Normalized toxicity value."
    )

with col2:
    v3_unc = st.number_input(
        "Clinical Uncertainty",
        min_value=0.0,
        max_value=1.0,
        value=float(default_state[2]),
        step=0.01,
        format="%.4f",
        help="Normalized uncertainty value."
    )

    v4_prog = st.selectbox(
        "Disease Progression",
        options=[0.0, 1.0],
        index=int(default_state[3]),
        format_func=lambda x: (
            "Absent (0)" if x == 0.0
            else "Present (1)"
        )
    )

with col3:
    v5_visit = st.number_input(
        "Normalized Visit Index",
        min_value=0.0,
        max_value=1.0,
        value=float(default_state[4]),
        step=0.0833,
        format="%.4f"
    )

    prev_act_options = [-1.0, 0.0, 1.0, 2.0, 3.0]

    current_prev = float(default_state[5])

    default_index = (
        prev_act_options.index(current_prev)
        if current_prev in prev_act_options
        else 0
    )

    v6_prev_action = st.selectbox(
        "Previous Action",
        options=prev_act_options,
        index=default_index,
        format_func=lambda x: (
            "None (-1)" if x == -1.0
            else f"{int(x)} — "
            f"{get_action_name(engine, int(x))}"
        )
    )


# =========================================================
# STATE VECTOR
# =========================================================

current_state = [
    v1_tumor,
    v2_tox,
    v3_unc,
    v4_prog,
    v5_visit,
    v6_prev_action
]

with st.expander("Inspect input state vector"):
    st.json({
        "tumor_burden": v1_tumor,
        "toxicity": v2_tox,
        "uncertainty": v3_unc,
        "disease_progression": v4_prog,
        "visit_index": v5_visit,
        "previous_action": v6_prev_action
    })


# =========================================================
# CLINICAL METRICS
# =========================================================

st.markdown(
    '<div class="section-title">State Overview</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    metric_card(
        "TUMOR BURDEN",
        f"{v1_tumor:.1%}",
        "Normalized model input"
    )

with m2:
    metric_card(
        "TOXICITY",
        f"{v2_tox:.1%}",
        "Normalized model input"
    )

with m3:
    metric_card(
        "UNCERTAINTY",
        f"{v3_unc:.1%}",
        "Normalized model input"
    )

with m4:
    metric_card(
        "PROGRESSION",
        "Present" if v4_prog == 1.0 else "Absent",
        "Encoded disease state"
    )


# =========================================================
# MULTI-AGENT INFERENCE
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">Multi-Agent Deliberation</div>',
    unsafe_allow_html=True
)

try:
    result = engine.predict(current_state)

    agent_actions = result["agent_actions"]

    if len(agent_actions) != 3:
        st.warning(
            "The interface expects three agents, but the "
            "inference engine returned a different number."
        )

    agent_cols = st.columns(len(agent_actions))

    for i, action_id in enumerate(agent_actions):
        with agent_cols[i]:
            action_name = get_action_name(
                engine, action_id
            )

            st.markdown(
                f"""
                <div class="agent-card">
                    <div class="agent-label">
                        AGENT {i + 1}
                    </div>
                    <div class="agent-vote">
                        {html.escape(action_name)}
                    </div>
                    <p style="
                        color:#64748B;
                        font-size:12px;
                        margin-top:12px;
                    ">
                        Action ID: {int(action_id)}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

except Exception as exc:
    st.error(f"Inference failed: {exc}")
    st.stop()


# =========================================================
# VOTING VISUALIZATION
# =========================================================

st.markdown(
    '<div class="section-title">Agent Voting Distribution</div>',
    unsafe_allow_html=True
)

vote_data = pd.DataFrame({
    "Agent": [
        f"Agent {i + 1}"
        for i in range(len(agent_actions))
    ],
    "Action": [
        get_action_name(engine, action)
        for action in agent_actions
    ]
})

fig = px.histogram(
    vote_data,
    x="Action",
    color="Action",
    category_orders={
        "Action": list(dict.fromkeys(vote_data["Action"]))
    },
    labels={
        "Action": "Selected action",
        "count": "Number of votes"
    },
    title="Distribution of agent decisions"
)

fig.update_layout(
    template="plotly_white",
    showlegend=False,
    margin=dict(l=15, r=15, t=60, b=15),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="Inter, sans-serif",
        color="#10243A"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# FINAL COORDINATOR DECISION
# =========================================================

st.markdown(
    '<div class="section-title">Coordinator Decision</div>',
    unsafe_allow_html=True
)

decision_name = result["voted_action_name"]
decision_id = result["voted_action_id"]
is_unanimous = result["is_unanimous"]

status = (
    "Unanimous agreement"
    if is_unanimous
    else "Majority decision"
)

safe_decision = html.escape(str(decision_name))
safe_status = html.escape(status)

st.markdown(
    f"""
    <div class="decision-card">
        <div class="decision-label">
            FINAL COORDINATOR OUTPUT
        </div>
        <div class="decision-value">
            {safe_decision}
        </div>
        <div style="
            color:#64748B;
            font-size:13px;
            margin-top:8px;
        ">
            Action ID: {int(decision_id)}
        </div>
        <span class="status-pill">
            {safe_status}
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "The displayed action represents the trained model's output "
    "for the supplied state. Agreement between agents does not "
    "establish clinical validity or treatment suitability."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    OncoMARL v2 · Multi-Agent Reinforcement Learning Research
    <br>
    Experimental decision-support prototype · Not for clinical use
</div>
""", unsafe_allow_html=True)