import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import joblib

=======
import joblib
import plotly.graph_objects as go
>>>>>>> 93b539d (fixed information)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Performance Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# WARM CREAM UI — Matching reference design
# ============================================================

=======
@st.cache_resource
def load_models():
    try:
        linear_model   = joblib.load("models/linear_model.pkl")
        logistic_model = joblib.load("models/logistic_model.pkl")
        kmeans_model   = joblib.load("models/kmeans_model.pkl")
        scaler_reg     = joblib.load("models/scaler_reg.pkl")
        scaler_clf     = joblib.load("models/scaler_clf.pkl")
        scaler_cluster = joblib.load("models/scaler_cluster.pkl")
        return linear_model, logistic_model, kmeans_model, scaler_reg, scaler_clf, scaler_cluster
    except Exception as e:
        st.error(f"Model Load Error: {str(e)}")
        return None, None, None, None, None, None

# ============================================================
# WARM CREAM THEME
# ============================================================
>>>>>>> 93b539d (fixed information)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600&display=swap');

    /* ── Root palette ── */
    :root {
        --bg:           #f2ece0;
        --sidebar-bg:   #ede6d8;
        --card-bg:      #ede6d8;
        --card-border:  #d9cfbc;
        --accent:       #8b6e4e;
        --accent-light: #c4a882;
        --text-primary: #3a2e22;
        --text-muted:   #8a7b6a;
        --pass-green:   #5a7a5a;
        --fail-red:     #8b3a3a;
        --amber:        #b07a2e;
        --bar-default:  #b8a48a;
        --bar-highlight:#8b6e4e;
    }


    /* ── Global background ── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main,
    .block-container {
        background-color: var(--bg) !important;
        font-family: 'Source Sans 3', sans-serif;
        color: var(--text-primary);
    }

    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--card-border) !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    /* Sidebar title */
    [data-testid="stSidebar"] h1 {
        font-family: 'Roboto Slab', serif !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin-bottom: 1.2rem;
    }

    /* Expander headers in sidebar */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        background: rgba(139, 110, 78, 0.12) !important;
        border-radius: 10px !important;
        border: 1px solid var(--card-border) !important;
        padding: 8px 12px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: transparent !important;
        border: none !important;
    }

    /* Slider track */
    [data-testid="stSlider"] > div > div > div > div {
        background: var(--accent-light) !important;
    }
    [data-testid="stSlider"] > div > div > div > div > div {
        background: var(--accent) !important;
    }

    /* Selectbox, radio, number input */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] > div > div > input {
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }

    /* ── Radio buttons ── */
    [data-testid="stRadio"] label {
        color: var(--text-primary) !important;
    }
    /* Unselected radio circle */
    [data-testid="stRadio"] input[type="radio"] + div,
    [data-testid="stRadio"] span[data-testid="stWidgetLabel"] ~ div label div:first-child {
        background-color: var(--card-bg) !important;
        border-color: var(--accent-light) !important;
    }
    /* All radio SVG circles — force light fill */
    [data-testid="stRadio"] svg circle {
        fill: var(--card-bg) !important;
        stroke: var(--accent-light) !important;
    }
    /* Number input */
    [data-testid="stNumberInput"] > div {
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px !important;
    }
    [data-testid="stNumberInput"] input,
    [data-testid="stNumberInput"] button {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: none !important;
    }
    [data-testid="stNumberInput"] button svg {
        fill: var(--text-primary) !important;
    }

    /* Checkbox */
    [data-testid="stCheckbox"] label {
        color: var(--text-primary) !important;
    }
    [data-testid="stCheckbox"] input[type="checkbox"] + div {
        background-color: var(--card-bg) !important;
        border-color: var(--accent-light) !important;
    }

    /* Sidebar button — Update Insights */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: var(--card-bg) !important;
        border: 1.5px solid var(--card-border) !important;
        color: var(--text-primary) !important;
        border-radius: 12px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 0.55rem 1rem !important;
        margin-top: 1.2rem;
        transition: background 0.2s, border-color 0.2s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(139, 110, 78, 0.15) !important;
        border-color: var(--accent) !important;
    }

    /* ── Page title ── */
    .portal-title {
        font-family: 'Roboto Slab', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .portal-subtitle {
        font-size: 0.95rem;
        color: var(--text-muted);
        margin-bottom: 1.6rem;
    }

    /* ── Section titles ── */
    .section-title {
        font-family: 'Roboto Slab', serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 1.8rem 0 1rem 0;
        padding-left: 14px;
        border-left: 4px solid var(--accent);
        letter-spacing: -0.01em;
    }

    /* ── Metric cards ── */
    .metric-container {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 22px 16px;
        text-align: center;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 10px;
    }
    .metric-value {
        font-family: 'Roboto Slab', serif;
        font-size: clamp(1.4rem, 2.8vw, 2.2rem);
        font-weight: 700;
        line-height: 1.1;
        word-break: break-word;
    }

    /* ── Insight boxes ── */
    .insight-box {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 16px 18px;
        font-size: 0.93rem;
        line-height: 1.55;
        color: var(--text-primary);
    }

    /* ── Caption ── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-size: 0.78rem !important;
    }

    /* ── Plotly chart container backgrounds ── */
    .js-plotly-plot .plotly,
    .js-plotly-plot .plotly .svg-container {
        background: transparent !important;
    }

    /* ── Hide default streamlit menu & footer but keep header for sidebar toggle ── */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { visibility: visible; background: transparent !important; }

    /* ── Sidebar toggle arrow ── */
    [data-testid="collapsedControl"] svg,
    button[kind="header"] svg,
    [data-testid="stSidebarCollapsedControl"] svg {
        fill: #3a2e22 !important;
        color: #3a2e22 !important;
        stroke: #3a2e22 !important;
    }
    /* ── Remove annoying red/blue focus outlines ── */
    * { outline: none !important; }
=======
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-size: 15px !important;
}

#MainMenu, footer, header, [data-testid="stHeader"] { visibility: hidden; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: var(--bg-deep) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.stSidebar button {
    width: 100% !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-align: left !important;
    margin-bottom: 0.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow) !important;
}
.stSidebar button:hover {
    background: var(--accent) !important;
    color: white !important;
    border-color: var(--accent) !important;
    transform: translateX(3px) !important;
}
.stSidebar button:focus, .stSidebar button:active {
    box-shadow: none !important;
    outline: none !important;
}

.stApp { background-color: var(--bg) !important; }
.block-container { padding: 2rem 3rem !important; }

/* SLIDERS */
div[data-baseweb="slider"] > div > div > div {
    background-color: var(--accent) !important;
}
.stSlider [role="slider"] {
    background-color: var(--accent) !important;
    border: 2px solid white !important;
    cursor: pointer !important;
}

/* SELECTBOX */
div[data-baseweb="select"] > div {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
div[data-baseweb="select"] > div:hover { border-color: var(--accent) !important; }
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(176,125,78,0.2) !important;
}
div[data-baseweb="select"] * { color: var(--text) !important; cursor: pointer !important; }
div[data-baseweb="select"] svg { fill: var(--accent) !important; }

.stRadio [data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label[data-testid="stWidgetLabel"] p {
    color: var(--text-dim) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* PRIMARY BUTTON */
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    padding: 0.9rem 1.5rem !important;
    font-size: 0.85rem !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #8d6035 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(176,125,78,0.3) !important;
}

/* SECONDARY BUTTON */
.stButton > button:not([kind="primary"]) {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    transition: all 0.2s !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* CONTAINERS */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem !important;
}

/* CARDS */
.intel-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 1.8rem;
    box-shadow: var(--shadow);
    transition: all 0.2s ease;
    height: 100%;
}
.intel-card:hover {
    box-shadow: 0 8px 32px rgba(100,70,30,0.12);
    transform: translateY(-2px);
}
.card-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
}
.card-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.card-body {
    font-size: 0.88rem;
    color: var(--text-dim);
    line-height: 1.6;
    margin-top: 0.5rem;
}

/* PAGE HEADERS */
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.3rem;
    line-height: 1.15;
}
.page-subtitle {
    font-size: 0.95rem;
    color: var(--text-dim);
    font-weight: 400;
    margin-bottom: 2rem;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text);
    margin: 2.5rem 0 1.2rem 0;
    padding-left: 0.8rem;
    border-left: 3px solid var(--accent);
}

/* HERO */
.hero-strip {
    background: linear-gradient(135deg, #ede0cc 0%, #f5efe4 60%, #e8ddc8 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 3.5rem;
    margin-bottom: 3rem;
    position: relative;
    overflow: hidden;
}
.hero-strip::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(176,125,78,0.15) 0%, transparent 70%);
    border-radius: 50%;
}

/* METRIC ROW */
.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.85rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.92rem;
}
.metric-row:last-child { border-bottom: none; }
.metric-key { color: var(--text-dim); font-weight: 500; }
.metric-val { color: var(--text); font-weight: 700; }

/* CONFUSION TABLE */
.conf-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.conf-table th {
    background: var(--bg-deep);
    color: var(--text-dim);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.8rem 1rem;
    text-align: center;
}
.conf-table td {
    padding: 1rem;
    text-align: center;
    border: 1px solid var(--border);
    font-weight: 600;
    font-size: 1.05rem;
    color: var(--text);
}
.conf-hit  { background: rgba(176,125,78,0.15); }
.conf-miss { background: var(--bg-card); color: var(--text-dim); }

/* RECOMMENDATION CARDS */
.rec-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem;
    font-size: 0.88rem;
    color: var(--text);
    line-height: 1.6;
    margin-bottom: 0.8rem;
    box-shadow: var(--shadow);
}

/* INPUT GROUP HEADER */
.input-group-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

input[type="number"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
>>>>>>> 93b539d (fixed information)
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
@st.cache_resource
def load_models():
    linear_model   = joblib.load('models/linear_model.pkl')
    logistic_model = joblib.load('models/logistic_model.pkl')
    kmeans_model   = joblib.load('models/kmeans_model.pkl')
    scaler_reg     = joblib.load('models/scaler_reg.pkl')
    scaler_clf     = joblib.load('models/scaler_clf.pkl')
    scaler_cluster = joblib.load('models/scaler_cluster.pkl')
    return linear_model, logistic_model, kmeans_model, scaler_reg, scaler_clf, scaler_cluster

linear_model, logistic_model, kmeans_model, scaler_reg, scaler_clf, scaler_cluster = load_models()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.title("Student Profile")

    with st.expander("📝 Academic Baseline", expanded=True):
        math_score = st.slider("Math Score", 0, 100, 65)
        reading_score = st.slider("Reading Score", 0, 100, 68)
        test_prep_label = st.selectbox("Test Prep Status", ["Completed", "Not Completed"])
        test_prep = 0 if test_prep_label == "Completed" else 1

    with st.expander("🕒 Habits & Lifestyle"):
        study_hours_map = {"< 5 hrs": 2.5, "5-10 hrs": 7.5, "> 10 hrs": 12.0}
        study_hours_label = st.selectbox("Weekly Study", list(study_hours_map.keys()), index=1)
        study_hours = study_hours_map[study_hours_label]

        sport_label = st.select_slider("Sport Participation", options=["Never", "Sometimes", "Regularly"], value="Sometimes")
        practice_sport = {"Never": 0, "Sometimes": 1, "Regularly": 2}[sport_label]

    with st.expander("🏠 Demographics"):
        parent_educ_map = {"High School": 1, "Some College": 2, "Associate's": 3, "Bachelor's": 4, "Master's": 5}
        parent_educ_label = st.selectbox("Parent Education", list(parent_educ_map.keys()))
        parent_educ = parent_educ_map[parent_educ_label]

        lunch_label = st.radio("Lunch Type", ["Standard", "Free/Reduced"])
        lunch_type = 1 if lunch_label == "Standard" else 0

        gender_male = 1 if st.radio("Gender", ["Female", "Male"]) == "Male" else 0
        nr_siblings = st.number_input("Siblings", 0, 6, 1)
        is_first_child = 1 if st.checkbox("First Child", value=True) else 0
        transport = 1 if st.radio("Transport", ["Public", "School Bus"]) == "School Bus" else 0

    st.button("Update Insights")
=======
    st.markdown("""
    <div style='padding: 2rem 1rem 1.5rem 1rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem;'>
        <div style='font-family: Playfair Display, serif; font-size: 1.1rem; font-weight: 700; color: var(--text); line-height: 1.3;'>Student Performance<br>Analytics Dashboard</div>
        <div style='font-size: 0.7rem; color: var(--text-dim); margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 0.06em;'>ML Analytics · Milestone 1</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🏠  HOME", use_container_width=True):
        st.session_state.page = "home"
        st.session_state.prediction_run = False
    if st.button("📊  PERFORMANCE", use_container_width=True):
        st.session_state.page = "performance"
    if st.button("🎯  PREDICT", use_container_width=True):
        st.session_state.page = "predict"

    st.markdown("""
    <div style='margin-top: 1.5rem; padding: 1.2rem; background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow);'>
        <div style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: var(--accent); margin-bottom: 0.8rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem;'>Model Performance</div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 600; margin-bottom: 0.4rem;'>Accuracy &nbsp;&nbsp;&nbsp; 91.76%</div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 600; margin-bottom: 0.4rem;'>R² Score &nbsp;&nbsp;&nbsp; 0.9397</div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 600; margin-bottom: 0.4rem;'>F1 Score &nbsp;&nbsp;&nbsp; 0.9176</div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 600;'>MAE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.04 marks</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top: 1rem; padding: 1.2rem; background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow);'>
        <div style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: var(--accent); margin-bottom: 0.8rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem;'>System Info</div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 500; margin-bottom: 0.4rem;'>📂 30,640 records loaded</div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 500; margin-bottom: 0.4rem;'>🤖 3 models active</div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 500;'>⚙️ 11 features enabled</div>
    </div>
    """, unsafe_allow_html=True)
>>>>>>> 93b539d (fixed information)


# ============================================================
# PREDICTIONS (Logic Preserved)
# ============================================================
input_data = np.array([[math_score, reading_score, study_hours, parent_educ, test_prep, lunch_type, practice_sport, nr_siblings, gender_male, is_first_child, transport]])

input_scaled_reg = scaler_reg.transform(input_data)
input_scaled_clf = scaler_clf.transform(input_data)

predicted_exam_score = float(np.clip(linear_model.predict(input_scaled_reg)[0], 0, 100))
predicted_result = logistic_model.predict(input_scaled_clf)[0]
result_proba = logistic_model.predict_proba(input_scaled_clf)[0]

cluster_input = np.array([[predicted_exam_score, study_hours, parent_educ, lunch_type, test_prep, practice_sport]])
cluster_scaled = scaler_cluster.transform(cluster_input)
cluster_id = int(kmeans_model.predict(cluster_scaled)[0])

cluster_centers_exam = kmeans_model.cluster_centers_[:, 0]
sorted_cluster_ids = np.argsort(cluster_centers_exam)
label_list = ["At-Risk", "Average", "High-Performer"]
cluster_label_map = {int(sorted_cluster_ids[i]): label_list[i] for i in range(3)}
learner_category = cluster_label_map[cluster_id]
=======
def home_page():
    st.markdown("""
    <div class='hero-strip'>
        <div style='font-size: 0.7rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; color: var(--accent); margin-bottom: 0.8rem;'>Gen AI Course · Milestone 1 · February 2026</div>
        <div class='page-title'>Intelligent Learning Analytics<br>&amp; Agentic AI Study Coach</div>
        <div style='font-size: 0.98rem; color: var(--text-dim); max-width: 650px; line-height: 1.7; margin-top: 0.6rem;'>
            A complete ML pipeline for student performance analysis — predicting exam scores, classifying Pass/Fail outcomes, and segmenting students into meaningful learner categories using 30,640 student records.
        </div>
        <div style='margin-top: 1.8rem; display: flex; gap: 0.8rem; flex-wrap: wrap;'>
            <div style='background: var(--accent); color: white; padding: 0.4rem 1rem; border-radius: 50px; font-size: 0.78rem; font-weight: 600;'>R² = 0.9397</div>
            <div style='background: var(--accent); color: white; padding: 0.4rem 1rem; border-radius: 50px; font-size: 0.78rem; font-weight: 600;'>Accuracy = 91.76%</div>
            <div style='background: var(--accent); color: white; padding: 0.4rem 1rem; border-radius: 50px; font-size: 0.78rem; font-weight: 600;'>3 Learner Segments</div>
            <div style='background: var(--accent); color: white; padding: 0.4rem 1rem; border-radius: 50px; font-size: 0.78rem; font-weight: 600;'>30,640 Records</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Platform Capabilities</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    caps = [
        ("📈", "Score Prediction",
         "Linear Regression forecasts student ExamScore with R² = 0.9397 and MAE of 3.04 marks from 11 academic and behavioural features."),
        ("✅", "Pass / Fail Classification",
         "Logistic Regression classifies students at 91.76% accuracy with class_weight='balanced' — no SMOTE required."),
        ("🔍", "Learner Segmentation",
         "K-Means (k=3) groups students into At-Risk, Average, and High-Performer clusters based on performance and behaviour."),
        ("⚠️", "Early Intervention",
         "Flags at-risk students before exams using behavioural and academic features, enabling targeted teacher action."),
    ]
    for col, (icon, title, body) in zip([c1, c2, c3, c4], caps):
        with col:
            st.markdown(f"""
            <div class='intel-card'>
                <div style='font-size: 1.6rem; margin-bottom: 0.8rem;'>{icon}</div>
                <div style='font-family: Playfair Display, serif; font-size: 1rem; font-weight: 600; color: var(--text); margin-bottom: 0.5rem;'>{title}</div>
                <div class='card-body'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>System Metrics</div>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    # Training set = 80% of 30,640 = 24,512; Test = 20% = 6,128
    metrics = [
        ("Total Records", "30,640"),
        ("Training Set", "24,512"),
        ("Test Set", "6,128"),
        ("Input Features", "11"),
        ("ML Models", "3"),
    ]
    for col, (lbl, val) in zip([m1, m2, m3, m4, m5], metrics):
        with col:
            st.markdown(f"""
            <div class='intel-card' style='text-align:center; padding: 1.6rem 1rem;'>
                <div class='card-label'>{lbl}</div>
                <div class='card-value' style='font-size: 2rem;'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Key Research Insight</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='intel-card' style='background: linear-gradient(135deg, #faf5ee, #f5ede0); border-left: 4px solid var(--accent);'>
        <div style='font-family: Playfair Display, serif; font-size: 1.08rem; font-weight: 600; color: var(--text); margin-bottom: 0.8rem;'>
            📌 Test Preparation is the Single Biggest Academic Differentiator
        </div>
        <div style='font-size: 0.9rem; color: var(--text-dim); line-height: 1.7; margin-bottom: 1.5rem;'>
            All three learner clusters study approximately the same hours per week (~6.91–6.94 hrs).
            Yet <strong style='color: var(--accent);'>100% of High-Performers completed test prep</strong>
            while 0% of Average students did, and only 8% of At-Risk students did.
            Study time alone does not predict success — <em>how</em> students prepare matters far more.
        </div>
        <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;'>
            <div style='text-align:center; background: rgba(192,88,64,0.06); border-radius: 10px; padding: 1rem; border: 1px solid rgba(192,88,64,0.2);'>
                <div style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #c05840; margin-bottom: 0.3rem;'>At-Risk</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.9rem; font-weight: 700; color: #c05840;'>58.64</div>
                <div style='font-size: 0.75rem; color: var(--text-dim); margin-top: 0.2rem;'>Avg Score · 8% prep</div>
                <div style='font-size: 0.72rem; color: var(--text-dim);'>7,818 students (25.5%)</div>
            </div>
            <div style='text-align:center; background: var(--bg-card); border-radius: 10px; padding: 1rem; border: 1px solid var(--border);'>
                <div style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-dim); margin-bottom: 0.3rem;'>Average</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.9rem; font-weight: 700; color: var(--text);'>68.59</div>
                <div style='font-size: 0.75rem; color: var(--text-dim); margin-top: 0.2rem;'>Avg Score · 0% prep</div>
                <div style='font-size: 0.72rem; color: var(--text-dim);'>13,454 students (43.9%)</div>
            </div>
            <div style='text-align:center; background: rgba(90,138,69,0.07); border-radius: 10px; padding: 1rem; border: 1px solid rgba(90,138,69,0.25);'>
                <div style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #5a8a45; margin-bottom: 0.3rem;'>High-Performer</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.9rem; font-weight: 700; color: #5a8a45;'>76.40</div>
                <div style='font-size: 0.75rem; color: var(--text-dim); margin-top: 0.2rem;'>Avg Score · 100% prep</div>
                <div style='font-size: 0.72rem; color: var(--text-dim);'>9,368 students (30.6%)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Team</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='intel-card' style='padding: 1.5rem 2rem;'>
        <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;'>
            <div>
                <div style='font-size: 0.9rem; color: var(--text); font-weight: 600; margin-bottom: 0.3rem;'>
                    Sathvik Koriginja (2401010231) &nbsp;&middot;&nbsp;
                    Anushka Tyagi (2401010090) &nbsp;&middot;&nbsp;
                    Apoorva Choudhary (2401010092)
                </div>
                <div style='font-size: 0.78rem; color: var(--text-dim);'>Gen AI Course · Milestone 1 · February 2026 · Dataset: Students Exam Scores Extended (Kaggle)</div>
            </div>
            <div style='font-size: 0.78rem; color: var(--text-dim);'>
                pandas · numpy · scikit-learn · plotly · streamlit
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
>>>>>>> 93b539d (fixed information)


# ============================================================
# DASHBOARD MAIN
# ============================================================
st.markdown("<div class='portal-title'>Student Intelligence Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='portal-subtitle'>Predictive success tracking and academic intervention analytics.</div>", unsafe_allow_html=True)

# ── Executive Summary ──────────────────────────────────────
st.markdown("<div class='section-title'>Executive Summary</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""<div class='metric-container'>
        <div class='metric-label'>Predicted Score</div>
        <div class='metric-value' style='color:#8b6e4e'>{predicted_exam_score:.1f}%</div>
    </div>""", unsafe_allow_html=True)

with col2:
    res_color = "#5a7a5a" if predicted_result == "Pass" else "#8b3a3a"
    st.markdown(f"""<div class='metric-container'>
        <div class='metric-label'>Outcome</div>
        <div class='metric-value' style='color:{res_color}'>{predicted_result.upper()}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    cat_color = {"At-Risk": "#8b3a3a", "Average": "#b07a2e", "High-Performer": "#5a7a5a"}[learner_category]
    st.markdown(f"""<div class='metric-container'>
        <div class='metric-label'>Learner Segment</div>
        <div class='metric-value' style='color:{cat_color}'>{learner_category}</div>
    </div>""", unsafe_allow_html=True)

# ── Performance Analytics ──────────────────────────────────

st.markdown("<div class='section-title'>Performance Analytics</div>", unsafe_allow_html=True)

v_col1, v_col2 = st.columns([1.5, 1])

with v_col1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Math', 'Reading', 'Predicted Exam'],
        y=[math_score, reading_score, predicted_exam_score],
        marker_color=['#b8a48a', '#b8a48a', '#8b6e4e'],
        marker_line_color=['#a89070', '#a89070', '#6b4e2e'],
        marker_line_width=1,
        width=0.45
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=20, r=20),
        height=320,
        font=dict(family='Source Sans 3, sans-serif', color='#3a2e22'),
        yaxis=dict(
            range=[0, 105],
            gridcolor='rgba(139,110,78,0.12)',
            tickfont=dict(color='#8a7b6a', size=11),
            zeroline=False
        ),
        xaxis=dict(
            tickfont=dict(color='#8a7b6a', size=12),
            showline=False
        ),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
=======
def performance_page():
    st.markdown("<div class='page-title'>Analysis Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Full model performance report validated with 5-fold cross-validation on 30,640 student records.</div>", unsafe_allow_html=True)

    # Linear Regression
    st.markdown("<div class='section-title'>Linear Regression — ExamScore Prediction</div>", unsafe_allow_html=True)
    lr1, lr2, lr3, lr4 = st.columns(4)
    reg_metrics = [
        ("R² Score",       "0.9397",          "Explains 94% of variance in ExamScore"),
        ("MAE",            "3.04 marks",       "Average prediction error out of 100"),
        ("RMSE",           "3.78 marks",       "Root mean squared error — tight predictions"),
        ("CV R² (5-fold)", "0.9394 ± 0.0021", "Stable across all folds — no overfitting"),
    ]
    for col, (lbl, val, sub) in zip([lr1, lr2, lr3, lr4], reg_metrics):
        with col:
            st.markdown(f"""
            <div class='intel-card' style='text-align:center;'>
                <div class='card-label'>{lbl}</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.65rem; font-weight: 700; color: var(--text); line-height: 1.1; margin: 0.4rem 0;'>{val}</div>
                <div class='card-body' style='font-size: 0.78rem;'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # Logistic Regression
    st.markdown("<div class='section-title'>Logistic Regression — Pass / Fail Classification</div>", unsafe_allow_html=True)
    cl1, cl2, cl3, cl4 = st.columns(4)
    clf_metrics = [
        ("Accuracy",          "91.76%",          "Correctly classifies 92% of students"),
        ("F1 Score (weighted)","0.9176",          "Strong balance of precision and recall"),
        ("CV Accuracy (5-fold)","92.47% ± 0.55%","Highly stable — consistent across folds"),
        ("Fail Recall",       "0.92",             "Catches 92% of all at-risk students"),
    ]
    for col, (lbl, val, sub) in zip([cl1, cl2, cl3, cl4], clf_metrics):
        with col:
            st.markdown(f"""
            <div class='intel-card' style='text-align:center;'>
                <div class='card-label'>{lbl}</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.65rem; font-weight: 700; color: var(--text); line-height: 1.1; margin: 0.4rem 0;'>{val}</div>
                <div class='card-body' style='font-size: 0.78rem;'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Classification Report Detail</div>", unsafe_allow_html=True)
    col_cm, col_rep = st.columns([1, 1.3], gap="large")

    with col_cm:
        # Confusion matrix: TP(Fail)=1120, FN=145, FP=89, TP(Pass)=4746
        st.markdown("""
        <div class='intel-card'>
            <div class='card-label'>Confusion Matrix — Test Set</div>
            <div style='margin-top: 1.2rem;'>
                <table class='conf-table'>
                    <tr>
                        <th></th>
                        <th>Predicted FAIL</th>
                        <th>Predicted PASS</th>
                    </tr>
                    <tr>
                        <th>Actual FAIL</th>
                        <td class='conf-hit'>1,120</td>
                        <td class='conf-miss'>145</td>
                    </tr>
                    <tr>
                        <th>Actual PASS</th>
                        <td class='conf-miss'>89</td>
                        <td class='conf-hit'>4,746</td>
                    </tr>
                </table>
                <div style='margin-top: 1rem; font-size: 0.78rem; color: var(--text-dim); line-height: 1.5;'>
                    Pass/Fail threshold = median ExamScore = <strong>69.0</strong> (data-driven).<br>
                    Highlighted cells = correct predictions.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_rep:
        # Per-class: Fail precision=0.91, recall=0.92 | Pass precision=0.92, recall=0.91 (from PDF)
        st.markdown("""
        <div class='intel-card'>
            <div class='card-label'>Per-Class Performance</div>
            <div style='margin-top: 1.2rem;'>
                <div class='metric-row'>
                    <span class='metric-key' style='font-weight:700; min-width:100px;'>Class</span>
                    <span class='metric-key' style='font-weight:700;'>Precision</span>
                    <span class='metric-key' style='font-weight:700;'>Recall</span>
                    <span class='metric-key' style='font-weight:700;'>Support</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-val' style='min-width:100px;'>Fail</span>
                    <span class='metric-val'>0.91</span>
                    <span class='metric-val'>0.92</span>
                    <span class='metric-val'>1,265</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-val' style='min-width:100px;'>Pass</span>
                    <span class='metric-val'>0.92</span>
                    <span class='metric-val'>0.91</span>
                    <span class='metric-val'>4,835</span>
                </div>
                <div style='background: rgba(176,125,78,0.07); border-radius: 8px; padding: 0.85rem 0; margin-top: 0.5rem; display: flex; justify-content: space-between; align-items: center; padding: 0.85rem;'>
                    <span style='font-weight:700; color:var(--text); min-width:100px;'>Weighted Avg</span>
                    <span style='font-weight:700; color:var(--text);'>0.9177</span>
                    <span style='font-weight:700; color:var(--text);'>0.9176</span>
                    <span style='font-weight:700; color:var(--text);'>6,100</span>
                </div>
                <div style='margin-top: 0.9rem; font-size: 0.78rem; color: var(--text-dim); line-height: 1.5;'>
                    Balanced performance across both classes — the model catches 92% of failing students without relying on synthetic data (SMOTE). Achieved via class_weight='balanced'.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Clustering section
    st.markdown("<div class='section-title'>K-Means Clustering — Learner Segmentation (k=3)</div>", unsafe_allow_html=True)
    ck1, ck2, ck3 = st.columns(3)
    clusters = [
        ("⚠️", "At-Risk",        "#c05840", "7,818 students · 25.5%",
         [("Avg ExamScore", "58.64"), ("Study Hrs/Wk", "6.94"),
          ("Parent Educ", "2.17 / 5"), ("Test Prep Completed", "8%")]),
        ("📘", "Average",        "#7a6a55", "13,454 students · 43.9%",
         [("Avg ExamScore", "68.59"), ("Study Hrs/Wk", "6.91"),
          ("Parent Educ", "2.16 / 5"), ("Test Prep Completed", "0%")]),
        ("🏆", "High-Performer", "#5a8a45", "9,368 students · 30.6%",
         [("Avg ExamScore", "76.40"), ("Study Hrs/Wk", "6.91"),
          ("Parent Educ", "2.20 / 5"), ("Test Prep Completed", "100%")]),
    ]
    for col, (icon, name, color, n, stats) in zip([ck1, ck2, ck3], clusters):
        rows = "".join([
            f"<div style='margin-bottom:0.6rem;'>"
            f"<div style='font-size:0.68rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.08em;'>{k}</div>"
            f"<div style='font-weight:700; color:var(--text); font-size:0.9rem;'>{v}</div>"
            f"</div>"
            for k, v in stats
        ])
        with col:
            st.markdown(f"""
            <div class='intel-card' style='border-top: 4px solid {color};'>
                <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>{icon}</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.1rem; font-weight: 700; color: {color}; margin-bottom: 0.2rem;'>{name}</div>
                <div style='font-size: 0.75rem; color: var(--text-dim); margin-bottom: 1.2rem;'>{n}</div>
                {rows}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Clustering Quality & Design Decision</div>", unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("""
        <div class='intel-card' style='text-align:center;'>
            <div class='card-label'>Silhouette Score</div>
            <div class='card-value'>0.2112</div>
            <div class='card-body' style='margin-top:0.6rem;'>Moderate separation — expected for behavioural data. Students naturally overlap between categories in real life. This is a dataset characteristic, not a modelling error.</div>
        </div>
        """, unsafe_allow_html=True)
    with q2:
        st.markdown("""
        <div class='intel-card' style='text-align:center;'>
            <div class='card-label'>Davies-Bouldin Index</div>
            <div class='card-value'>1.7311</div>
            <div class='card-body' style='margin-top:0.6rem;'>k=3 selected over best_k=5 (silhouette 0.2211 vs 0.2112 — negligible difference). k=3 maps directly to At-Risk / Average / High-Performer for interpretability.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Preprocessing &amp; Validation Methodology</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='intel-card'>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 2.5rem;'>
            <div>
                <div style='font-family: Playfair Display, serif; font-weight: 600; margin-bottom: 0.8rem; color: var(--text); font-size: 1rem;'>Preprocessing Pipeline (10 Steps)</div>
                <div style='font-size: 0.85rem; color: var(--text-dim); line-height: 2;'>
                    ✓ Text standardisation — lowercase, strip whitespace<br>
                    ✓ Merged 'some high school' into 'high_school'<br>
                    ✓ WklyStudyHours &rarr; midpoints: &lt;5&rarr;2.5, 5-10&rarr;7.5, &gt;10&rarr;12.0<br>
                    ✓ Ordinal encoding with manual mapping (no LabelEncoder)<br>
                    ✓ Mode fill for categorical, median fill for numerical<br>
                    ✓ No duplicates found (all 30,640 rows unique)<br>
                    ✓ IQR outlier clipping — NrSiblings: 291 values clipped<br>
                    ✓ Pass/Fail threshold = median ExamScore (69.0)
                </div>
            </div>
            <div>
                <div style='font-family: Playfair Display, serif; font-weight: 600; margin-bottom: 0.8rem; color: var(--text); font-size: 1rem;'>Validation Strategy</div>
                <div style='font-size: 0.85rem; color: var(--text-dim); line-height: 2;'>
                    ✓ 80/20 stratified train/test split (random_state=42)<br>
                    ✓ 5-fold cross-validation on both supervised models<br>
                    ✓ class_weight='balanced' — no SMOTE required<br>
                    ✓ Separate scalers for regression, classification, clustering<br>
                    ✓ StandardScaler fit on train only — no data leakage<br>
                    ✓ ExamScore excluded from X — no circular dependency<br>
                    ✓ All results genuine — no artificial thresholds or synthetic data
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PREDICT PAGE
# ============================================================
def predict_page():
    linear_model, logistic_model, kmeans_model, scaler_reg, scaler_clf, scaler_cluster = load_models()

    if None in [linear_model, logistic_model, scaler_reg, scaler_clf]:
        st.warning("⚠️ Prediction Engine unavailable. Please verify model files in /models/")
        return

    st.markdown("<div class='page-title'>Student Success Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Enter the student's academic and behavioural profile to generate ML-powered predictions.</div>", unsafe_allow_html=True)
>>>>>>> 93b539d (fixed information)

with v_col2:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result_proba[1] * 100,
        title={'text': "Pass Confidence %", 'font': {'size': 14, 'color': '#3a2e22', 'family': 'Source Sans 3'}},
        number={'font': {'size': 52, 'color': '#3a2e22', 'family': 'Roboto Slab'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#8a7b6a', 'tickfont': {'color': '#8a7b6a', 'size': 10}},
            'bar': {'color': "#8b6e4e", 'thickness': 0.25},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50],  'color': 'rgba(139, 58, 58, 0.15)'},
                {'range': [50, 100],'color': 'rgba(90, 122, 90, 0.15)'}
            ],
            'threshold': {
                'line': {'color': '#8b6e4e', 'width': 2},
                'thickness': 0.75,
                'value': result_proba[1] * 100
            }
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=320,
        margin=dict(t=40, b=10, l=30, r=30),
        font=dict(family='Source Sans 3, sans-serif')
    )
    st.plotly_chart(fig_gauge, use_container_width=True)


# ── Strategic Recommendations ──────────────────────────────

st.markdown("<div class='section-title'>Strategic Recommendations</div>", unsafe_allow_html=True)

r_col1, r_col2 = st.columns(2)

with r_col1:
    if learner_category == "At-Risk":
        st.markdown("<div class='insight-box'><b>🚨 Immediate Intervention:</b> Prioritize fundamental review and weekly mock tests.</div>", unsafe_allow_html=True)
    elif learner_category == "Average":
        st.markdown("<div class='insight-box'><b>📈 Growth Plan:</b> Increase study hours to >10 hrs/week to move into High-Performer status.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='insight-box'><b>🏆 Sustainability:</b> Maintain prep consistency and explore advanced-level peer mentoring.</div>", unsafe_allow_html=True)

with r_col2:
    if test_prep == 1:
        st.markdown("<div class='insight-box'><b>💡 Key Insight:</b> Completing a test prep course is the #1 correlated factor for score boosts.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='insight-box'><b>✅ Note:</b> Student has already completed prep courses, which positively weights the prediction.</div>", unsafe_allow_html=True)

st.caption("Milestone 1 Analytics Dashboard | Gen AI Course | February 2026")
=======
    # ---- INPUT PANEL ----
    with col_input:
        st.markdown("""
        <div style='font-family: Playfair Display, serif; font-size: 1.05rem; font-weight: 600;
                    color: var(--text); margin-bottom: 1rem;'>Student Profile</div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div class='input-group-header'>🎓 Academic Baseline</div>", unsafe_allow_html=True)
            math_score    = st.slider("Math Score", 0, 100, 65)
            reading_score = st.slider("Reading Score", 0, 100, 68)
            test_prep_label = st.selectbox(
                "Test Preparation",
                ["Completed", "Not Completed"],
                help="TestPrep_none = 0 (completed) or 1 (not completed)"
            )
            test_prep = 0 if test_prep_label == "Completed" else 1

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div class='input-group-header'>⏱ Weekly Habits</div>", unsafe_allow_html=True)
            # WklyStudyHours converted to midpoints per preprocessing: <5→2.5, 5-10→7.5, >10→12.0
            study_map = {
                "Low — <5 hrs  (midpoint 2.5)":   2.5,
                "Moderate — 5–10 hrs (midpoint 7.5)": 7.5,
                "High — >10 hrs  (midpoint 12.0)":  12.0,
            }
            study_label = st.selectbox("Weekly Study Time", list(study_map.keys()), index=1)
            study_hours = study_map[study_label]

            # PracticeSport ordinal: never=0, sometimes=1, regularly=2
            sport_label    = st.select_slider("Sport Participation", options=["Never", "Sometimes", "Regularly"], value="Sometimes")
            practice_sport = {"Never": 0, "Sometimes": 1, "Regularly": 2}[sport_label]

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div class='input-group-header'>👤 Demographics &amp; Background</div>", unsafe_allow_html=True)

            # ParentEduc ordinal: high_school=1, some_college=2, associates=3, bachelors=4, masters=5
            educ_map = {
                "High School":          1,
                "Some College":         2,
                "Associate's Degree":   3,
                "Bachelor's Degree":    4,
                "Master's Degree":      5,
            }
            educ_label  = st.selectbox("Parent Education Level", list(educ_map.keys()))
            parent_educ = educ_map[educ_label]

            c1, c2 = st.columns(2)
            with c1:
                lunch_label = st.radio("Lunch Type", ["Standard", "Reduced/Free"], horizontal=True)
                lunch_type  = 1 if lunch_label == "Standard" else 0
            with c2:
                gender_label = st.radio("Gender", ["Female", "Male"], horizontal=True)
                gender_male  = 1 if gender_label == "Male" else 0

            c3, c4 = st.columns(2)
            with c3:
                first_child_label = st.radio("First Child?", ["Yes", "No"], horizontal=True)
                is_first_child    = 1 if first_child_label == "Yes" else 0
            with c4:
                transport_label = st.radio("Transport", ["School Bus", "Public"], horizontal=True)
                transport_bus   = 1 if transport_label == "School Bus" else 0

            nr_siblings = st.number_input("Number of Siblings", min_value=0, max_value=6, value=1)

        st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
        if st.button("🎯  UPDATE INSIGHTS", type="primary", use_container_width=True):
            st.session_state.prediction_run = True

    # ---- RESULTS PANEL ----
    with col_result:
        if not st.session_state.prediction_run:
            st.markdown("""
            <div style='background: var(--bg-card); border: 1.5px dashed var(--border);
                        border-radius: var(--radius); height: 880px; display: flex;
                        flex-direction: column; align-items: center; justify-content: center;
                        text-align: center; padding: 4rem;'>
                <div style='font-size: 3rem; opacity: 0.12; margin-bottom: 1.2rem;'>📊</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.2rem;
                            font-weight: 600; color: var(--text-dim);'>Ready for Input</div>
                <div style='font-size: 0.88rem; color: var(--text-dim); margin-top: 0.8rem;
                            max-width: 260px; line-height: 1.6; opacity: 0.7;'>
                    Fill in the student profile on the left and click
                    <strong>Update Insights</strong> to generate predictions.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Feature order (11 features) per project context:
            # MathScore, ReadingScore, WklyStudyHours, ParentEduc,
            # TestPrep_none, LunchType_standard, PracticeSport, NrSiblings,
            # Gender_male, IsFirstChild_yes, TransportMeans_school_bus
            input_data = np.array([[
                math_score, reading_score, study_hours, parent_educ,
                test_prep,  lunch_type,    practice_sport, nr_siblings,
                gender_male, is_first_child, transport_bus
            ]])

            input_reg = scaler_reg.transform(input_data)
            input_clf = scaler_clf.transform(input_data)

            predicted_score  = float(linear_model.predict(input_reg)[0])
            predicted_result = logistic_model.predict(input_clf)[0]
            proba            = logistic_model.predict_proba(input_clf)[0]
            # logistic_model.classes_ = ['Fail', 'Pass']  index 0=Fail, 1=Pass
            fail_prob = proba[0] * 100
            pass_prob = proba[1] * 100
            conf_val  = pass_prob if predicted_result == "Pass" else fail_prob

            # Clustering features: ExamScore, WklyStudyHours, ParentEduc,
            # LunchType_standard, TestPrep_none, PracticeSport
            cluster_input = np.array([[
                predicted_score, study_hours, parent_educ,
                lunch_type, test_prep, practice_sport
            ]])
            # Initial Determination via KMeans (Primary)
            if scaler_cluster and kmeans_model:
                cluster_scaled = scaler_cluster.transform(cluster_input)
                raw_label      = kmeans_model.predict(cluster_scaled)[0]
                centers        = kmeans_model.cluster_centers_
                order          = np.argsort(centers[:, 0])  # sort by ExamScore (index 0)
                name_map       = {order[0]: "At-Risk", order[1]: "Average", order[2]: "High-Performer"}
                learner_seg    = name_map[raw_label]
            else:
                # Score-based fallback using cluster means from report
                if predicted_score < 63.6:
                    learner_seg = "At-Risk"
                elif predicted_score < 72.5:
                    learner_seg = "Average"
                else:
                    learner_seg = "High-Performer"

            # ── Logic Alignment Overrides ──
            # Apply intuitive overrides to ensure segment follows predicted score
            if predicted_score < 60:
                learner_seg = "At-Risk"
            elif predicted_score > 82:
                learner_seg = "High-Performer"
            else:
                # In the 60-82 range, we refine based on specific thresholds
                if learner_seg == "High-Performer" and predicted_score < 76:
                    learner_seg = "Average"
                if learner_seg == "At-Risk" and predicted_score > 65:
                    learner_seg = "Average"

            is_pass      = predicted_result == "Pass"
            result_color = "#5a8a45" if is_pass else "#c05840"
            seg_colors   = {"At-Risk": "#c05840", "Average": "#7a6a55", "High-Performer": "#5a8a45"}
            seg_icons    = {"At-Risk": "⚠️", "Average": "📘", "High-Performer": "🏆"}
            seg_color    = seg_colors.get(learner_seg, "#7a6a55")

            # ── Executive Summary ──
            st.markdown("<div class='section-title' style='margin-top:0;'>Executive Summary</div>", unsafe_allow_html=True)
            e1, e2, e3 = st.columns(3)
            with e1:
                st.markdown(f"""
                <div class='intel-card' style='text-align:center; border-top: 3px solid var(--accent); padding: 1.4rem 1rem;'>
                    <div class='card-label'>Predicted Score</div>
                    <div style='font-family:Playfair Display,serif; font-size:2.3rem; font-weight:700; color:var(--accent);'>{predicted_score:.1f}%</div>
                    <div style='font-size:0.72rem; color:var(--text-dim); margin-top:0.2rem;'>Threshold: 69.0</div>
                </div>
                """, unsafe_allow_html=True)
            with e2:
                st.markdown(f"""
                <div class='intel-card' style='text-align:center; border-top: 3px solid {result_color}; padding: 1.4rem 1rem;'>
                    <div class='card-label'>Outcome</div>
                    <div style='font-family:Playfair Display,serif; font-size:2.3rem; font-weight:700; color:{result_color};'>{predicted_result.upper()}</div>
                    <div style='font-size:0.72rem; color:var(--text-dim); margin-top:0.2rem;'>Confidence: {conf_val:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with e3:
                st.markdown(f"""
                <div class='intel-card' style='text-align:center; border-top: 3px solid {seg_color}; padding: 1.4rem 1rem;'>
                    <div class='card-label'>Learner Segment</div>
                    <div style='font-size:1.5rem; margin:0.2rem 0;'>{seg_icons.get(learner_seg, "📊")}</div>
                    <div style='font-family:Playfair Display,serif; font-size:1.1rem; font-weight:700; color:{seg_color};'>{learner_seg}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Performance Analytics ──
            st.markdown("<div class='section-title'>Performance Analytics</div>", unsafe_allow_html=True)
            chart_col, gauge_col = st.columns([1.2, 1])

            with chart_col:
                fig_bar = go.Figure(go.Bar(
                    x=["Math", "Reading", "Predicted Exam"],
                    y=[math_score, reading_score, predicted_score],
                    marker=dict(
                        color=["#d4c4b0", "#c4b8a0", "#b07d4e"],
                        line=dict(color="white", width=1.5)
                    ),
                    width=0.45,
                    text=[f"{math_score}", f"{reading_score}", f"{predicted_score:.1f}"],
                    textposition="outside",
                    textfont=dict(color="#7a6a55", size=11, family="DM Sans")
                ))
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, b=5, l=5, r=5),
                    height=215,
                    yaxis=dict(range=[0, 118],
                               gridcolor="rgba(100,80,50,0.07)",
                               tickfont=dict(color="#7a6a55", size=10)),
                    xaxis=dict(tickfont=dict(color="#7a6a55", size=11)),
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with gauge_col:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pass_prob,
                    title=dict(text="Pass Confidence %",
                               font=dict(color="#7a6a55", size=11, family="DM Sans")),
                    number=dict(font=dict(color="#2c2416", size=30, family="Playfair Display"),
                                suffix="%"),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickfont=dict(color="#7a6a55", size=8)),
                        bar=dict(color="#b07d4e", thickness=0.25),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0,  50], color="rgba(192,88,64,0.10)"),
                            dict(range=[50, 75], color="rgba(176,125,78,0.10)"),
                            dict(range=[75,100], color="rgba(90,138,69,0.10)"),
                        ],
                        threshold=dict(line=dict(color="#b07d4e", width=2),
                                       thickness=0.75, value=50)
                    )
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=30, b=5, l=15, r=15),
                    height=215,
                    font=dict(family="DM Sans")
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Strategic Recommendations ──
            st.markdown("<div class='section-title'>Strategic Recommendations</div>", unsafe_allow_html=True)

            recs = []

            # Test prep — highest impact insight from clustering analysis
            if test_prep == 1:
                recs.append(("📝", "Enroll in Test Preparation",
                    "Student has not completed test prep. Clustering shows 100% of High-Performers "
                    "completed it vs 0% of Average students — this is the single most actionable "
                    "improvement available."))
            else:
                recs.append(("🏆", "Sustain & Advance",
                    "Student has completed test preparation — the strongest differentiator in the dataset. "
                    "Maintain consistency and explore advanced peer mentoring or enrichment programmes."))

            # Study hours
            if study_hours < 7.5:
                recs.append(("⏱", "Increase Weekly Study Hours",
                    "Student is in the Low study category (<5 hrs/wk). Moving to Moderate (5–10 hrs/wk) "
                    "is consistently associated with better outcomes across all learner segments."))
            else:
                short_label = study_label.split("—")[0].strip()
                recs.append(("✅", "Study Commitment On Track",
                    f"Study time ({short_label}) is well-positioned. Focus on quality of preparation "
                    "rather than adding more hours."))

            # Early intervention
            if predicted_score < 69.0:
                recs.append(("⚠️", "Early Intervention Required",
                    f"Predicted score ({predicted_score:.1f}) is below the Pass threshold (69.0). "
                    "Targeted academic support before the exam is recommended."))

            # Segment-specific guidance
            if learner_seg == "At-Risk":
                recs.append(("📚", "At-Risk Study Plan",
                    "Revise fundamentals daily, identify and address weak subject areas, and prioritise "
                    "consistent daily practice over long infrequent sessions."))
            elif learner_seg == "Average":
                recs.append(("📖", "Move Toward High-Performer",
                    "Practice moderate-to-advanced problems and attempt weekly mock tests. "
                    "Completing test preparation is the clearest path to the High-Performer cluster."))

            for icon, title, body in recs:
                st.markdown(f"""
                <div class='rec-card'>
                    <strong>{icon} {title}</strong><br>
                    <span style='color: var(--text-dim);'>{body}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            if st.button("↩  Reset Predictor", key="reset"):
                st.session_state.prediction_run = False
                st.rerun()


# ============================================================
# ROUTING
# ============================================================
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "performance":
    performance_page()
elif st.session_state.page == "predict":
    predict_page()
>>>>>>> 93b539d (fixed information)
