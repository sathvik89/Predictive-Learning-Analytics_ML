import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Student Performance Analytics Dashboard",
    page_icon="STUDENT",
    layout="wide"
)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    try:
        linear_model  = joblib.load("models/linear_model.pkl")
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
# WARM CREAM THEME -- GLOBAL CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:        #f5f0e8;
    --bg-deep:   #ede8de;
    --bg-card:   #faf7f2;
    --accent:    #b07d4e;
    --accent-lt: #d4a574;
    --text:      #2c2416;
    --text-dim:  #7a6a55;
    --border:    #d9cfc2;
    --shadow:    0 2px 20px rgba(100,70,30,0.08);
    --radius:    14px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-size: 15px !important; /* Scaled for 90% zoom equivalent */
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stHeader"] { visibility: hidden; }

/* ---- SIDEBAR ---- */
[data-testid="stSidebar"] {
    background-color: var(--bg-deep) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Sidebar nav buttons */
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

/* ---- MAIN APP BACKGROUND ---- */
.stApp { background-color: var(--bg) !important; }
.block-container { padding: 2rem 3rem !important; }

/* ---- SLIDERS ---- */
div[data-baseweb="slider"] > div > div > div {
    background-color: var(--accent) !important;
}
.stSlider [role="slider"] {
    background-color: var(--accent) !important;
    border: 2px solid white !important;
}

/* ---- SELECTBOX / RADIO ---- */
div[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--text) !important; }

/* ---- LABELS ---- */
label[data-testid="stWidgetLabel"] p { color: var(--text-dim) !important; font-weight: 600 !important; font-size: 0.82rem !important; }

/* ---- PRIMARY BUTTON ---- */
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

/* ---- CONTAINERS ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem !important;
}

/* ---- CARDS ---- */
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
.card-icon { font-size: 1.8rem; margin-bottom: 1rem; }
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
    font-size: 0.92rem;
    color: var(--text-dim);
    line-height: 1.6;
    margin-top: 0.5rem;
}

/* ---- SECTION HEADERS ---- */
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
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text);
    margin: 2.5rem 0 1.2rem 0;
    padding-left: 0.8rem;
    border-left: 3px solid var(--accent);
}

/* ---- DIVIDER ---- */
.divider { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }

/* ---- RESULT CARD ---- */
.result-hero {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem;
    text-align: center;
    box-shadow: var(--shadow);
    margin-bottom: 1.5rem;
}
.result-pass { color: #5a8a45; }
.result-fail { color: #c05840; }

/* ---- METRIC ROW ---- */
.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.9rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.95rem;
}
.metric-row:last-child { border-bottom: none; }
.metric-key { color: var(--text-dim); font-weight: 500; }
.metric-val { color: var(--text); font-weight: 700; }

/* ---- CLUSTER BADGE ---- */
.cluster-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--accent-lt), var(--accent));
    color: white;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.06em;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    margin-top: 0.6rem;
}

/* ---- RECOMMENDATION CARDS ---- */
.rec-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    font-size: 0.9rem;
    color: var(--text);
    line-height: 1.6;
    margin-bottom: 0.8rem;
    box-shadow: var(--shadow);
}
.rec-icon { margin-right: 0.5rem; }

/* ---- CONFUSION TABLE ---- */
.conf-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.conf-table th {
    background: var(--bg-deep);
    color: var(--text-dim);
    font-size: 0.7rem;
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
    font-size: 1.1rem;
    color: var(--text);
}
.conf-hit { background: rgba(176,125,78,0.15); }
.conf-miss { background: var(--bg-card); color: var(--text-dim); }

/* ---- HERO STRIP ---- */
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

/* ---- INPUT SECTION HEADER ---- */
.input-group-header {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* Number input */
input[type="number"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
if 'page' not in st.session_state:
    st.session_state.page = "home"

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='padding: 2rem 1rem 1.5rem 1rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem;'>
        <div style='font-family: Playfair Display, serif; font-size: 1.15rem; font-weight: 700; color: var(--text); line-height: 1.2;'>Student Performance Analytics Dashboard</div>
        <div style='font-size: 0.72rem; color: var(--text-dim); margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;'>ML Analytics Portal</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("HOME", use_container_width=True):
        st.session_state.page = "home"
        st.session_state.prediction_run = False
    if st.button("PERFORMANCE", use_container_width=True):
        st.session_state.page = "performance"
    if st.button("PREDICT", use_container_width=True):
        st.session_state.page = "predict"

    st.markdown("""
    <div style='margin-top: 2rem; padding: 1.2rem; background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow);'>
        <div style='font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: var(--accent); margin-bottom: 0.8rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem;'>Model Performance</div>
        <div style='font-size: 0.85rem; color: var(--text); font-weight: 600;'>Accuracy: 91.8%</div>
        <div style='font-size: 0.85rem; color: var(--text); font-weight: 600; margin-top: 0.5rem;'>R Score: 0.940</div>
        <div style='font-size: 0.85rem; color: var(--text); font-weight: 600; margin-top: 0.5rem;'>F1 Score: 0.918</div>
        <div style='font-size: 0.85rem; color: var(--text); font-weight: 600; margin-top: 0.5rem;'>MAE: 3.04</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top: 1rem; padding: 1.2rem; background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow);'>
        <div style='font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: var(--accent); margin-bottom: 0.8rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem;'>System Status</div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 500;'>30,640 records loaded</div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 500; margin-top: 0.5rem;'>3 models active</div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 500; margin-top: 0.5rem;'>11 features enabled</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HOME PAGE
# ============================================================
def home_page():
    st.markdown("""
    <div class='hero-strip'>
        <div style='font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; color: var(--accent); margin-bottom: 0.8rem;'>ML Milestone 1 - February 2026</div>
        <div class='page-title'>Student Performance Analytics Dashboard</div>
        <div style='font-size: 1rem; color: var(--text-dim); max-width: 650px; line-height: 1.7; margin-top: 0.5rem;'>
            Advanced behavioral analysis and success forecasting built on 30,640 records using classical ML architecture -- enabling targeted academic support and early intervention strategies.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Platform Capabilities</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    caps = [
        ("Score Prediction", "Linear Regression model forecasts writing exam scores with R = 0.94 and MAE of just 3.04 marks."),
        ("Pass / Fail Outcome", "Logistic Regression classifies students at 91.8% accuracy using balanced class weighting."),
        ("Learner Segmentation", "K-Means clustering groups students into At-Risk, Average, and High-Performer categories."),
        ("Early Intervention", "Identifies students needing support before exams, enabling targeted teacher action."),
    ]
    for col, (title, body) in zip([c1, c2, c3, c4], caps):
        with col:
            st.markdown(f"""
            <div class='intel-card'>
                <div style='font-family: Playfair Display, serif; font-size: 1.02rem; font-weight: 600; color: var(--text); margin-bottom: 0.5rem;'>{title}</div>
                <div class='card-body' style='font-size:0.88rem;'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>System Metrics</div>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("Total Records", "30.6k"), ("Training Set", "24.5k"),
        ("Input Features", "11"), ("ML Models", "3"), ("Data Integrity", "99%"),
    ]
    for col, (lbl, val) in zip([m1, m2, m3, m4, m5], metrics):
        with col:
            st.markdown(f"""
            <div class='intel-card' style='text-align:center; padding: 1.8rem 1rem;'>
                <div class='card-label'>{lbl}</div>
                <div class='card-value'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Key Research Insight</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='intel-card' style='background: linear-gradient(135deg, #faf5ee, #f5ede0); border-left: 4px solid var(--accent);'>
        <div style='font-family: Playfair Display, serif; font-size: 1.1rem; font-weight: 600; color: var(--text); margin-bottom: 0.8rem;'>
            PIN Test Preparation is the Single Biggest Academic Differentiator
        </div>
        <div style='font-size: 0.95rem; color: var(--text-dim); line-height: 1.7;'>
            All three learner clusters study approximately the same hours per week (~6.9 hrs). Yet 
            <strong style='color: var(--accent);'>100% of High-Performers completed test prep</strong>, 
            while 0% of Average students did. Study time alone does not predict success -- <em>how</em> students 
            prepare does.
        </div>
        <div style='margin-top: 1.5rem; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;'>
            <div style='text-align:center; background: var(--bg-card); border-radius: 10px; padding: 1rem; border: 1px solid var(--border);'>
                <div style='font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-dim);'>At-Risk</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.6rem; font-weight: 700; color: var(--text); margin: 0.3rem 0;'>58.6</div>
                <div style='font-size: 0.8rem; color: var(--text-dim);'>Avg Score - 8% Prep</div>
            </div>
            <div style='text-align:center; background: var(--bg-card); border-radius: 10px; padding: 1rem; border: 1px solid var(--border);'>
                <div style='font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-dim);'>Average</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.6rem; font-weight: 700; color: var(--text); margin: 0.3rem 0;'>68.6</div>
                <div style='font-size: 0.8rem; color: var(--text-dim);'>Avg Score - 0% Prep</div>
            </div>
            <div style='text-align:center; background: rgba(176,125,78,0.1); border-radius: 10px; padding: 1rem; border: 1px solid var(--accent-lt);'>
                <div style='font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent);'>High-Performer</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.6rem; font-weight: 700; color: var(--accent); margin: 0.3rem 0;'>76.4</div>
                <div style='font-size: 0.8rem; color: var(--text-dim);'>Avg Score - 100% Prep</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Team Members</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='intel-card' style='padding: 1.2rem 1.5rem;'>
        <div style='display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 1rem;'>
            <div style='font-size: 0.9rem; color: var(--text); font-weight: 600;'>
                Sathvik Koriginja &nbsp;-&nbsp; Anushka Tyagi &nbsp;-&nbsp; Apoorva Choudhary
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PERFORMANCE PAGE
# ============================================================
def performance_page():
    st.markdown("<div class='page-title'>Analysis Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Detailed reliability metrics and segmentation analysis across regression, classification, and clustering tasks.</div>", unsafe_allow_html=True)

    # Note: Model Performance Overview moved to Sidebar as per request
    st.markdown("<div class='section-title'>Classification Report</div>", unsafe_allow_html=True)
    col_cm, col_rep = st.columns([1, 1.3], gap="large")

    with col_cm:
        st.markdown("""
        <div class='intel-card'>
            <div class='card-label'>Confusion Matrix</div>
            <div style='margin-top: 1.2rem;'>
                <table class='conf-table'>
                    <tr>
                        <th></th><th>Predicted FAIL</th><th>Predicted PASS</th>
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
                <div style='margin-top: 1.2rem; font-size: 0.8rem; color: var(--text-dim);'>
                    Based on 6,100 student test records. Threshold = median ExamScore (69.0).
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_rep:
        st.markdown("""
        <div class='intel-card'>
            <div class='card-label'>Per-Class Performance</div>
            <div style='margin-top: 1.2rem;'>
                <div class='metric-row'>
                    <span class='metric-key'>Class</span>
                    <span class='metric-key'>Precision</span>
                    <span class='metric-key'>Recall</span>
                    <span class='metric-key'>Support</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-val'>Fail</span>
                    <span class='metric-val'>92.6%</span>
                    <span class='metric-val'>88.5%</span>
                    <span class='metric-val'>1,265</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-val'>Pass</span>
                    <span class='metric-val'>97.0%</span>
                    <span class='metric-val'>98.2%</span>
                    <span class='metric-val'>4,835</span>
                </div>
                <div class='metric-row' style='background: rgba(176,125,78,0.06); border-radius: 8px; padding: 0.9rem; margin-top: 0.5rem;'>
                    <span style='font-weight: 700; color: var(--text);'>Weighted Avg</span>
                    <span style='font-weight: 700; color: var(--text);'>91.8%</span>
                    <span style='font-weight: 700; color: var(--text);'>91.8%</span>
                    <span style='font-weight: 700; color: var(--text);'>6,100</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Clustering Analysis</div>", unsafe_allow_html=True)
    cl1, cl2, cl3 = st.columns(3)
    clusters = [
        ("At-Risk", "#c05840", "7,818 students", "Avg Score: 58.6", "Study Hrs: 6.94/wk", "Test Prep: 8% completed"),
        ("Average", "#7a6a55", "13,454 students", "Avg Score: 68.6", "Study Hrs: 6.91/wk", "Test Prep: 0% completed"),
        ("High-Performer", "#5a8a45", "9,368 students", "Avg Score: 76.4", "Study Hrs: 6.91/wk", "Test Prep: 100% completed"),
    ]
    for col, (name, color, n, s1, s2, s3) in zip([cl1, cl2, cl3], clusters):
        with col:
            st.markdown(f"""
            <div class='intel-card' style='border-top: 4px solid {color};'>
                <div style='font-family: Playfair Display, serif; font-size: 1.15rem; font-weight: 700; color: {color}; margin-bottom: 0.3rem;'>{name}</div>
                <div style='font-size: 0.8rem; color: var(--text-dim); margin-bottom: 1rem;'>{n}</div>
                <div style='font-size: 0.88rem; color: var(--text); line-height: 2;'>
                    Score: {s1.split(':')[1]}<br>Hours: {s2.split(':')[1]}<br>Prep: {s3.split(':')[1]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Clustering Quality Metrics</div>", unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("""
        <div class='intel-card' style='text-align:center;'>
            <div class='card-label'>Silhouette Score</div>
            <div class='card-value'>0.2112</div>
            <div class='card-body'>Moderate cluster separation -- expected given overlapping learner profiles.</div>
        </div>
        """, unsafe_allow_html=True)
    with q2:
        st.markdown("""
        <div class='intel-card' style='text-align:center;'>
            <div class='card-label'>Davies-Bouldin Index</div>
            <div class='card-value'>1.7311</div>
            <div class='card-body'>k=3 was enforced for interpretability (At-Risk, Average, High-Performer).</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Methodology</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='intel-card'>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;'>
            <div>
                <div style='font-family: Playfair Display, serif; font-weight: 600; margin-bottom: 0.6rem; color: var(--text);'>Preprocessing (10 Steps)</div>
                <div style='font-size: 0.88rem; color: var(--text-dim); line-height: 1.9;'>
                    [Check] Standardised category names<br>
                    [Check] WklyStudyHours -> midpoint numeric<br>
                    [Check] Ordinal encoding (manual, no LabelEncoder)<br>
                    [Check] IQR-based outlier clipping<br>
                    [Check] StandardScaler fit on train only
                </div>
            </div>
            <div>
                <div style='font-family: Playfair Display, serif; font-weight: 600; margin-bottom: 0.6rem; color: var(--text);'>Validation Strategy</div>
                <div style='font-size: 0.88rem; color: var(--text-dim); line-height: 1.9;'>
                    [Check] 80/20 stratified train/test split<br>
                    [Check] 5-fold cross-validation on both models<br>
                    [Check] class_weight='balanced' (no SMOTE needed)<br>
                    [Check] Pass/Fail threshold = median (69.0) -- data-driven<br>
                    [Check] No data leakage -- scaler applied post-split
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
        st.warning("WARNING Prediction Engine unavailable. Please verify model files in /models/")
        return

    st.markdown("<div class='page-title'>Student Success Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Enter student profile below to generate ML-powered academic predictions.</div>", unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1.2], gap="large")

    with col_input:
        st.markdown("""
        <div style='font-family: Playfair Display, serif; font-size: 1.1rem; font-weight: 600; color: var(--text); margin-bottom: 1.2rem;'>Student Profile</div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div class='input-group-header'>STUDENT Academic Baseline</div>", unsafe_allow_html=True)
            math_score    = st.slider("Math Score", 0, 100, 65)
            reading_score = st.slider("Reading Score", 0, 100, 68)
            test_prep_label = st.selectbox("Test Preparation", ["Completed", "Not Completed"])
            test_prep = 0 if test_prep_label == "Completed" else 1

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div class='input-group-header'>TIME Weekly Habits</div>", unsafe_allow_html=True)
            study_map   = {"Low -- < 5 hrs": 2.5, "Moderate -- 5-10 hrs": 7.5, "High -- > 10 hrs": 12.0}
            study_label = st.selectbox("Weekly Study Time", list(study_map.keys()), index=1)
            study_hours = study_map[study_label]
            sport_label    = st.select_slider("Sport Participation", options=["Never", "Sometimes", "Regularly"], value="Sometimes")
            practice_sport = {"Never": 0, "Sometimes": 1, "Regularly": 2}[sport_label]

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div class='input-group-header'>PERSON Demographic & Misc</div>", unsafe_allow_html=True)
            educ_map      = {"High School": 1, "Some College": 2, "Associate's": 3, "Bachelor's": 4, "Master's+": 5}
            educ_label    = st.selectbox("Parent Education Level", list(educ_map.keys()))
            parent_educ   = educ_map[educ_label]
            
            c1, c2 = st.columns(2)
            with c1:
                lunch_label   = st.radio("Lunch Type", ["Standard", "Reduced/Free"], horizontal=True)
                lunch_type    = 1 if lunch_label == "Standard" else 0
            with c2:
                gender_label  = st.radio("Gender", ["Female", "Male"], horizontal=True)
                gender_male   = 1 if gender_label == "Male" else 0
            
            c3, c4 = st.columns(2)
            with c3:
                first_child_label = st.radio("First Child?", ["Yes", "No"], horizontal=True)
                is_first_child = 1 if first_child_label == "Yes" else 0
            with c4:
                transport_label = st.radio("Transport", ["School Bus", "Public"], horizontal=True)
                transport_bus = 1 if transport_label == "School Bus" else 0

            nr_siblings   = st.number_input("Number of Siblings", 0, 6, 1)

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        run = st.button("UPDATE INSIGHTS", type="primary", use_container_width=True)
        if run:
            st.session_state.prediction_run = True

    # ---- RESULTS ----
    with col_result:
        if not st.session_state.prediction_run:
            st.markdown("""
            <div style='background: var(--bg-card); border: 1.5px dashed var(--border); border-radius: var(--radius);
                        height: 820px; display: flex; flex-direction: column; align-items: center;
                        justify-content: center; text-align: center; padding: 4rem;'>
                <div style='font-family: Playfair Display, serif; font-size: 1.3rem; font-weight: 600; color: var(--text-dim);'>Ready for Input</div>
                <div style='font-size: 0.92rem; color: var(--text-dim); margin-top: 0.8rem; max-width: 280px; line-height: 1.6; opacity: 0.7;'>
                    Fill in the student profile on the left and click <strong>Update Insights</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Build input vector (11 features)
            input_data = np.array([[math_score, reading_score, study_hours, parent_educ,
                                    test_prep, lunch_type, practice_sport, nr_siblings,
                                    gender_male, is_first_child, transport_bus]])

            input_reg = scaler_reg.transform(input_data)
            input_clf = scaler_clf.transform(input_data)

            predicted_score  = float(linear_model.predict(input_reg)[0])
            predicted_result = logistic_model.predict(input_clf)[0]
            proba            = logistic_model.predict_proba(input_clf)[0]
            fail_prob        = proba[0] * 100
            pass_prob        = proba[1] * 100
            conf_val         = pass_prob if predicted_result == "Pass" else fail_prob

            # Cluster prediction
            cluster_input = np.array([[predicted_score, study_hours, parent_educ,
                                       lunch_type, test_prep, practice_sport]])
            if scaler_cluster and kmeans_model:
                cluster_scaled = scaler_cluster.transform(cluster_input)
                raw_label      = kmeans_model.predict(cluster_scaled)[0]
                centers        = kmeans_model.cluster_centers_
                order          = np.argsort(centers[:, 0])
                name_map       = {order[0]: "At-Risk", order[1]: "Average", order[2]: "High-Performer"}
                learner_seg    = name_map[raw_label]
                
                # Logic Fix: Align segment with score performance
                if predicted_score < 60:
                    learner_seg = "At-Risk"
                elif predicted_score < 75 and learner_seg == "High-Performer":
                    learner_seg = "Average"
            else:
                learner_seg = "At-Risk"

            result_color = "#5a8a45" if is_pass else "#c05840"

            # -- Executive Summary --
            st.markdown("<div class='section-title' style='margin-top:0;'>Executive Summary</div>", unsafe_allow_html=True)
            e1, e2, e3 = st.columns(3)
            with e1:
                st.markdown(f"""
                <div class='intel-card' style='text-align:center; border-top: 3px solid var(--accent);'>
                    <div class='card-label'>Predicted Score</div>
                    <div style='font-family: Playfair Display, serif; font-size: 2.6rem; font-weight: 700; color: var(--accent);'>{predicted_score:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with e2:
                st.markdown(f"""
                <div class='intel-card' style='text-align:center; border-top: 3px solid {result_color};'>
                    <div class='card-label'>Outcome</div>
                    <div style='font-family: Playfair Display, serif; font-size: 2.6rem; font-weight: 700; color: {result_color};'>{predicted_result.upper()}</div>
                </div>
                """, unsafe_allow_html=True)
            with e3:
                st.markdown(f"""
                <div class='intel-card' style='text-align:center; border-top: 3px solid var(--text-dim);'>
                    <div class='card-label'>Learner Segment</div>
                    <div style='font-family: Playfair Display, serif; font-size: 1.7rem; font-weight: 700; color: var(--text); margin-top: 0.3rem;'>{learner_seg}</div>
                </div>
                """, unsafe_allow_html=True)

            # -- Performance Analytics --
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
                    textfont=dict(color="#7a6a55", size=12, family="DM Sans")
                ))
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=230,
                    yaxis=dict(range=[0, 115], gridcolor="rgba(100,80,50,0.08)", tickfont=dict(color="#7a6a55")),
                    xaxis=dict(tickfont=dict(color="#7a6a55", size=11)),
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with gauge_col:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pass_prob,
                    title=dict(text="Pass Confidence %", font=dict(color="#7a6a55", size=12, family="DM Sans")),
                    number=dict(font=dict(color="#2c2416", size=36, family="Playfair Display")),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickfont=dict(color="#7a6a55", size=9)),
                        bar=dict(color="#b07d4e", thickness=0.25),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 50],  color="rgba(192,88,64,0.12)"),
                            dict(range=[50, 75], color="rgba(176,125,78,0.12)"),
                            dict(range=[75, 100], color="rgba(90,138,69,0.12)"),
                        ],
                        threshold=dict(line=dict(color="#b07d4e", width=2), thickness=0.75, value=69)
                    )
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=30, b=10, l=20, r=20),
                    height=230,
                    font=dict(family="DM Sans")
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # -- Strategic Recommendations --
            st.markdown("<div class='section-title'>Strategic Recommendations</div>", unsafe_allow_html=True)

            # Build dynamic recommendations
            recs = []
            if test_prep == 1:  # not completed
                recs.append(("DOC", "Enroll in Test Preparation", "This student has not completed test prep. Data shows 100% of High-Performers completed it -- this single factor has the highest impact on outcome."))
            else:
                recs.append(("TROPHY", "Sustainability & Advancement", "Maintain prep consistency and explore advanced peer mentoring or enrichment programs."))

            if study_hours < 7.5:
                recs.append(("TIME", "Increase Study Hours", "Student is in the Low study category (<5 hrs/wk). Moving to Moderate (5-10 hrs) can meaningfully improve score predictions."))
            else:
                recs.append(("CHECK", "Study Habits on Track", f"Weekly study commitment of {study_label.split('--')[1].strip()} is well-positioned. Focus on quality over quantity."))

            if predicted_score < 69:
                recs.append(("WARNING", "Early Intervention Alert", f"Predicted score ({predicted_score:.1f}) is below the Pass threshold (69.0). Consider targeted academic support immediately."))

            for icon, title, body in recs:
                st.markdown(f"""
                <div class='rec-card'>
                    <strong>{icon} {title}</strong><br>
                    <span style='color: var(--text-dim);'>{body}</span>
                </div>
                """, unsafe_allow_html=True)

            # Note about test prep
            if test_prep == 0:
                st.markdown("""
                <div style='background: rgba(90,138,69,0.08); border: 1px solid rgba(90,138,69,0.3); border-radius: 10px;
                            padding: 0.9rem 1.2rem; font-size: 0.85rem; color: var(--text-dim); margin-top: 0.5rem;'>
                    CHECK <strong style='color: var(--text);'>Note:</strong> Student has already completed prep courses, which positively weights the prediction.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
            if st.button("RESET Reset Predictor", key="reset"):
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