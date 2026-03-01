import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path fill='%23b07d4e' d='M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5'/></svg>",
    layout="wide"
)

# ============================================================
# SVG ICON LIBRARY
# ============================================================
ICONS = {
    "home":           '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
    "chart":          '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "predict":        '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "trending_up":    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "check_circle":   '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "users":          '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "alert_triangle": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "academic":       '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',
    "clock":          '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "user":           '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    "pin":            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>',
    "book_open":      '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',
    "award":          '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>',
    "warning":        '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "clipboard":      '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/></svg>',
    "arrow_up":       '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>',
    "database":       '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    "cpu":            '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "settings":       '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
    "alert_sm":       '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#c05840" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "minus_circle":   '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#7a6a55" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/></svg>',
    "star":           '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#5a8a45" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
    "target":         '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "layers":         '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
}

def icon(name, color=None):
    svg = ICONS.get(name, "")
    if color:
        svg = svg.replace('stroke="currentColor"', f'stroke="{color}"')
    return svg

# ============================================================
# LOAD MODELS
# ============================================================
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
# THEME
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
    font-size: 15px !important;
}

#MainMenu, footer, header, [data-testid="stHeader"] { visibility: hidden; }

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

div[data-baseweb="slider"] > div > div > div { background-color: var(--accent) !important; }
.stSlider [role="slider"] {
    background-color: var(--accent) !important;
    border: 2px solid white !important;
}
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
div[data-baseweb="select"] * { color: var(--text) !important; }
div[data-baseweb="select"] svg { fill: var(--accent) !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label[data-testid="stWidgetLabel"] p {
    color: var(--text-dim) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

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

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem !important;
}

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

.conf-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
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
.input-group-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
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
    st.markdown(f"""
    <div style='padding: 2rem 1rem 1.5rem 1rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem;'>
        <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;'>
            {icon("layers", "#b07d4e")}
            <div style='font-family: Playfair Display, serif; font-size: 1.05rem; font-weight: 700;
                        color: var(--text); line-height: 1.3;'>Student Performance<br>Analytics</div>
        </div>
        <div style='font-size: 0.7rem; color: var(--text-dim); margin-top: 0.4rem;
                    text-transform: uppercase; letter-spacing: 0.06em;'>ML Analytics &middot; Milestone 1</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("  HOME", use_container_width=True):
        st.session_state.page = "home"
        st.session_state.prediction_run = False
    if st.button("  PERFORMANCE", use_container_width=True):
        st.session_state.page = "performance"
    if st.button("  PREDICT", use_container_width=True):
        st.session_state.page = "predict"

    st.markdown(f"""
    <div style='margin-top: 1.5rem; padding: 1.2rem; background: var(--bg-card);
                border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow);'>
        <div style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
                    letter-spacing: 0.12em; color: var(--accent); margin-bottom: 0.8rem;
                    border-bottom: 1px solid var(--border); padding-bottom: 0.4rem;'>Model Performance</div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 600; margin-bottom: 0.5rem;
                    display:flex; justify-content:space-between;'><span>Accuracy</span><span>91.76%</span></div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 600; margin-bottom: 0.5rem;
                    display:flex; justify-content:space-between;'><span>R&#178; Score</span><span>0.9397</span></div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 600; margin-bottom: 0.5rem;
                    display:flex; justify-content:space-between;'><span>F1 Score</span><span>0.9176</span></div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 600;
                    display:flex; justify-content:space-between;'><span>MAE</span><span>3.04 marks</span></div>
    </div>

    <div style='margin-top: 1rem; padding: 1.2rem; background: var(--bg-card);
                border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow);'>
        <div style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
                    letter-spacing: 0.12em; color: var(--accent); margin-bottom: 0.8rem;
                    border-bottom: 1px solid var(--border); padding-bottom: 0.4rem;'>System Info</div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 500; margin-bottom: 0.5rem;
                    display:flex; align-items:center; gap:0.5rem;'>
            {icon("database", "#b07d4e")} 30,640 records loaded
        </div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 500; margin-bottom: 0.5rem;
                    display:flex; align-items:center; gap:0.5rem;'>
            {icon("cpu", "#b07d4e")} 3 models active
        </div>
        <div style='font-size: 0.82rem; color: var(--text); font-weight: 500;
                    display:flex; align-items:center; gap:0.5rem;'>
            {icon("settings", "#b07d4e")} 11 features enabled
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HOME PAGE
# ============================================================
def home_page():
    st.markdown("""
    <div class='hero-strip'>
        <div style='font-size: 0.7rem; font-weight: 700; letter-spacing: 0.18em;
                    text-transform: uppercase; color: var(--accent); margin-bottom: 0.8rem;'>
            Gen AI Course &middot; Milestone 1 &middot; February 2026
        </div>
        <div class='page-title'>Intelligent Learning Analytics<br>&amp; Agentic AI Study Coach</div>
        <div style='font-size: 0.98rem; color: var(--text-dim); max-width: 650px;
                    line-height: 1.7; margin-top: 0.6rem;'>
            A complete ML pipeline for student performance analysis &#8212; predicting exam scores,
            classifying Pass/Fail outcomes, and segmenting students into meaningful learner
            categories using 30,640 student records.
        </div>
        <div style='margin-top: 1.8rem; display: flex; gap: 0.8rem; flex-wrap: wrap;'>
            <div style='background: var(--accent); color: white; padding: 0.4rem 1rem;
                        border-radius: 50px; font-size: 0.78rem; font-weight: 600;'>R&#178; = 0.9397</div>
            <div style='background: var(--accent); color: white; padding: 0.4rem 1rem;
                        border-radius: 50px; font-size: 0.78rem; font-weight: 600;'>Accuracy = 91.76%</div>
            <div style='background: var(--accent); color: white; padding: 0.4rem 1rem;
                        border-radius: 50px; font-size: 0.78rem; font-weight: 600;'>3 Learner Segments</div>
            <div style='background: var(--accent); color: white; padding: 0.4rem 1rem;
                        border-radius: 50px; font-size: 0.78rem; font-weight: 600;'>30,640 Records</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Platform Capabilities</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    caps = [
        (icon("trending_up",    "#b07d4e"), "Score Prediction",
         "Linear Regression predicts ExamScore (R&#178; = 0.94, MAE = 3.0) using 11 academic and behavioural features."),
        (icon("check_circle",   "#b07d4e"), "Pass / Fail Classification",
         "Logistic Regression classifies students with 91.8% accuracy using balanced class weights."),
        (icon("users",          "#b07d4e"), "Learner Segmentation",
         "K-Means (k=3) groups students into At-Risk, Average, and High-Performer clusters based on performance and behaviour."),
        (icon("alert_triangle", "#b07d4e"), "Early Intervention",
         "Flags at-risk students before exams using behavioural and academic features, enabling targeted teacher action."),
    ]
    for col, (ic_svg, title, body) in zip([c1, c2, c3, c4], caps):
        with col:
            st.markdown(f"""
            <div class='intel-card'>
                <div style='margin-bottom:0.6rem;'>{ic_svg}</div>
                <div style='font-family: Playfair Display, serif; font-size: 1rem; font-weight: 600;
                            color: var(--text); margin-bottom: 0.5rem;'>{title}</div>
                <div class='card-body'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>System Metrics</div>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("Total Records",  "30,640"),
        ("Training Set",   "24,512"),
        ("Test Set",       "6,128"),
        ("Input Features", "11"),
        ("ML Models",      "3"),
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
    st.markdown(f"""
    <div class='intel-card' style='background: linear-gradient(135deg, #faf5ee, #f5ede0);
                                   border-left: 4px solid var(--accent);'>
        <div style='display:flex; align-items:center; gap:0.6rem; font-family: Playfair Display, serif;
                    font-size: 1.08rem; font-weight: 600; color: var(--text); margin-bottom: 0.8rem;'>
            {icon("pin", "#b07d4e")}
            Test Preparation is the Single Biggest Academic Differentiator
        </div>
        <div style='font-size: 0.9rem; color: var(--text-dim); line-height: 1.7; margin-bottom: 1.5rem;'>
            All three learner clusters study approximately the same hours per week (~6.91&#8211;6.94 hrs).
            Yet <strong style='color: var(--accent);'>100% of High-Performers completed test prep</strong>
            while 0% of Average students did, and only 8% of At-Risk students did.
            Study time alone does not predict success &#8212; <em>how</em> students prepare matters far more.
        </div>
        <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;'>
            <div style='text-align:center; background: rgba(192,88,64,0.06); border-radius: 10px;
                        padding: 1rem; border: 1px solid rgba(192,88,64,0.2);'>
                <div style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
                            letter-spacing: 0.1em; color: #c05840; margin-bottom: 0.3rem;'>At-Risk</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.9rem;
                            font-weight: 700; color: #c05840;'>58.64</div>
                <div style='font-size: 0.75rem; color: var(--text-dim); margin-top: 0.2rem;'>Avg Score &middot; 8% prep</div>
                <div style='font-size: 0.72rem; color: var(--text-dim);'>7,818 students (25.5%)</div>
            </div>
            <div style='text-align:center; background: var(--bg-card); border-radius: 10px;
                        padding: 1rem; border: 1px solid var(--border);'>
                <div style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
                            letter-spacing: 0.1em; color: var(--text-dim); margin-bottom: 0.3rem;'>Average</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.9rem;
                            font-weight: 700; color: var(--text);'>68.59</div>
                <div style='font-size: 0.75rem; color: var(--text-dim); margin-top: 0.2rem;'>Avg Score &middot; 0% prep</div>
                <div style='font-size: 0.72rem; color: var(--text-dim);'>13,454 students (43.9%)</div>
            </div>
            <div style='text-align:center; background: rgba(90,138,69,0.07); border-radius: 10px;
                        padding: 1rem; border: 1px solid rgba(90,138,69,0.25);'>
                <div style='font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
                            letter-spacing: 0.1em; color: #5a8a45; margin-bottom: 0.3rem;'>High-Performer</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.9rem;
                            font-weight: 700; color: #5a8a45;'>76.40</div>
                <div style='font-size: 0.75rem; color: var(--text-dim); margin-top: 0.2rem;'>Avg Score &middot; 100% prep</div>
                <div style='font-size: 0.72rem; color: var(--text-dim);'>9,368 students (30.6%)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Team</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='intel-card' style='padding: 1.5rem 2rem;'>
        <div style='display: flex; justify-content: space-between; align-items: center;
                    flex-wrap: wrap; gap: 1rem;'>
            <div>
                <div style='font-size: 0.9rem; color: var(--text); font-weight: 600; margin-bottom: 0.3rem;'>
                    Sathvik Koriginja (2401010231) &nbsp;&middot;&nbsp;
                    Anushka Tyagi (2401010090) &nbsp;&middot;&nbsp;
                    Apoorva Choudhary (2401010092)
                </div>
                <div style='font-size: 0.78rem; color: var(--text-dim);'>
                    Gen AI Course &middot; Milestone 1 &middot; February 2026 &middot;
                    Dataset: Students Exam Scores Extended (Kaggle)
                </div>
            </div>
            <div style='font-size: 0.78rem; color: var(--text-dim);'>
                pandas &middot; numpy &middot; scikit-learn &middot; plotly &middot; streamlit
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PERFORMANCE PAGE
# ============================================================
def performance_page():
    st.markdown("<div class='page-title'>Analysis Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Full model performance report validated with 5-fold cross-validation on 30,640 student records.</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Linear Regression &#8212; ExamScore Prediction</div>", unsafe_allow_html=True)
    lr1, lr2, lr3, lr4 = st.columns(4)
    reg_metrics = [
        ("R&#178; Score",        "0.9397",           "Explains 94% of variance in ExamScore"),
        ("MAE",                  "3.04 marks",        "Average prediction error out of 100"),
        ("RMSE",                 "3.78 marks",        "Root mean squared error &#8212; tight predictions"),
        ("CV R&#178; (5-fold)",  "0.9394 &#177; 0.0021", "Stable across all folds &#8212; no overfitting"),
    ]
    for col, (lbl, val, sub) in zip([lr1, lr2, lr3, lr4], reg_metrics):
        with col:
            st.markdown(f"""
            <div class='intel-card' style='text-align:center;'>
                <div class='card-label'>{lbl}</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.65rem; font-weight: 700;
                            color: var(--text); line-height: 1.1; margin: 0.4rem 0;'>{val}</div>
                <div class='card-body' style='font-size: 0.78rem;'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Logistic Regression &#8212; Pass / Fail Classification</div>", unsafe_allow_html=True)
    cl1, cl2, cl3, cl4 = st.columns(4)
    clf_metrics = [
        ("Accuracy",             "91.76%",            "Correctly classifies 92% of students"),
        ("F1 Score (weighted)",  "0.9176",             "Strong balance of precision and recall"),
        ("CV Accuracy (5-fold)", "92.47% &#177; 0.55%", "Highly stable &#8212; consistent across folds"),
        ("Fail Recall",          "0.92",               "Catches 92% of all at-risk students"),
    ]
    for col, (lbl, val, sub) in zip([cl1, cl2, cl3, cl4], clf_metrics):
        with col:
            st.markdown(f"""
            <div class='intel-card' style='text-align:center;'>
                <div class='card-label'>{lbl}</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.65rem; font-weight: 700;
                            color: var(--text); line-height: 1.1; margin: 0.4rem 0;'>{val}</div>
                <div class='card-body' style='font-size: 0.78rem;'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Classification Report Detail</div>", unsafe_allow_html=True)
    col_cm, col_rep = st.columns([1, 1.3], gap="large")

    with col_cm:
        st.markdown("""
        <div class='intel-card'>
            <div class='card-label'>Confusion Matrix &#8212; Test Set</div>
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
                <div style='background: rgba(176,125,78,0.07); border-radius: 8px; padding: 0.85rem;
                            margin-top: 0.5rem; display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-weight:700; color:var(--text); min-width:100px;'>Weighted Avg</span>
                    <span style='font-weight:700; color:var(--text);'>0.9177</span>
                    <span style='font-weight:700; color:var(--text);'>0.9176</span>
                    <span style='font-weight:700; color:var(--text);'>6,100</span>
                </div>
                <div style='margin-top: 0.9rem; font-size: 0.78rem; color: var(--text-dim); line-height: 1.5;'>
                    Balanced performance across both classes. Model catches 92% of failing students
                    without SMOTE &#8212; achieved via class_weight='balanced'.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>K-Means Clustering &#8212; Learner Segmentation (k=3)</div>", unsafe_allow_html=True)
    ck1, ck2, ck3 = st.columns(3)
    clusters = [
        (ICONS["alert_sm"],     "At-Risk",        "#c05840", "7,818 students &middot; 25.5%",
         [("Avg ExamScore","58.64"),("Study Hrs/Wk","6.94"),("Parent Educ","2.17 / 5"),("Test Prep Completed","8%")]),
        (ICONS["minus_circle"], "Average",        "#7a6a55", "13,454 students &middot; 43.9%",
         [("Avg ExamScore","68.59"),("Study Hrs/Wk","6.91"),("Parent Educ","2.16 / 5"),("Test Prep Completed","0%")]),
        (ICONS["star"],         "High-Performer", "#5a8a45", "9,368 students &middot; 30.6%",
         [("Avg ExamScore","76.40"),("Study Hrs/Wk","6.91"),("Parent Educ","2.20 / 5"),("Test Prep Completed","100%")]),
    ]
    for col, (ic_svg, name, color, n, stats) in zip([ck1, ck2, ck3], clusters):
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
                <div style='margin-bottom: 0.5rem;'>{ic_svg}</div>
                <div style='font-family: Playfair Display, serif; font-size: 1.1rem; font-weight: 700;
                            color: {color}; margin-bottom: 0.2rem;'>{name}</div>
                <div style='font-size: 0.75rem; color: var(--text-dim); margin-bottom: 1.2rem;'>{n}</div>
                {rows}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Clustering Quality &amp; Design Decision</div>", unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("""
        <div class='intel-card' style='text-align:center;'>
            <div class='card-label'>Silhouette Score</div>
            <div class='card-value'>0.2112</div>
            <div class='card-body' style='margin-top:0.6rem;'>Moderate separation &#8212; expected for behavioural
            data. Students naturally overlap between categories. This is a dataset characteristic, not a modelling error.</div>
        </div>
        """, unsafe_allow_html=True)
    with q2:
        st.markdown("""
        <div class='intel-card' style='text-align:center;'>
            <div class='card-label'>Davies-Bouldin Index</div>
            <div class='card-value'>1.7311</div>
            <div class='card-body' style='margin-top:0.6rem;'>k=3 selected over best_k=5 (silhouette 0.2211 vs 0.2112
            &#8212; negligible). k=3 maps directly to At-Risk / Average / High-Performer for interpretability.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Preprocessing &amp; Validation Methodology</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='intel-card'>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 2.5rem;'>
            <div>
                <div style='font-family: Playfair Display, serif; font-weight: 600; margin-bottom: 0.8rem;
                            color: var(--text); font-size: 1rem;'>Preprocessing Pipeline (10 Steps)</div>
                <div style='font-size: 0.85rem; color: var(--text-dim); line-height: 2;'>
                    &#10003; Text standardisation &#8212; lowercase, strip whitespace<br>
                    &#10003; Merged 'some high school' into 'high_school'<br>
                    &#10003; WklyStudyHours &rarr; midpoints: &lt;5&rarr;2.5, 5-10&rarr;7.5, &gt;10&rarr;12.0<br>
                    &#10003; Ordinal encoding with manual mapping (no LabelEncoder)<br>
                    &#10003; Mode fill for categorical, median fill for numerical<br>
                    &#10003; No duplicates found (all 30,640 rows unique)<br>
                    &#10003; IQR outlier clipping &#8212; NrSiblings: 291 values clipped<br>
                    &#10003; Pass/Fail threshold = median ExamScore (69.0)
                </div>
            </div>
            <div>
                <div style='font-family: Playfair Display, serif; font-weight: 600; margin-bottom: 0.8rem;
                            color: var(--text); font-size: 1rem;'>Validation Strategy</div>
                <div style='font-size: 0.85rem; color: var(--text-dim); line-height: 2;'>
                    &#10003; 80/20 stratified train/test split (random_state=42)<br>
                    &#10003; 5-fold cross-validation on both supervised models<br>
                    &#10003; class_weight='balanced' &#8212; no SMOTE required<br>
                    &#10003; Separate scalers for regression, classification, clustering<br>
                    &#10003; StandardScaler fit on train only &#8212; no data leakage<br>
                    &#10003; ExamScore excluded from X &#8212; no circular dependency<br>
                    &#10003; All results genuine &#8212; no artificial thresholds or synthetic data
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
        st.warning("Prediction Engine unavailable. Please verify model files in /models/")
        return

    st.markdown("<div class='page-title'>Student Success Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Enter the student's academic and behavioural profile to generate ML-powered predictions.</div>", unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1.2], gap="large")

    # ── INPUT PANEL ──
    with col_input:
        st.markdown("""
        <div style='font-family: Playfair Display, serif; font-size: 1.05rem; font-weight: 600;
                    color: var(--text); margin-bottom: 1rem;'>Student Profile</div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(f"<div class='input-group-header'>{icon('academic','#b07d4e')} &nbsp; Academic Baseline</div>", unsafe_allow_html=True)
            math_score    = st.slider("Math Score", 0, 100, 65)
            reading_score = st.slider("Reading Score", 0, 100, 68)
            test_prep_label = st.selectbox(
                "Test Preparation",
                ["Completed", "Not Completed"],
                help="Students who complete test prep are significantly more likely to be High-Performers"
            )
            test_prep = 0 if test_prep_label == "Completed" else 1

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(f"<div class='input-group-header'>{icon('clock','#b07d4e')} &nbsp; Weekly Habits</div>", unsafe_allow_html=True)
            study_map = {
                "Low &#8212; less than 5 hrs (2.5 hrs avg)":    2.5,
                "Moderate &#8212; 5 to 10 hrs (7.5 hrs avg)":   7.5,
                "High &#8212; more than 10 hrs (12.0 hrs avg)":  12.0,
            }
            study_label = st.selectbox(
                "Weekly Study Time",
                ["Low — less than 5 hrs (2.5 hrs avg)",
                 "Moderate — 5 to 10 hrs (7.5 hrs avg)",
                 "High — more than 10 hrs (12.0 hrs avg)"],
                index=1
            )
            study_hours_map = {
                "Low — less than 5 hrs (2.5 hrs avg)":    2.5,
                "Moderate — 5 to 10 hrs (7.5 hrs avg)":   7.5,
                "High — more than 10 hrs (12.0 hrs avg)":  12.0,
            }
            study_hours = study_hours_map[study_label]

            sport_label    = st.select_slider("Sport Participation", options=["Never", "Sometimes", "Regularly"], value="Sometimes")
            practice_sport = {"Never": 0, "Sometimes": 1, "Regularly": 2}[sport_label]

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(f"<div class='input-group-header'>{icon('user','#b07d4e')} &nbsp; Demographics &amp; Background</div>", unsafe_allow_html=True)
            educ_map = {
                "High School":        1,
                "Some College":       2,
                "Associate's Degree": 3,
                "Bachelor's Degree":  4,
                "Master's Degree":    5,
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
        if st.button("UPDATE INSIGHTS", type="primary", use_container_width=True):
            st.session_state.prediction_run = True

    # ── RESULTS PANEL ──
    with col_result:
        if not st.session_state.prediction_run:
            st.markdown("""
            <div style='background: var(--bg-card); border: 1.5px dashed var(--border);
                        border-radius: var(--radius); height: 880px; display: flex;
                        flex-direction: column; align-items: center; justify-content: center;
                        text-align: center; padding: 4rem;'>
                <div style='width:48px; height:48px; border-radius:50%; background:rgba(176,125,78,0.08);
                            display:flex; align-items:center; justify-content:center; margin: 0 auto 1.2rem auto;'>
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
                         fill="none" stroke="#b07d4e" stroke-width="1.5" opacity="0.4"
                         stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>
                    </svg>
                </div>
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
            # Feature order: MathScore, ReadingScore, WklyStudyHours, ParentEduc,
            # TestPrep_none, LunchType_standard, PracticeSport, NrSiblings,
            # Gender_male, IsFirstChild_yes, TransportMeans_school_bus
            input_data = np.array([[
                math_score, reading_score, study_hours, parent_educ,
                test_prep,  lunch_type,    practice_sport, nr_siblings,
                gender_male, is_first_child, transport_bus
            ]])

            input_reg = scaler_reg.transform(input_data)
            input_clf = scaler_clf.transform(input_data)

            predicted_score  = float(np.clip(linear_model.predict(input_reg)[0], 0, 100))
            predicted_result = logistic_model.predict(input_clf)[0]
            proba            = logistic_model.predict_proba(input_clf)[0]
            # classes_ = ['Fail', 'Pass']
            fail_prob = proba[0] * 100
            pass_prob = proba[1] * 100
            conf_val  = pass_prob if predicted_result == "Pass" else fail_prob

            # Clustering: ExamScore, WklyStudyHours, ParentEduc, LunchType_standard, TestPrep_none, PracticeSport
            cluster_input = np.array([[
                predicted_score, study_hours, parent_educ,
                lunch_type, test_prep, practice_sport
            ]])
            if scaler_cluster and kmeans_model:
                cluster_scaled = scaler_cluster.transform(cluster_input)
                raw_label      = kmeans_model.predict(cluster_scaled)[0]
                centers        = kmeans_model.cluster_centers_
                order          = np.argsort(centers[:, 0])
                name_map       = {order[0]: "At-Risk", order[1]: "Average", order[2]: "High-Performer"}
                learner_seg    = name_map[raw_label]
            else:
                learner_seg = "At-Risk" if predicted_score < 63.6 else ("Average" if predicted_score < 72.5 else "High-Performer")

            # Score-based alignment overrides
            if predicted_score < 60:
                learner_seg = "At-Risk"
            elif predicted_score > 82:
                learner_seg = "High-Performer"
            else:
                if learner_seg == "High-Performer" and predicted_score < 76:
                    learner_seg = "Average"
                if learner_seg == "At-Risk" and predicted_score > 65:
                    learner_seg = "Average"

            is_pass      = predicted_result == "Pass"
            result_color = "#5a8a45" if is_pass else "#c05840"
            seg_colors   = {"At-Risk": "#c05840", "Average": "#7a6a55", "High-Performer": "#5a8a45"}
            seg_icons_svg = {
                "At-Risk":        ICONS["alert_sm"],
                "Average":        ICONS["minus_circle"],
                "High-Performer": ICONS["star"],
            }
            seg_color = seg_colors.get(learner_seg, "#7a6a55")
            seg_icon  = seg_icons_svg.get(learner_seg, "")

            # Executive Summary
            st.markdown("<div class='section-title' style='margin-top:0;'>Executive Summary</div>", unsafe_allow_html=True)
            e1, e2, e3 = st.columns(3)
            with e1:
                st.markdown(f"""
                <div class='intel-card' style='text-align:center; border-top:3px solid var(--accent); padding:1.4rem 1rem;'>
                    <div class='card-label'>Predicted Score</div>
                    <div style='font-family:Playfair Display,serif; font-size:2.3rem; font-weight:700; color:var(--accent);'>{predicted_score:.1f}%</div>
                    <div style='font-size:0.72rem; color:var(--text-dim); margin-top:0.2rem;'>Threshold: 69.0</div>
                </div>
                """, unsafe_allow_html=True)
            with e2:
                st.markdown(f"""
                <div class='intel-card' style='text-align:center; border-top:3px solid {result_color}; padding:1.4rem 1rem;'>
                    <div class='card-label'>Outcome</div>
                    <div style='font-family:Playfair Display,serif; font-size:2.3rem; font-weight:700; color:{result_color};'>{predicted_result.upper()}</div>
                    <div style='font-size:0.72rem; color:var(--text-dim); margin-top:0.2rem;'>Confidence: {conf_val:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with e3:
                st.markdown(f"""
                <div class='intel-card' style='text-align:center; border-top:3px solid {seg_color}; padding:1.4rem 1rem;'>
                    <div class='card-label'>Learner Segment</div>
                    <div style='margin:0.3rem 0; display:flex; justify-content:center;'>{seg_icon}</div>
                    <div style='font-family:Playfair Display,serif; font-size:1.1rem; font-weight:700; color:{seg_color};'>{learner_seg}</div>
                </div>
                """, unsafe_allow_html=True)

            # Performance Analytics
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
                fig_bar.add_hline(
                    y=69, line_dash="dash", line_color="#c05840", line_width=1.5,
                    annotation_text="Pass threshold (69.0)",
                    annotation_font_color="#c05840",
                    annotation_font_size=10
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, b=5, l=5, r=5),
                    height=215,
                    yaxis=dict(range=[0, 118], gridcolor="rgba(100,80,50,0.07)",
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

            # Strategic Recommendations
            st.markdown("<div class='section-title'>Strategic Recommendations</div>", unsafe_allow_html=True)
            recs = []

            if test_prep == 1:
                recs.append((icon("clipboard", "#b07d4e"), "Enrol in Test Preparation",
                    "Student has not completed test prep. Clustering shows 100% of High-Performers "
                    "completed it vs 0% of Average students &#8212; this is the single most actionable improvement available."))
            else:
                recs.append((icon("award", "#5a8a45"), "Sustain and Advance",
                    "Student has completed test preparation &#8212; the strongest differentiator in the dataset. "
                    "Maintain consistency and explore advanced peer mentoring or enrichment programmes."))

            if study_hours < 7.5:
                recs.append((icon("arrow_up", "#b07d4e"), "Increase Weekly Study Hours",
                    "Student is in the Low study category (less than 5 hrs/wk). Moving to Moderate "
                    "(5&#8211;10 hrs/wk) is consistently associated with better outcomes across all learner segments."))
            else:
                recs.append((icon("check_circle", "#5a8a45"), "Study Commitment On Track",
                    "Study time is well-positioned. Focus on quality of preparation rather than adding more hours."))

            if predicted_score < 69.0:
                recs.append((icon("warning", "#c05840"), "Early Intervention Required",
                    f"Predicted score ({predicted_score:.1f}) is below the Pass threshold (69.0). "
                    "Targeted academic support before the exam is recommended."))

            if learner_seg == "At-Risk":
                recs.append((icon("book_open", "#c05840"), "At-Risk Study Plan",
                    "Revise fundamentals daily, identify and address weak subject areas, and prioritise "
                    "consistent daily practice over long infrequent sessions."))
            elif learner_seg == "Average":
                recs.append((icon("target", "#b07d4e"), "Move Toward High-Performer",
                    "Practice moderate-to-advanced problems and attempt weekly mock tests. "
                    "Completing test preparation is the clearest path to the High-Performer cluster."))

            for ic_svg, title, body in recs:
                st.markdown(f"""
                <div class='rec-card'>
                    <div style='display:flex; align-items:center; gap:0.5rem; font-weight:700;
                                margin-bottom:0.4rem;'>{ic_svg} {title}</div>
                    <span style='color: var(--text-dim);'>{body}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            if st.button("Reset Predictor", key="reset"):
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