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
    page_title="Student Intelligence Portal",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# WARM CREAM UI — Matching reference design
# ============================================================

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
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS (Logic Preserved)
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
# SIDEBAR — USER INPUTS
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