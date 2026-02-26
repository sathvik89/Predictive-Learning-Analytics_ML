
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
# FIXED ADAPTIVE CSS (Supports Dark & Light Mode)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* Global Font Fix */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Modern Card UI - Works in Dark/Light Mode */
    .metric-container {
        background: rgba(255, 255, 255, 0.05); /* Semi-transparent */
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
        margin-bottom: 10px;
    }

    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #8899ac; /* Neutral blue-gray */
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        /* Value colors handled via inline style for logic */
    }

    /* Sidebar Cleanup */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.1);
    }

    /* Section Headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 30px 0 15px 0;
        padding-left: 12px;
        border-left: 4px solid #3b82f6;
    }

    /* Insight Boxes */
    .insight-box {
        background: rgba(59, 130, 246, 0.1);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS (Logic Preserved)
# ============================================================
@st.cache_resource
def load_models():
    # Note: Replace these with your actual paths
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

# ============================================================
# PREDICTIONS
# ============================================================
input_data = np.array([[math_score, reading_score, study_hours, parent_educ, test_prep, lunch_type, practice_sport, nr_siblings, gender_male, is_first_child, transport]])

# Model Processing
input_scaled_reg = scaler_reg.transform(input_data)
input_scaled_clf = scaler_clf.transform(input_data)

predicted_exam_score = float(np.clip(linear_model.predict(input_scaled_reg)[0], 0, 100))
predicted_result = logistic_model.predict(input_scaled_clf)[0]
result_proba = logistic_model.predict_proba(input_scaled_clf)[0]

# Clustering
cluster_input = np.array([[predicted_exam_score, study_hours, parent_educ, lunch_type, test_prep, practice_sport]])
cluster_scaled = scaler_cluster.transform(cluster_input)
cluster_id = int(kmeans_model.predict(cluster_scaled)[0])

# Category Mapping
cluster_centers_exam = kmeans_model.cluster_centers_[:, 0]
sorted_cluster_ids = np.argsort(cluster_centers_exam)
label_list = ["At-Risk", "Average", "High-Performer"]
cluster_label_map = {int(sorted_cluster_ids[i]): label_list[i] for i in range(3)}
learner_category = cluster_label_map[cluster_id]

# ============================================================
# DASHBOARD MAIN
# ============================================================
st.markdown("<h1 style='color: #3b82f6;'>Student Intelligence Portal</h1>", unsafe_allow_html=True)
st.markdown("Predictive success tracking and academic intervention analytics.")

st.markdown("<div class='section-title'>Executive Summary</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""<div class='metric-container'>
        <div class='metric-label'>Predicted Score</div>
        <div class='metric-value' style='color:#3b82f6'>{predicted_exam_score:.1f}%</div>
    </div>""", unsafe_allow_html=True)

with col2:
    res_color = "#10b981" if predicted_result == "Pass" else "#ef4444"
    st.markdown(f"""<div class='metric-container'>
        <div class='metric-label'>Outcome Prediction</div>
        <div class='metric-value' style='color:{res_color}'>{predicted_result.upper()}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    cat_color = {"At-Risk": "#ef4444", "Average": "#f59e0b", "High-Performer": "#10b981"}[learner_category]
    st.markdown(f"""<div class='metric-container'>
        <div class='metric-label'>Learner Segment</div>
        <div class='metric-value' style='color:{cat_color}'>{learner_category}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Performance Analytics</div>", unsafe_allow_html=True)

v_col1, v_col2 = st.columns([1.5, 1])

with v_col1:
    # Bar Chart with clean theme
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Math', 'Reading', 'Predicted Exam'],
        y=[math_score, reading_score, predicted_exam_score],
        marker_color=['#64748b', '#64748b', '#3b82f6'],
        width=0.4
    ))
    fig.update_layout(
        template="plotly_dark", # Forces dark theme compatibility
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=20, r=20),
        height=350,
        yaxis=dict(range=[0, 105])
    )
    st.plotly_chart(fig, use_container_width=True)

with v_col2:
    # Pass Probability Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result_proba[1] * 100,
        title={'text': "Pass Confidence %", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "rgba(255,255,255,0.05)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [50, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
            ]
        }
    ))
    fig_gauge.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(t=50, b=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

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

st.caption("Milestone 1 Analytics Dashboard | Inter-Adaptive UI Engine")

