# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import plotly.graph_objects as go
# import plotly.express as px

# # ============================================================
# # PAGE CONFIG
# # ============================================================
# st.set_page_config(
#     page_title  = "Student Learning Analytics",
#     page_icon   = "📚",
#     layout      = "wide"
# )

# # ============================================================
# # LOAD MODELS
# # ============================================================
# @st.cache_resource
# def load_models():
#     linear_model   = joblib.load('models/linear_model.pkl')
#     logistic_model = joblib.load('models/logistic_model.pkl')
#     kmeans_model   = joblib.load('models/kmeans_model.pkl')
#     scaler_reg     = joblib.load('models/scaler_reg.pkl')
#     scaler_clf     = joblib.load('models/scaler_clf.pkl')
#     scaler_cluster = joblib.load('models/scaler_cluster.pkl')
#     return linear_model, logistic_model, kmeans_model, scaler_reg, scaler_clf, scaler_cluster

# linear_model, logistic_model, kmeans_model, scaler_reg, scaler_clf, scaler_cluster = load_models()

# # ============================================================
# # HEADER
# # ============================================================
# st.title("📚 Student Learning Analytics Dashboard")
# st.markdown("### Predict exam performance, classify Pass/Fail, and identify learner category")
# st.divider()

# # ============================================================
# # SIDEBAR — USER INPUTS
# # ============================================================
# st.sidebar.title(" Student Parameters")
# st.sidebar.markdown("Adjust the values to match a student profile")

# math_score    = st.sidebar.slider("Math Score",          0, 100, 65)
# reading_score = st.sidebar.slider("Reading Score",       0, 100, 68)

# study_hours   = st.sidebar.selectbox(
#     "Weekly Study Hours",
#     options     = [2.5, 7.5, 12.0],
#     format_func = lambda x: "Less than 5 hrs" if x == 2.5 else ("5 to 10 hrs" if x == 7.5 else "More than 10 hrs")
# )

# parent_educ   = st.sidebar.selectbox(
#     "Parent Education Level",
#     options     = [1, 2, 3, 4, 5],
#     format_func = lambda x: {1:"High School", 2:"Some College", 3:"Associate's Degree", 4:"Bachelor's Degree", 5:"Master's Degree"}[x]
# )

# test_prep     = st.sidebar.radio(
#     "Test Preparation",
#     options     = [0, 1],
#     format_func = lambda x: "Completed ✅" if x == 0 else "Not Completed ❌"
# )

# lunch_type    = st.sidebar.radio(
#     "Lunch Type",
#     options     = [0, 1],
#     format_func = lambda x: "Free / Reduced" if x == 0 else "Standard"
# )

# practice_sport = st.sidebar.selectbox(
#     "Practice Sport",
#     options      = [0, 1, 2],
#     format_func  = lambda x: {0:"Never", 1:"Sometimes", 2:"Regularly"}[x]
# )

# nr_siblings   = st.sidebar.slider("Number of Siblings", 0, 6, 1)
# gender_male   = st.sidebar.radio("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
# is_first_child = st.sidebar.radio("Is First Child?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
# transport     = st.sidebar.radio("Transport Means", options=[0, 1], format_func=lambda x: "Public" if x == 0 else "School Bus")

# # ============================================================
# # PREPARE INPUT
# # ============================================================
# input_data = np.array([[
#     math_score, reading_score, study_hours, parent_educ,
#     test_prep, lunch_type, practice_sport,
#     nr_siblings, gender_male, is_first_child, transport
# ]])

# # Scale for regression and classification
# input_scaled_reg = scaler_reg.transform(input_data)
# input_scaled_clf = scaler_clf.transform(input_data)

# # Cluster features (ExamScore, WklyStudyHours, ParentEduc, LunchType, TestPrep, PracticeSport)
# cluster_input        = np.array([[0, study_hours, parent_educ, lunch_type, test_prep, practice_sport]])
# input_scaled_cluster = scaler_cluster.transform(cluster_input)

# # ============================================================
# # PREDICTIONS
# # ============================================================
# predicted_exam_score = linear_model.predict(input_scaled_reg)[0]
# predicted_exam_score = np.clip(predicted_exam_score, 0, 100)

# predicted_result     = logistic_model.predict(input_scaled_clf)[0]
# result_proba         = logistic_model.predict_proba(input_scaled_clf)[0]

# cluster_input_final  = np.array([[predicted_exam_score, study_hours, parent_educ,
#                                    lunch_type, test_prep, practice_sport]])
# cluster_scaled       = scaler_cluster.transform(cluster_input_final)
# cluster_id           = kmeans_model.predict(cluster_scaled)[0]

# # Map cluster to label using ExamScore ordering (same as training)
# cluster_score_map = {0: 58.64, 1: 68.59, 2: 76.40}
# sorted_clusters   = sorted(cluster_score_map, key=cluster_score_map.get)
# label_map         = {sorted_clusters[0]: "At-Risk", sorted_clusters[1]: "Average", sorted_clusters[2]: "High-Performer"}
# learner_category  = label_map.get(cluster_id, "Average")

# # ============================================================
# # RECOMMENDATION
# # ============================================================
# def get_recommendation(category, test_prep_val, study_hrs, math_sc, reading_sc):
#     recs = []
#     if category == "At-Risk":
#         recs.append("📖 Revise fundamentals daily — focus on weak subjects before moving to new topics")
#         recs.append("⏰ Increase weekly study hours consistently")
#     elif category == "Average":
#         recs.append("📝 Practice moderate to advanced problems to push to the next level")
#         recs.append("📊 Attempt weekly mock tests to track your progress")
#     else:
#         recs.append("🏆 Excellent performance! Explore competitive or advanced-level materials")
#         recs.append("🤝 Consider mentoring peers — teaching reinforces your own understanding")

#     if test_prep_val == 1:
#         recs.append("✅ Complete a test preparation course — it is the single biggest performance differentiator")
#     if study_hrs < 7.5:
#         recs.append("📚 Increase weekly study hours — students studying 7+ hrs/week consistently outperform others")
#     if math_sc < 50:
#         recs.append("➕ Focus on Math fundamentals — low math score is dragging down overall performance")
#     if reading_sc < 50:
#         recs.append("📗 Improve reading skills — reading comprehension directly impacts writing/exam performance")
#     return recs

# recommendations = get_recommendation(
#     learner_category, test_prep, study_hours, math_score, reading_score
# )

# # ============================================================
# # DISPLAY RESULTS
# # ============================================================

# # ── Row 1: 3 metric cards
# col1, col2, col3 = st.columns(3)

# with col1:
#     st.metric(
#         label = "📝 Predicted Exam Score",
#         value = f"{predicted_exam_score:.1f} / 100"
#     )

# with col2:
#     result_color = "🟢" if predicted_result == "Pass" else "🔴"
#     st.metric(
#         label = "🎯 Predicted Result",
#         value = f"{result_color} {predicted_result}",
#         delta = f"Confidence: {max(result_proba)*100:.1f}%"
#     )

# with col3:
#     cat_emoji = {"At-Risk": "⚠️", "Average": "📊", "High-Performer": "🌟"}
#     st.metric(
#         label = "👤 Learner Category",
#         value = f"{cat_emoji[learner_category]} {learner_category}"
#     )

# st.divider()

# # ── Row 2: Charts + Recommendations
# col_left, col_right = st.columns([1.2, 1])

# with col_left:
#     st.subheader("📊 Score Breakdown")

#     # Bar chart of input scores vs average
#     avg_scores = {"Math": 66, "Reading": 69, "Predicted Exam": 68}
#     student_scores = {"Math": math_score, "Reading": reading_score, "Predicted Exam": round(predicted_exam_score, 1)}

#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         name = "Your Student",
#         x    = list(student_scores.keys()),
#         y    = list(student_scores.values()),
#         marker_color = ["#2E75B6", "#2E75B6", "#1F3864"]
#     ))
#     fig.add_trace(go.Bar(
#         name = "Dataset Average",
#         x    = list(avg_scores.keys()),
#         y    = list(avg_scores.values()),
#         marker_color = ["#BDD7EE", "#BDD7EE", "#9DC3E6"]
#     ))
#     fig.update_layout(
#         barmode     = "group",
#         height      = 300,
#         margin      = dict(l=20, r=20, t=30, b=20),
#         plot_bgcolor= "white",
#         yaxis       = dict(range=[0, 110])
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     # Gauge chart for predicted exam score
#     fig2 = go.Figure(go.Indicator(
#         mode  = "gauge+number",
#         value = predicted_exam_score,
#         title = {"text": "Predicted Exam Score"},
#         gauge = {
#             "axis": {"range": [0, 100]},
#             "bar":  {"color": "#2E75B6"},
#             "steps": [
#                 {"range": [0,  50], "color": "#FFE0E0"},
#                 {"range": [50, 70], "color": "#FFF3CD"},
#                 {"range": [70, 100],"color": "#D4EDDA"},
#             ],
#             "threshold": {
#                 "line": {"color": "red", "width": 4},
#                 "thickness": 0.75,
#                 "value": 50
#             }
#         }
#     ))
#     fig2.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
#     st.plotly_chart(fig2, use_container_width=True)

# with col_right:
#     st.subheader("💡 Study Recommendations")
#     for rec in recommendations:
#         st.info(rec)

#     st.divider()
#     st.subheader("📈 Pass Probability")
#     pass_prob = result_proba[1] if predicted_result == "Pass" else result_proba[0]
#     fail_prob = 1 - pass_prob

#     fig3 = go.Figure(go.Pie(
#         labels = ["Pass", "Fail"],
#         values = [pass_prob * 100, fail_prob * 100],
#         hole   = 0.5,
#         marker = dict(colors=["#2ecc71", "#e74c3c"])
#     ))
#     fig3.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
#     st.plotly_chart(fig3, use_container_width=True)

# st.divider()

# # ── Row 3: Cluster info
# st.subheader("👥 Learner Category Breakdown")
# cluster_data = {
#     "Category":       ["At-Risk", "Average", "High-Performer"],
#     "Avg Exam Score": [58.64, 68.59, 76.40],
#     "Study Hrs/Wk":   [6.94,  6.91,  6.91],
#     "TestPrep Done":  ["8%",  "0%",  "100%"],
#     "Students":       [7818,  13454, 9368]
# }
# cluster_df = pd.DataFrame(cluster_data)
# st.dataframe(cluster_df, use_container_width=True, hide_index=True)

# # Highlight current student
# st.markdown(f"**Your student falls in the ➜ {cat_emoji[learner_category]} {learner_category} category**")

# st.divider()
# st.caption("Built for Gen AI Course — Milestone 1 | Sathvik Koriginja, Anushka Tyagi, Apoorva Choudhary")











# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import plotly.graph_objects as go
# import plotly.express as px

# # ============================================================
# # PAGE CONFIG
# # ============================================================
# st.set_page_config(
#     page_title="Student Intelligence Portal",
#     page_icon="🎓",
#     layout="wide"
# )

# # ============================================================
# # FIXED ADAPTIVE CSS (Supports Dark & Light Mode)
# # ============================================================
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

#     /* Global Font Fix */
#     html, body, [class*="css"] {
#         font-family: 'Inter', sans-serif;
#     }

#     /* Modern Card UI - Works in Dark/Light Mode */
#     .metric-container {
#         background: rgba(255, 255, 255, 0.05); /* Semi-transparent */
#         border: 1px solid rgba(128, 128, 128, 0.2);
#         padding: 24px;
#         border-radius: 16px;
#         text-align: center;
#         backdrop-filter: blur(10px);
#         margin-bottom: 10px;
#     }

#     .metric-label {
#         font-size: 0.8rem;
#         font-weight: 600;
#         color: #8899ac; /* Neutral blue-gray */
#         text-transform: uppercase;
#         letter-spacing: 0.1em;
#         margin-bottom: 8px;
#     }

#     .metric-value {
#         font-size: 2.4rem;
#         font-weight: 800;
#         /* Value colors handled via inline style for logic */
#     }

#     /* Sidebar Cleanup */
#     [data-testid="stSidebar"] {
#         border-right: 1px solid rgba(128, 128, 128, 0.1);
#     }

#     /* Section Headers */
#     .section-title {
#         font-size: 1.4rem;
#         font-weight: 700;
#         margin: 30px 0 15px 0;
#         padding-left: 12px;
#         border-left: 4px solid #3b82f6;
#     }

#     /* Insight Boxes */
#     .insight-box {
#         background: rgba(59, 130, 246, 0.1);
#         border-radius: 12px;
#         padding: 15px;
#         border: 1px solid rgba(59, 130, 246, 0.2);
#         margin-bottom: 10px;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ============================================================
# # LOAD MODELS (Logic Preserved)
# # ============================================================
# @st.cache_resource
# def load_models():
#     # Note: Replace these with your actual paths
#     linear_model   = joblib.load('models/linear_model.pkl')
#     logistic_model = joblib.load('models/logistic_model.pkl')
#     kmeans_model   = joblib.load('models/kmeans_model.pkl')
#     scaler_reg     = joblib.load('models/scaler_reg.pkl')
#     scaler_clf     = joblib.load('models/scaler_clf.pkl')
#     scaler_cluster = joblib.load('models/scaler_cluster.pkl')
#     return linear_model, logistic_model, kmeans_model, scaler_reg, scaler_clf, scaler_cluster

# linear_model, logistic_model, kmeans_model, scaler_reg, scaler_clf, scaler_cluster = load_models()

# # ============================================================
# # SIDEBAR — USER INPUTS
# # ============================================================
# with st.sidebar:
#     st.title("Student Profile")
    
#     with st.expander("📝 Academic Baseline", expanded=True):
#         math_score = st.slider("Math Score", 0, 100, 65)
#         reading_score = st.slider("Reading Score", 0, 100, 68)
#         test_prep_label = st.selectbox("Test Prep Status", ["Completed", "Not Completed"])
#         test_prep = 0 if test_prep_label == "Completed" else 1

#     with st.expander("🕒 Habits & Lifestyle"):
#         study_hours_map = {"< 5 hrs": 2.5, "5-10 hrs": 7.5, "> 10 hrs": 12.0}
#         study_hours_label = st.selectbox("Weekly Study", list(study_hours_map.keys()), index=1)
#         study_hours = study_hours_map[study_hours_label]
        
#         sport_label = st.select_slider("Sport Participation", options=["Never", "Sometimes", "Regularly"], value="Sometimes")
#         practice_sport = {"Never": 0, "Sometimes": 1, "Regularly": 2}[sport_label]

#     with st.expander("🏠 Demographics"):
#         parent_educ_map = {"High School": 1, "Some College": 2, "Associate's": 3, "Bachelor's": 4, "Master's": 5}
#         parent_educ_label = st.selectbox("Parent Education", list(parent_educ_map.keys()))
#         parent_educ = parent_educ_map[parent_educ_label]
        
#         lunch_label = st.radio("Lunch Type", ["Standard", "Free/Reduced"])
#         lunch_type = 1 if lunch_label == "Standard" else 0
        
#         gender_male = 1 if st.radio("Gender", ["Female", "Male"]) == "Male" else 0
#         nr_siblings = st.number_input("Siblings", 0, 6, 1)
#         is_first_child = 1 if st.checkbox("First Child", value=True) else 0
#         transport = 1 if st.radio("Transport", ["Public", "School Bus"]) == "School Bus" else 0

# # ============================================================
# # PREDICTIONS
# # ============================================================
# input_data = np.array([[math_score, reading_score, study_hours, parent_educ, test_prep, lunch_type, practice_sport, nr_siblings, gender_male, is_first_child, transport]])

# # Model Processing
# input_scaled_reg = scaler_reg.transform(input_data)
# input_scaled_clf = scaler_clf.transform(input_data)

# predicted_exam_score = float(np.clip(linear_model.predict(input_scaled_reg)[0], 0, 100))
# predicted_result = logistic_model.predict(input_scaled_clf)[0]
# result_proba = logistic_model.predict_proba(input_scaled_clf)[0]

# # Clustering
# cluster_input = np.array([[predicted_exam_score, study_hours, parent_educ, lunch_type, test_prep, practice_sport]])
# cluster_scaled = scaler_cluster.transform(cluster_input)
# cluster_id = int(kmeans_model.predict(cluster_scaled)[0])

# # Category Mapping
# cluster_centers_exam = kmeans_model.cluster_centers_[:, 0]
# sorted_cluster_ids = np.argsort(cluster_centers_exam)
# label_list = ["At-Risk", "Average", "High-Performer"]
# cluster_label_map = {int(sorted_cluster_ids[i]): label_list[i] for i in range(3)}
# learner_category = cluster_label_map[cluster_id]

# # ============================================================
# # DASHBOARD MAIN
# # ============================================================
# st.markdown("<h1 style='color: #3b82f6;'>Student Intelligence Portal</h1>", unsafe_allow_html=True)
# st.markdown("Predictive success tracking and academic intervention analytics.")

# st.markdown("<div class='section-title'>Executive Summary</div>", unsafe_allow_html=True)

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.markdown(f"""<div class='metric-container'>
#         <div class='metric-label'>Predicted Score</div>
#         <div class='metric-value' style='color:#3b82f6'>{predicted_exam_score:.1f}%</div>
#     </div>""", unsafe_allow_html=True)

# with col2:
#     res_color = "#10b981" if predicted_result == "Pass" else "#ef4444"
#     st.markdown(f"""<div class='metric-container'>
#         <div class='metric-label'>Outcome Prediction</div>
#         <div class='metric-value' style='color:{res_color}'>{predicted_result.upper()}</div>
#     </div>""", unsafe_allow_html=True)

# with col3:
#     cat_color = {"At-Risk": "#ef4444", "Average": "#f59e0b", "High-Performer": "#10b981"}[learner_category]
#     st.markdown(f"""<div class='metric-container'>
#         <div class='metric-label'>Learner Segment</div>
#         <div class='metric-value' style='color:{cat_color}'>{learner_category}</div>
#     </div>""", unsafe_allow_html=True)

# st.markdown("<div class='section-title'>Performance Analytics</div>", unsafe_allow_html=True)

# v_col1, v_col2 = st.columns([1.5, 1])

# with v_col1:
#     # Bar Chart with clean theme
#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=['Math', 'Reading', 'Predicted Exam'],
#         y=[math_score, reading_score, predicted_exam_score],
#         marker_color=['#64748b', '#64748b', '#3b82f6'],
#         width=0.4
#     ))
#     fig.update_layout(
#         template="plotly_dark", # Forces dark theme compatibility
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         margin=dict(t=20, b=20, l=20, r=20),
#         height=350,
#         yaxis=dict(range=[0, 105])
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with v_col2:
#     # Pass Probability Gauge
#     fig_gauge = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=result_proba[1] * 100,
#         title={'text': "Pass Confidence %", 'font': {'size': 16}},
#         gauge={
#             'axis': {'range': [0, 100], 'tickwidth': 1},
#             'bar': {'color': "#3b82f6"},
#             'bgcolor': "rgba(255,255,255,0.05)",
#             'steps': [
#                 {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
#                 {'range': [50, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
#             ]
#         }
#     ))
#     fig_gauge.update_layout(
#         template="plotly_dark",
#         paper_bgcolor='rgba(0,0,0,0)',
#         height=350,
#         margin=dict(t=50, b=20)
#     )
#     st.plotly_chart(fig_gauge, use_container_width=True)

# st.markdown("<div class='section-title'>Strategic Recommendations</div>", unsafe_allow_html=True)

# r_col1, r_col2 = st.columns(2)

# with r_col1:
#     if learner_category == "At-Risk":
#         st.markdown("<div class='insight-box'><b>🚨 Immediate Intervention:</b> Prioritize fundamental review and weekly mock tests.</div>", unsafe_allow_html=True)
#     elif learner_category == "Average":
#         st.markdown("<div class='insight-box'><b>📈 Growth Plan:</b> Increase study hours to >10 hrs/week to move into High-Performer status.</div>", unsafe_allow_html=True)
#     else:
#         st.markdown("<div class='insight-box'><b>🏆 Sustainability:</b> Maintain prep consistency and explore advanced-level peer mentoring.</div>", unsafe_allow_html=True)

# with r_col2:
#     if test_prep == 1:
#         st.markdown("<div class='insight-box'><b>💡 Key Insight:</b> Completing a test prep course is the #1 correlated factor for score boosts.</div>", unsafe_allow_html=True)
#     else:
#         st.markdown("<div class='insight-box'><b>✅ Note:</b> Student has already completed prep courses, which positively weights the prediction.</div>", unsafe_allow_html=True)

# st.caption("Milestone 1 Analytics Dashboard | Inter-Adaptive UI Engine")
















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
    page_title = "Student Learning Analytics",
    page_icon  = "📚",
    layout     = "wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1F3864, #2E75B6);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }

    .insight-box {
        background: #f0f7ff;
        border-left: 4px solid #2E75B6;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
        font-size: 0.95rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1F3864;
        margin: 16px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
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
# HEADER
# ============================================================
st.markdown("# 📚 Student Learning Analytics Dashboard")
st.markdown("**Predict exam performance · Classify Pass/Fail · Identify learner category**")
st.markdown("> Adjust the student parameters in the sidebar to see live predictions and personalised recommendations.")
st.divider()

# ============================================================
# SIDEBAR — USER INPUTS
# ============================================================
st.sidebar.title("🎛️ Student Profile")
st.sidebar.markdown("Set the student's details below:")

st.sidebar.markdown("#### 📝 Academic Scores")
math_score    = st.sidebar.slider("Math Score (0-100)",    0, 100, 65,
    help="Student's math exam score out of 100")
reading_score = st.sidebar.slider("Reading Score (0-100)", 0, 100, 68,
    help="Student's reading exam score out of 100")

st.sidebar.markdown("#### 📖 Study Habits")
study_hours_label = st.sidebar.selectbox(
    "Weekly Study Hours",
    options = ["Less than 5 hrs", "5 to 10 hrs", "More than 10 hrs"],
    help    = "How many hours the student studies per week on average"
)
study_hours_map = {"Less than 5 hrs": 2.5, "5 to 10 hrs": 7.5, "More than 10 hrs": 12.0}
study_hours     = study_hours_map[study_hours_label]

test_prep_label = st.sidebar.radio(
    "Has the student completed Test Preparation?",
    options = ["Yes — Completed ✅", "No — Not Completed ❌"],
    help    = "Test preparation is a course taken before exams to improve performance"
)
# TestPrep_none = 1 means NOT completed, 0 means completed
test_prep = 0 if test_prep_label == "Yes — Completed ✅" else 1

st.sidebar.markdown("#### 🏠 Background & Lifestyle")
parent_educ_label = st.sidebar.selectbox(
    "Parent Education Level",
    options = ["High School", "Some College", "Associate's Degree", "Bachelor's Degree", "Master's Degree"],
    help    = "Highest education level completed by either parent"
)
parent_educ_map = {"High School": 1, "Some College": 2, "Associate's Degree": 3,
                   "Bachelor's Degree": 4, "Master's Degree": 5}
parent_educ = parent_educ_map[parent_educ_label]

lunch_label = st.sidebar.radio(
    "Lunch Type",
    options = ["Standard 🍱", "Free / Reduced 🆓"],
    help    = "Standard lunch = regular fee paid. Free/Reduced = subsidised (lower income household)"
)
lunch_type = 1 if lunch_label == "Standard 🍱" else 0

sport_label = st.sidebar.selectbox(
    "Practices Sport",
    options = ["Never", "Sometimes", "Regularly"],
    help    = "How often the student participates in physical activity"
)
sport_map      = {"Never": 0, "Sometimes": 1, "Regularly": 2}
practice_sport = sport_map[sport_label]

nr_siblings = st.sidebar.slider("Number of Siblings", 0, 6, 1,
    help="Number of siblings in the household")

gender_label  = st.sidebar.radio("Gender", options=["Female", "Male"])
gender_male   = 1 if gender_label == "Male" else 0

first_child_label = st.sidebar.radio("Is First Child?", options=["Yes", "No"])
is_first_child    = 1 if first_child_label == "Yes" else 0

transport_label = st.sidebar.radio("Transport to School",
    options=["School Bus 🚌", "Public Transport 🚇"])
transport = 1 if transport_label == "School Bus 🚌" else 0

# ============================================================
# PREPARE INPUT & PREDICT
# ============================================================
input_data = np.array([[
    math_score, reading_score, study_hours, parent_educ,
    test_prep, lunch_type, practice_sport,
    nr_siblings, gender_male, is_first_child, transport
]])

input_scaled_reg = scaler_reg.transform(input_data)
input_scaled_clf = scaler_clf.transform(input_data)

# Predict exam score
predicted_exam_score = float(np.clip(linear_model.predict(input_scaled_reg)[0], 0, 100))

# Predict pass/fail
predicted_result = logistic_model.predict(input_scaled_clf)[0]
result_proba     = logistic_model.predict_proba(input_scaled_clf)[0]
pass_prob        = float(result_proba[list(logistic_model.classes_).index('Pass')] * 100)
fail_prob        = 100 - pass_prob

# Cluster prediction
cluster_input         = np.array([[predicted_exam_score, study_hours, parent_educ,
                                    lunch_type, test_prep, practice_sport]])
cluster_scaled        = scaler_cluster.transform(cluster_input)
cluster_id            = int(kmeans_model.predict(cluster_scaled)[0])

# Map cluster id to label using known cluster center scores
cluster_centers_exam  = kmeans_model.cluster_centers_[:, 0]  # ExamScore is first feature
sorted_cluster_ids    = np.argsort(cluster_centers_exam)
label_list            = ["At-Risk", "Average", "High-Performer"]
cluster_label_map     = {int(sorted_cluster_ids[i]): label_list[i] for i in range(3)}
learner_category      = cluster_label_map[cluster_id]

# ============================================================
# RECOMMENDATIONS
# ============================================================
def get_recommendations(category, test_prep_val, study_hrs, math_sc, reading_sc, exam_sc):
    recs = []

    if category == "At-Risk":
        recs.append(("🚨", "You are in the At-Risk group.", "Focus on revising fundamentals daily before attempting new topics. Consistency is more important than intensity."))
    elif category == "Average":
        recs.append(("📈", "You are performing at an Average level.", "To move to High-Performer, start practising advanced problems and attempt at least one mock test per week."))
    else:
        recs.append(("🏆", "Excellent! You are a High-Performer.", "Keep up the consistency. Consider exploring competitive materials or helping peers — teaching reinforces your own learning."))

    if test_prep_val == 1:
        recs.append(("✅", "Complete a Test Preparation Course.", "Our data shows 100% of High-Performers completed test prep vs 0% of Average students. It is the single biggest differentiator."))

    if study_hrs < 7.5:
        recs.append(("⏰", "Increase weekly study hours.", f"You currently study {study_hrs} hrs/week. Students studying 7+ hrs/week consistently score higher."))

    if math_sc < 55:
        recs.append(("➕", "Math score needs attention.", f"Your math score ({math_sc}) is below average. Focus on practising math problems daily."))

    if reading_sc < 55:
        recs.append(("📗", "Reading score needs improvement.", f"Your reading score ({reading_sc}) is low. Strong reading directly improves writing and exam performance."))

    return recs

recommendations = get_recommendations(
    learner_category, test_prep, study_hours,
    math_score, reading_score, predicted_exam_score
)

# ============================================================
# SECTION 1 — PREDICTION RESULTS
# ============================================================
st.markdown("## 🎯 Prediction Results")
st.markdown("Based on the student profile you entered, here is what our ML models predict:")

col1, col2, col3 = st.columns(3)

with col1:
    score_color = "#2ecc71" if predicted_exam_score >= 50 else "#e74c3c"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📝 Predicted Exam Score</div>
        <div class="metric-value" style="color:{score_color}">{predicted_exam_score:.1f}<span style="font-size:1rem"> / 100</span></div>
        <div class="metric-label">Predicted by Linear Regression model</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    res_color = "#2ecc71" if predicted_result == "Pass" else "#e74c3c"
    res_emoji = "✅" if predicted_result == "Pass" else "❌"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🎯 Pass / Fail Result</div>
        <div class="metric-value" style="color:{res_color}">{res_emoji} {predicted_result}</div>
        <div class="metric-label">Confidence: {max(pass_prob, fail_prob):.1f}% | Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    cat_emoji  = {"At-Risk": "⚠️", "Average": "📊", "High-Performer": "🌟"}
    cat_colors = {"At-Risk": "#e74c3c", "Average": "#f39c12", "High-Performer": "#2ecc71"}
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">👤 Learner Category</div>
        <div class="metric-value" style="color:{cat_colors[learner_category]}">{cat_emoji[learner_category]} {learner_category}</div>
        <div class="metric-label">Identified by K-Means Clustering</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ============================================================
# SECTION 2 — MODEL ACCURACY METRICS
# ============================================================
st.markdown("## 📊 How Accurate Are Our Models?")
st.markdown("These are the performance metrics our models achieved during training and testing on 30,640 student records:")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Linear Regression R²",   "93.97%",  help="Model explains 94% of variance in exam scores. Higher is better (max 100%)")
with m2:
    st.metric("Regression MAE",          "3.04 marks", help="On average, predicted exam score is off by only 3 marks out of 100")
with m3:
    st.metric("Classification Accuracy", "91.76%",  help="Model correctly classifies 92 out of every 100 students as Pass or Fail")
with m4:
    st.metric("F1 Score",                "0.9176",  help="Balanced measure of precision and recall. Close to 1.0 is excellent")
with m5:
    st.metric("At-Risk Detection Rate",  "92%",     help="Model correctly identifies 92% of all students who are actually at risk of failing")

st.divider()

# ============================================================
# SECTION 3 — CHARTS + RECOMMENDATIONS
# ============================================================
st.markdown("## 📈 Score Analysis & Recommendations")
col_left, col_right = st.columns([1.3, 1])

with col_left:
    # ── Chart 1: Student vs Average comparison
    st.markdown("#### Your Student vs Dataset Average")
    st.caption("This chart compares the student's scores against the average scores of all 30,640 students in our dataset.")

    categories    = ["Math Score", "Reading Score", "Predicted Exam Score"]
    student_vals  = [math_score, reading_score, round(predicted_exam_score, 1)]
    average_vals  = [66.1, 69.2, 68.1]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name         = "This Student",
        x            = categories,
        y            = student_vals,
        marker_color = "#2E75B6",
        text         = student_vals,
        textposition = "outside"
    ))
    fig1.add_trace(go.Bar(
        name         = "Dataset Average",
        x            = categories,
        y            = average_vals,
        marker_color = "#BDD7EE",
        text         = average_vals,
        textposition = "outside"
    ))
    fig1.update_layout(
        barmode      = "group",
        height       = 320,
        yaxis        = dict(range=[0, 115], title="Score (out of 100)"),
        plot_bgcolor = "white",
        margin       = dict(l=10, r=10, t=10, b=10),
        legend       = dict(orientation="h", y=-0.2)
    )
    fig1.add_hline(y=50, line_dash="dash", line_color="red",
                   annotation_text="Pass Threshold (50)", annotation_position="top right")
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Gauge for predicted exam score
    st.markdown("#### Predicted Exam Score Gauge")
    st.caption("Green zone = Pass (50+), Yellow zone = borderline (40-50), Red zone = Fail (below 40)")

    fig2 = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = predicted_exam_score,
        delta = {"reference": 50, "increasing": {"color": "#2ecc71"}, "decreasing": {"color": "#e74c3c"}},
        title = {"text": "Predicted Exam Score / 100"},
        gauge = {
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": "#2E75B6", "thickness": 0.3},
            "steps": [
                {"range": [0,  40], "color": "#FFD5D5"},
                {"range": [40, 50], "color": "#FFF3CD"},
                {"range": [50, 100],"color": "#D4EDDA"},
            ],
            "threshold": {
                "line":      {"color": "red", "width": 3},
                "thickness": 0.75,
                "value":     50
            }
        }
    ))
    fig2.update_layout(height=270, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    # ── Pass probability donut
    st.markdown("#### Pass / Fail Probability")
    st.caption("How confident the model is in its Pass/Fail prediction for this student.")

    fig3 = go.Figure(go.Pie(
        labels   = [f"Pass ({pass_prob:.1f}%)", f"Fail ({fail_prob:.1f}%)"],
        values   = [pass_prob, fail_prob],
        hole     = 0.55,
        marker   = dict(colors=["#2ecc71", "#e74c3c"]),
        textinfo = "label+percent"
    ))
    fig3.update_layout(
        height        = 260,
        margin        = dict(l=10, r=10, t=10, b=10),
        showlegend    = False,
        annotations   = [dict(
            text      = f"{'PASS' if predicted_result == 'Pass' else 'FAIL'}",
            x=0.5, y=0.5, font_size=22, font_color="#1F3864",
            showarrow = False, font=dict(weight="bold")
        )]
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Recommendations
    st.markdown("#### 💡 Personalised Recommendations")
    st.caption("Based on this student's profile and learner category:")
    for emoji, title, detail in recommendations:
        st.markdown(f"""
        <div class="insight-box">
            <strong>{emoji} {title}</strong><br>
            <span style="color:#444">{detail}</span>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ============================================================
# SECTION 4 — LEARNER CATEGORIES EXPLAINED
# ============================================================
st.markdown("## 👥 What Are the Learner Categories?")
st.markdown("Our K-Means clustering model grouped all 30,640 students into 3 categories based on their exam scores, study habits, and test preparation. Here is what each category looks like:")

c1, c2, c3 = st.columns(3)

def category_card(col, emoji, name, color, exam, study, testprep, count, pct, description):
    with col:
        st.markdown(f"""
        <div style="background:{color}18; border:2px solid {color};
                    border-radius:12px; padding:16px; text-align:center; height:100%">
            <div style="font-size:2rem">{emoji}</div>
            <div style="font-size:1.2rem; font-weight:bold; color:{color}">{name}</div>
            <div style="font-size:0.85rem; color:#555; margin:8px 0">{description}</div>
            <hr style="border-color:{color}40">
            <div style="text-align:left; font-size:0.88rem">
                📝 Avg Exam Score: <b>{exam}</b><br>
                ⏰ Study Hrs/Wk: <b>{study}</b><br>
                ✅ Completed Test Prep: <b>{testprep}</b><br>
                👥 Students: <b>{count:,} ({pct})</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

category_card(c1, "⚠️", "At-Risk",        "#e74c3c",
    exam=58.64, study="6.94 hrs", testprep="Only 8%",  count=7818,  pct="25.5%",
    description="Students struggling with performance. Need focused support and intervention.")

category_card(c2, "📊", "Average",         "#f39c12",
    exam=68.59, study="6.91 hrs", testprep="0%",       count=13454, pct="43.9%",
    description="Students performing at a moderate level. Can improve with better preparation.")

category_card(c3, "🌟", "High-Performer",  "#2ecc71",
    exam=76.40, study="6.91 hrs", testprep="100%",     count=9368,  pct="30.6%",
    description="Top performing students. All completed test preparation before exams.")

st.divider()

# ── Key insight callout
st.markdown("### 🔍 The Most Important Finding")
st.info("""
**Test Preparation is the single biggest differentiator between Average and High-Performer students.**

- 🌟 High-Performers → **100%** completed test preparation
- 📊 Average students → **0%** completed test preparation
- ⚠️ At-Risk students → only **8%** completed test preparation

Interestingly, all three groups study approximately the **same number of hours per week (~6.9 hrs)**.
This means it's not just about HOW LONG you study — it's about HOW you prepare.
""")

st.divider()

# ============================================================
# SECTION 5 — WHAT EACH MODEL DOES
# ============================================================
with st.expander("ℹ️ How Do These Models Work? (Click to expand)"):
    st.markdown("""
    ### The 3 ML Models Behind This Dashboard

    #### 1. 📈 Linear Regression — Predicts Exam Score
    Linear Regression finds the mathematical relationship between input features (Math score,
    Reading score, study hours, etc.) and the target exam score. It draws the best-fit line
    through the data points. Our model achieves **R² = 0.94** meaning it explains 94% of
    the variation in exam scores.

    #### 2. 🎯 Logistic Regression — Predicts Pass or Fail
    Logistic Regression calculates the probability of a student passing based on their profile.
    If the probability is above 50%, it predicts Pass. Our model achieves **91.76% accuracy**
    and correctly identifies **92% of at-risk students** — which is the most critical metric
    for a learning analytics system.

    #### 3. 👥 K-Means Clustering — Groups Students into Categories
    K-Means Clustering groups students based on similarity without using labels. It found
    **3 natural groups** in the data corresponding to At-Risk, Average, and High-Performer
    students. The Silhouette Score of **0.21** indicates the clusters are reasonably well
    separated given that student behaviour naturally overlaps.

    ---
    **Dataset:** Students Exam Scores Extended | 30,640 records | Kaggle (desalegngeb)
    """)

st.divider()
st.caption("📚 Built for Gen AI Course — Milestone 1  |  Sathvik Koriginja · Anushka Tyagi · Apoorva Choudhary")