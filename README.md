# Intelligent Learning Analytics System

**Milestone 1 — ML-Based Student Performance Prediction**  
| February 2026

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

---

## Overview

This project implements a machine learning pipeline for student learning analytics. The system ingests student demographic, academic, and behavioural data to produce three outputs: a predicted exam score, a Pass/Fail classification, and a learner segmentation category. A Streamlit dashboard exposes these predictions through an interactive interface deployable on Streamlit Community Cloud.

This is **Milestone 1** of a two-phase project. Milestone 2 will extend the system into an agentic AI study coach powered by LangGraph and an open-source LLM.

**Live Demo:** [student-analytics.streamlit.app](https://your-app-link.streamlit.app)

---

## Team

| Name |
|------|
| Sathvik Koriginja |
| Anushka Tyagi |
| Apoorva Choudhary |

---

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Models & Results](#models--results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Dashboard Features](#dashboard-features)
- [Design Decisions](#design-decisions)
- [Roadmap](#roadmap)

---

## Dataset

**Source:** [Students Exam Scores Extended — Kaggle (desalegngeb)](https://www.kaggle.com/datasets/desalegngeb/students-exam-scores)

| Property | Value |
|----------|-------|
| Records | 30,640 |
| Original features | 14 |
| Final features (X) | 11 |
| Regression target | ExamScore (continuous, 0–100) |
| Classification target | Result (Pass / Fail) |

The dataset contains no explicit dependent variable. `WritingScore` was selected as the regression target as it is independently measured and genuinely predictable from Math and Reading scores alongside behavioural features, without introducing data leakage. The column was renamed to `ExamScore` throughout the pipeline.

### Feature Summary

| Feature | Original Type | Preprocessing Applied |
|---------|--------------|----------------------|
| MathScore | Numerical | Outlier clipping (IQR) |
| ReadingScore | Numerical | Outlier clipping (IQR) |
| WklyStudyHours | Categorical (<5 / 5-10 / >10) | Converted to midpoint numeric (2.5 / 7.5 / 12.0) |
| ParentEduc | Ordinal (6 levels) | Manual ordinal mapping (1–5) |
| TestPrep | Binary | get_dummies encoding |
| LunchType | Binary | get_dummies encoding |
| PracticeSport | Ordinal (3 levels) | Manual ordinal mapping (0–2) |
| NrSiblings | Numerical | Median imputation, IQR clipping |
| Gender | Binary | get_dummies encoding |
| EthnicGroup | Nominal (A–E) | One-hot encoding (drop_first=True) |
| ParentMaritalStatus | Nominal (4 levels) | One-hot encoding (drop_first=True) |

---

## Preprocessing Pipeline

```
Step 01  Load dataset, drop unnamed index column
Step 02  Inspect shape, dtypes, missing value counts
Step 03  Lowercase and strip whitespace from all string columns
Step 04  Standardise category names — merge 'some high school' into 'high_school'
Step 05  Fill missing values — mode for categorical, median for numerical
Step 06  Remove duplicate rows
Step 07  Encode columns — ordinal mapping, get_dummies, one-hot encoding
Step 08  Convert bool columns to int (0/1)
Step 09  Clip outliers using IQR method [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
Step 10  Engineer targets — ExamScore renamed, Pass/Fail threshold at median (69.0)
Step 11  Train/test split — 80/20, stratified on classification target
Step 12  Scale features — StandardScaler fit on train set only
```

**Notable decisions:**

- `WklyStudyHours` converted to midpoint values rather than ordinal ranks to preserve real numeric magnitude
- `LabelEncoder` avoided for ordinal columns — it assigns alphabetical order which is incorrect for education levels
- Pass/Fail threshold derived from the **median of ExamScore (69.0)** rather than a fixed value, ensuring balanced classes and removing subjective cutoff assumptions
- `StandardScaler` fit exclusively on training data to prevent test set information leaking into the model

---

## Models & Results

### Linear Regression — ExamScore Prediction

| Metric | Test Set | Cross-Validation (5-Fold) |
|--------|----------|--------------------------|
| R² Score | 0.9397 | 0.9394 ± 0.0021 |
| MAE | 3.04 marks | — |
| RMSE | 3.78 marks | — |

The model explains 94% of variance in ExamScore. The negligible standard deviation across folds (±0.0021) confirms stability and the absence of overfitting. The correlation between Math/Reading scores and ExamScore reflects genuine shared academic ability — not a mathematical identity — and does not constitute data leakage.

---

### Logistic Regression — Pass/Fail Classification

| Metric | Value | Cross-Validation (5-Fold) |
|--------|-------|--------------------------|
| Accuracy | 91.76% | 92.47% ± 0.55% |
| Precision (weighted) | 0.9177 | — |
| Recall (weighted) | 0.9176 | — |
| F1 Score (weighted) | 0.9176 | — |
| Fail Recall | 0.92 | — |

`class_weight='balanced'` was applied to handle the natural class imbalance without synthetic data generation (SMOTE), preserving data integrity. The model correctly identifies 92% of at-risk students — the primary metric of interest for a learning analytics system.

---

### K-Means Clustering — Learner Segmentation

| Metric | Value |
|--------|-------|
| Clusters (k) | 3 |
| Silhouette Score | 0.2112 |
| Davies-Bouldin Index | 1.7311 |

The optimal k was evaluated using the Elbow Method, Silhouette Score, and Davies-Bouldin Index across k = 2 to 8. k=3 was selected as it maps directly to the three learner categories required by the project specification. The marginal silhouette improvement at k=5 (0.2211) did not justify the loss of interpretability.

**Cluster Profiles:**

| Category | Avg Exam Score | Avg Study Hrs/Wk | Test Prep Completed | Count |
|----------|---------------|-----------------|---------------------|-------|
| At-Risk | 58.64 | 6.94 | 8% | 7,818 (25.5%) |
| Average | 68.59 | 6.91 | 0% | 13,454 (43.9%) |
| High-Performer | 76.40 | 6.91 | 100% | 9,368 (30.6%) |

**Primary finding:** Test preparation completion is the single most significant differentiator between learner categories. All three groups study a comparable number of hours per week (~6.9 hrs), indicating that study duration alone is insufficient — structured preparation has a measurable impact on outcomes.

---

## Project Structure

```
student-analytics/
├── app.py                   # Streamlit application entry point
├── requirements.txt         # Python dependencies
├── README.md
└── models/
    ├── linear_model.pkl     # Trained LinearRegression
    ├── logistic_model.pkl   # Trained LogisticRegression
    ├── kmeans_model.pkl     # Trained KMeans (k=3)
    ├── scaler_reg.pkl       # StandardScaler — regression
    ├── scaler_clf.pkl       # StandardScaler — classification
    └── scaler_cluster.pkl   # StandardScaler — clustering
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/sathvik89/Predictive-Learning-Analytics_ML.git
cd Predictive-Learning-Analytics_ML

# Create and activate a virtual environment
# On macOS / Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

### Dependencies

```
streamlit==1.32.0
pandas==2.1.0
numpy==1.26.0
scikit-learn==1.4.0
joblib==1.3.0
plotly==5.18.0
```

---

## Dashboard Features

| Feature | Description |
|---------|-------------|
| Interactive sidebar | 11 configurable student parameters with real-time updates |
| Exam score prediction | Predicted score with gauge visualisation and delta vs threshold |
| Pass/Fail classification | Predicted class with confidence probability breakdown |
| Learner category | Cluster assignment with category profile comparison |
| Score comparison chart | Student scores vs dataset averages with pass threshold line |
| Personalised recommendations | Rule-based study advice derived from cluster assignment and feature values |
| Model performance metrics | R², MAE, Accuracy, F1, and At-Risk detection rate displayed inline |
| Model explainer | Plain-language description of each model for non-technical users |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| WritingScore selected as regression target | No explicit dependent variable in dataset. WritingScore is independently measured and predictable without leakage |
| Median-based Pass/Fail threshold (69.0) | Avoids arbitrary fixed cutoffs. Ensures balanced class distribution for model training |
| Manual ordinal encoding for ParentEduc and PracticeSport | LabelEncoder produces alphabetical ordering which is semantically incorrect for these features |
| Midpoint conversion for WklyStudyHours | Preserves real numeric magnitude. Rank encoding (0/1/2) would lose the proportional difference between study hour bands |
| class_weight='balanced' over SMOTE | Handles imbalance without generating synthetic records. Preserves the integrity of the original data distribution |
| k=3 enforced for clustering | Directly corresponds to the three actionable learner categories. Silhouette score at k=3 (0.2112) vs k=5 (0.2211) — marginal difference does not justify reduced interpretability |
| Scaler fit on training data only | Fitting on the full dataset would allow test set distribution to influence model training — a standard data leakage error |

---

## Roadmap

**Milestone 2 — Agentic AI Study Coach**

| Component | Technology |
|-----------|-----------|
| Agent workflow and state management | LangGraph |
| Personalised study plan generation | Open-source LLM |
| Learning resource retrieval | RAG — Chroma / FAISS vector store |
| Session memory and progress tracking | LangGraph persistent state |
| Conversational interface | Multi-turn LLM chat with tool use |
| Deployment | Hugging Face Spaces / Streamlit Community Cloud |

---

## License

This project is licensed under the MIT License.

---

*Built for Gen AI Course — Milestone 1 | Sathvik Koriginja, Anushka Tyagi, Apoorva Choudhary*
