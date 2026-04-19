# Intelligent Learning Analytics & Agentic AI Study Coach

**Unified Project: Milestone 1 + Milestone 2**

---

## Quick Navigation

- [Milestone 1 — ML-Based Student Performance Prediction](#milestone-1--ml-based-student-performance-prediction)
- [Milestone 2 — Agentic AI Study Coach](#milestone-2--agentic-ai-study-coach)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)

---

# Milestone 1 — ML-Based Student Performance Prediction

## Overview

Milestone 1 implements a machine learning pipeline that predicts student academic performance using demographic, academic, and behavioural data.

The system provides three outputs:
- Exam score prediction (regression)
- Pass/Fail classification
- Learner segmentation using clustering

A Streamlit dashboard is used to interactively visualize predictions.

Live Demo:  
https://predictive-learning-analytics-ml.streamlit.app/

---

## Models & Performance

| Model | Task | Performance |
|------|------|------------|
| Linear Regression | Predict ExamScore | R² = 0.9397 |
| Logistic Regression | Pass/Fail | Accuracy = 91.76% |
| K-Means Clustering | Learner Segmentation | Silhouette = 0.2112 |

---

## Key Design Decisions

- WritingScore used as proxy target (no leakage risk)
- Median-based Pass/Fail threshold (69.0)
- Manual ordinal encoding for education-related features
- Midpoint encoding for study hours
- class_weight='balanced' instead of SMOTE
- k=3 clustering aligned with interpretability requirement
- Train-only scaling to avoid data leakage

---

## Dataset

- 30,640 student records
- 14 original features
- 11 engineered features

Source: Kaggle Students Exam Scores Extended Dataset

---

## Preprocessing Pipeline

1. Data cleaning and normalization  
2. Missing value imputation  
3. Ordinal + one-hot encoding  
4. Outlier handling (IQR method)  
5. Feature scaling (StandardScaler)  
6. Train-test split (80/20)  
7. Model training and evaluation  

---

# Milestone 2 — Agentic AI Study Coach

## Overview

Milestone 2 transforms the ML system into a **conversational agentic AI tutor**.

Instead of static inputs, students interact through natural language, and the system dynamically decides:

- What analysis to run
- What knowledge to retrieve
- Whether to generate a study plan or quiz
- How to respond using ML + LLM reasoning

This is implemented using **LangGraph-based multi-agent orchestration**.

---

## Key Features

### 1. Conversational AI Interface
Students interact via chat instead of form inputs.

### 2. LangGraph Agent System
A structured graph of specialized nodes:

- Analyser Node (ML inference)
- Retriever Node (RAG system)
- Planner Node (study plan generation)
- Quizzer Node (MCQ system)
- End Node (response finalization)

---

### 3. Machine Learning Integration
- Extracts student data from natural language
- Runs all Milestone 1 models dynamically
- Predicts:
  - Exam score
  - Pass/Fail status
  - Learner category

---

### 4. Retrieval-Augmented Generation (RAG)
- FAISS vector database
- Sentence-transformer embeddings (MiniLM)
- Academic knowledge base
- Optional Tavily web search fallback

---

### 5. Personalized Study Plans
- 7-day structured study plan
- Generated from retrieved knowledge
- Adapted to learner category:
  - At-Risk → fundamentals + revision
  - Average → balanced learning
  - High Performer → advanced + challenge tasks

---

### 6. AI Quiz System
- 5 MCQs per session
- Auto-generated from retrieved content
- Automatic grading + explanations
- Performance-based feedback

---

### 7. Persistent Memory
- PostgreSQL (Neon)
- Stores:
  - chat history
  - agent state
  - session data
- Allows conversation resumption

---

### 8. Academic Guardrails
- Blocks non-academic queries
- Prevents cheating requests
- Ensures safe tutoring behavior

---

# System Architecture

```

User (Streamlit UI)
|
v
LangGraph Agent (Master Node)
|
v
+----------------------------------+
| Specialist Nodes                 |
| - Analyser (ML Models)           |
| - Retriever (RAG System)         |
| - Planner (Study Plan Generator) |
| - Quizzer (MCQ System)           |
| - End Node                       |
+----------------------------------+
|
v
PostgreSQL (Persistent Memory)

````

---

# Tech Stack

| Layer | Technology |
|------|------------|
| ML Models | Scikit-Learn |
| Agent Framework | LangGraph |
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| Backend | Python |
| UI | Streamlit |
| Database | PostgreSQL (Neon) |
| Deployment | Streamlit Cloud |
| Web Search | Tavily API |

---

# Getting Started

## Clone Repository

```bash
git clone https://github.com/sathvik89/Predictive-Learning-Analytics_ML.git
cd Predictive-Learning-Analytics_ML
````

---

## Create Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

# Project Structure

```
Predictive-Learning-Analytics_ML/
├── app.py
├── requirements.txt
├── models/
│   ├── linear_model.pkl
│   ├── logistic_model.pkl
│   ├── kmeans_model.pkl
│   ├── scaler_reg.pkl
│   ├── scaler_clf.pkl
│   └── scaler_cluster.pkl
└── README.md
```

---

# Roadmap

## Completed

* ML-based student prediction system
* Streamlit dashboard
* LangGraph agent system
* RAG pipeline
* Quiz system
* Persistent memory

---

## Future Work

* User authentication system
* Step-by-step adaptive quiz flow
* Larger academic knowledge base
* Student analytics dashboard
* Mobile UI optimization
* Personalized long-term learning tracking

---

# Project Summary

This project demonstrates the evolution from a traditional machine learning pipeline into a fully agentic AI tutoring system. It integrates predictive modeling, retrieval-augmented generation, and multi-agent orchestration to create a system that not only analyzes student performance but actively supports learning through conversation, planning, and assessment.

---

**Built for Gen AI Course — Milestone 1 & 2**
Sathvik Koriginja | Anushka Tyagi | Apoorva Choudhary

```
