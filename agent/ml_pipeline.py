import joblib
import pandas as pd
import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), "..", "models")

def _load(name):
    return joblib.load(os.path.join(_BASE, name))

# Lazy-load models once at import time
try:
    reg_model     = _load("linear_model.pkl")
    clf_model     = _load("logistic_model.pkl")
    cluster_model = _load("kmeans_model.pkl")
    scaler_reg    = _load("scaler_reg.pkl")
    scaler_clf    = _load("scaler_clf.pkl")
    scaler_cluster= _load("scaler_cluster.pkl")
    MODELS_LOADED = True
except Exception as e:
    print(f"[ml_pipeline] WARNING – Could not load models: {e}")
    MODELS_LOADED = False


# ─────────────────────────────────────────────
# CLUSTER → CATEGORY MAPPING
# ─────────────────────────────────────────────
CATEGORY_MAP = {0: "At-Risk", 1: "Average", 2: "High-Performer"}
#median values
FEATURE_DEFAULTS = {
    "math": 67.0,
    "reading": 70.0,
    "study_hours": 7.5,
    "siblings": 2.0,
    "parent_educ": 2,
    "sport": 1,
    "test_prep": "none",
    "lunch": "standard",
    "gender": "female",
    "is_first_child": "yes",
    "transport": "school_bus",
}

FEATURE_LABELS = {
    "math": "Math score",
    "reading": "Reading score",
    "study_hours": "Weekly study hours",
    "siblings": "Number of siblings",
    "parent_educ": "Parent education level",
    "sport": "Sports participation",
    "test_prep": "Test preparation",
    "lunch": "Lunch type",
    "gender": "Gender",
    "is_first_child": "First-child status",
    "transport": "Transport mode",
}


def _build_category_map() -> dict[int, str]:
    """Map raw KMeans labels by score order, matching the dashboard logic."""
    if not MODELS_LOADED:
        return CATEGORY_MAP
    try:
        order = cluster_model.cluster_centers_[:, 0].argsort()
        return {
            int(order[0]): "At-Risk",
            int(order[1]): "Average",
            int(order[2]): "High-Performer",
        }
    except Exception as e:
        print(f"[ml_pipeline] WARNING – Could not derive cluster map: {e}")
        return CATEGORY_MAP


DERIVED_CATEGORY_MAP = _build_category_map()


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────
def run_ml_pipeline(user_data: dict) -> dict:
    """
    Accepts a dict of student features (any subset) and returns:
      { predicted_score, status, category }
    Missing fields are filled with sensible defaults.
    """
    if not MODELS_LOADED:
        return {
            "predicted_score": "N/A",
            "status": "Unknown",
            "category": "Unknown",
            "supplied_fields": sorted(user_data.keys()),
            "assumed_defaults": {},
            "assumption_note": "ML model files are unavailable, so no prediction could be generated.",
        }

    # ── 1. Merge supplied data with defaults ──────────────────────────────────
    assumed_defaults = {
        key: value
        for key, value in FEATURE_DEFAULTS.items()
        if key not in user_data or user_data.get(key) is None
    }
    data = {
        "MathScore":                user_data.get("math", FEATURE_DEFAULTS["math"]),
        "ReadingScore":             user_data.get("reading", FEATURE_DEFAULTS["reading"]),
        "WklyStudyHours":           user_data.get("study_hours", FEATURE_DEFAULTS["study_hours"]),
        "NrSiblings":               user_data.get("siblings", FEATURE_DEFAULTS["siblings"]),
        "ParentEduc":               user_data.get("parent_educ", FEATURE_DEFAULTS["parent_educ"]),
        "PracticeSport":            user_data.get("sport", FEATURE_DEFAULTS["sport"]),
        "TestPrep_none":            1 if user_data.get("test_prep", FEATURE_DEFAULTS["test_prep"]) == "none" else 0,
        "LunchType_standard":       1 if user_data.get("lunch", FEATURE_DEFAULTS["lunch"]) == "standard" else 0,
        "Gender_male":              1 if user_data.get("gender", FEATURE_DEFAULTS["gender"]) == "male" else 0,
        "IsFirstChild_yes":         1 if user_data.get("is_first_child", FEATURE_DEFAULTS["is_first_child"]) == "yes" else 0,
        "TransportMeans_school_bus":1 if user_data.get("transport", FEATURE_DEFAULTS["transport"]) == "school_bus" else 0,
    }

    # ── 2. Feature vector for regression / classification (11 features) ───────
    feat_cols = [
        "MathScore", "ReadingScore", "WklyStudyHours", "ParentEduc",
        "TestPrep_none", "LunchType_standard", "PracticeSport",
        "NrSiblings", "Gender_male", "IsFirstChild_yes", "TransportMeans_school_bus",
    ]
    X_input = pd.DataFrame([data])[feat_cols]

    # ── 3. Regression ─────────────────────────────────────────────────────────
    X_reg_scaled     = scaler_reg.transform(X_input)
    pred_exam_score  = reg_model.predict(X_reg_scaled)[0]

    # ── 4. Classification ─────────────────────────────────────────────────────
    X_clf_scaled = scaler_clf.transform(X_input)
    pass_fail    = clf_model.predict(X_clf_scaled)[0]

    # ── 5. Clustering (6 features) ────────────────────────────────────────────
    cluster_data = {
        "ExamScore":          pred_exam_score,
        "WklyStudyHours":     data["WklyStudyHours"],
        "ParentEduc":         data["ParentEduc"],
        "LunchType_standard": data["LunchType_standard"],
        "TestPrep_none":      data["TestPrep_none"],
        "PracticeSport":      data["PracticeSport"],
    }
    cluster_cols   = ["ExamScore", "WklyStudyHours", "ParentEduc",
                      "LunchType_standard", "TestPrep_none", "PracticeSport"]
    X_cluster      = pd.DataFrame([cluster_data])[cluster_cols]
    X_cluster_scaled = scaler_cluster.transform(X_cluster)
    cluster_id     = cluster_model.predict(X_cluster_scaled)[0]

    return {
        "predicted_score": round(float(pred_exam_score), 2),
        "status":          str(pass_fail),
        "category":        DERIVED_CATEGORY_MAP.get(int(cluster_id), "Unknown"),
        "supplied_fields": sorted(user_data.keys()),
        "assumed_defaults": {
            FEATURE_LABELS.get(key, key): value for key, value in assumed_defaults.items()
        },
        "assumption_note": (
            "Some missing inputs were filled with baseline values from the training setup. "
            "Providing more profile details will make the analysis more personalised."
            if assumed_defaults else
            "All required model inputs were supplied or already available in the saved profile."
        ),
    }
