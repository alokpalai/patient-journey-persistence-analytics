import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

WAREHOUSE_DIR = "data/warehouse"
MODEL_PATH = "outputs/models/xgboost_persistence_model.pkl"
OUTPUT_DIR = "outputs/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = ["age_at_first_therapy", "comorbidity_count", "prior_encounter_count", "switched_therapy"]


def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(f"{WAREHOUSE_DIR}/model_features.parquet")

    X = df[FEATURE_COLS].copy()
    X["switched_therapy"] = X["switched_therapy"].astype(int)

    return model, X, df


def compute_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def plot_global_importance(shap_values, X):
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_summary.png")


def explain_one_patient(explainer, shap_values, X, df, patient_index=0):
    patient_key = df.iloc[patient_index]["patient_key"]
    actual_outcome = df.iloc[patient_index]["discontinued"]

    print(f"\nExplaining patient {patient_key} (actual outcome: discontinued={actual_outcome})")
    print(X.iloc[patient_index])

    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values[patient_index],
        X.iloc[patient_index],
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_patient_{patient_index}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved shap_patient_{patient_index}.png")


if __name__ == "__main__":
    model, X, df = load_model_and_data()
    explainer, shap_values = compute_shap_values(model, X)

    plot_global_importance(shap_values, X)

    # explain one high-risk and one low-risk patient for contrast
    predicted_proba = model.predict_proba(X)[:, 1]
    highest_risk_idx = int(np.argmax(predicted_proba))
    lowest_risk_idx = int(np.argmin(predicted_proba))

    explain_one_patient(explainer, shap_values, X, df, patient_index=highest_risk_idx)
    explain_one_patient(explainer, shap_values, X, df, patient_index=lowest_risk_idx)

    # print mean absolute SHAP value per feature — a simple importance ranking
    mean_abs_shap = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nFeature importance (mean |SHAP value|):")
    print(mean_abs_shap)