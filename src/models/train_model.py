import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb

WAREHOUSE_DIR = "data/warehouse"


def load_data():
    df = pd.read_parquet(f"{WAREHOUSE_DIR}/model_features.parquet")
    return df


def temporal_split(df, test_fraction=0.2):
    df = df.sort_values("first_therapy_date").reset_index(drop=True)

    split_index = int(len(df) * (1 - test_fraction))
    split_date = df.loc[split_index, "first_therapy_date"]

    train = df[df["first_therapy_date"] < split_date].copy()
    test = df[df["first_therapy_date"] >= split_date].copy()

    print(f"Split date: {split_date}")
    print(f"Train: {len(train)} patients ({train['first_therapy_date'].min()} to {train['first_therapy_date'].max()})")
    print(f"Test:  {len(test)} patients ({test['first_therapy_date'].min()} to {test['first_therapy_date'].max()})")

    return train, test

FEATURE_COLS = ["age_at_first_therapy", "comorbidity_count", "prior_encounter_count", "switched_therapy"]
TARGET_COL = "discontinued"


def prepare_xy(df):
    X = df[FEATURE_COLS].copy()
    X["switched_therapy"] = X["switched_therapy"].astype(int)
    y = df[TARGET_COL].astype(int)
    return X, y


def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAUC: {auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")

    # top-decile capture: of the 10% of patients the model scores as highest risk,
    # what fraction actually discontinued?
    test_results = pd.DataFrame({"y_true": y_test.values, "y_score": y_pred_proba})
    test_results = test_results.sort_values("y_score", ascending=False)
    top_decile_n = max(1, int(len(test_results) * 0.1))
    top_decile = test_results.head(top_decile_n)
    capture_rate = top_decile["y_true"].mean()

    print(f"Top-decile capture rate: {capture_rate:.3f} (baseline rate: {y_test.mean():.3f})")

    return auc, precision, recall, f1

if __name__ == "__main__":
    df = load_data()
    train, test = temporal_split(df)

    X_train, y_train = prepare_xy(train)
    X_test, y_test = prepare_xy(test)

    print(f"\nDiscontinuation rate — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    model = train_xgboost(X_train, y_train)
    evaluate(model, X_test, y_test)

    import joblib
    import os
    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(model, "outputs/models/xgboost_persistence_model.pkl")
    print("\nModel saved to outputs/models/xgboost_persistence_model.pkl")