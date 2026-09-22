import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

WAREHOUSE_DIR = "data/warehouse"
OUTPUT_DIR = "outputs/figures"

FEATURE_COLS = ["age_at_first_therapy", "comorbidity_count", "prior_encounter_count", "switched_therapy"]


def load_data():
    df = pd.read_parquet(f"{WAREHOUSE_DIR}/model_features.parquet")
    return df


def run_kmeans(df, n_clusters=4):
    X = df[FEATURE_COLS].copy()
    X["switched_therapy"] = X["switched_therapy"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["segment"] = kmeans.fit_predict(X_scaled)

    return df, kmeans


def profile_segments(df):
    profile = df.groupby("segment").agg(
        patient_count=("patient_key", "count"),
        avg_age=("age_at_first_therapy", "mean"),
        avg_comorbidity=("comorbidity_count", "mean"),
        avg_prior_encounters=("prior_encounter_count", "mean"),
        pct_switched=("switched_therapy", "mean"),
        discontinuation_rate=("discontinued", "mean"),
    ).round(2)

    profile = profile.sort_values("discontinuation_rate", ascending=False)

    print("\nSegment profiles:")
    print(profile)

    return profile


if __name__ == "__main__":
    df = load_data()
    df, kmeans = run_kmeans(df, n_clusters=4)
    profile = profile_segments(df)

    df.to_parquet(f"{WAREHOUSE_DIR}/patient_segments.parquet", index=False)
    print(f"\npatient_segments: {len(df)} rows written")