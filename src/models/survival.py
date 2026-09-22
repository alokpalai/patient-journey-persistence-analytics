import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt

WAREHOUSE_DIR = "data/warehouse"
OUTPUT_DIR = "outputs/figures"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_survival_data():
    df = pd.read_parquet(f"{WAREHOUSE_DIR}/model_features.parquet")

    df["first_therapy_date"] = pd.to_datetime(df["first_therapy_date"]).dt.tz_localize(None)
    df["last_event_date"] = pd.to_datetime(df["last_event_date"]).dt.tz_localize(None)

    # duration = how long the patient was observed, in days, from first therapy
    df["duration_days"] = (df["last_event_date"] - df["first_therapy_date"]).dt.days

    # event = 1 if they discontinued (the event we're modeling time-to),
    # 0 if they were still on therapy at their last observed record (censored)
    df["event_observed"] = df["discontinued"].astype(int)

    # drop any zero or negative durations — same-day only records give no usable duration
    df = df[df["duration_days"] > 0].copy()

    return df


def plot_overall_km(df):
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["duration_days"], event_observed=df["event_observed"])

    kmf.plot_survival_function()
    plt.title("Overall Therapy Persistence — Kaplan-Meier Curve")
    plt.xlabel("Days since first therapy")
    plt.ylabel("Proportion still on therapy")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/km_overall.png", dpi=150)
    plt.close()

    print(f"Median survival time: {kmf.median_survival_time_} days")


def plot_km_by_comorbidity(df):
    df = df.copy()
    df["comorbidity_group"] = pd.cut(
        df["comorbidity_count"],
        bins=[-1, 3, 7, 100],
        labels=["Low (0-3)", "Medium (4-7)", "High (8+)"],
    )

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8, 6))

    for group in df["comorbidity_group"].cat.categories:
        subset = df[df["comorbidity_group"] == group]
        kmf.fit(subset["duration_days"], subset["event_observed"], label=group)
        kmf.plot_survival_function(ax=ax)

    plt.title("Therapy Persistence by Comorbidity Burden")
    plt.xlabel("Days since first therapy")
    plt.ylabel("Proportion still on therapy")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/km_by_comorbidity.png", dpi=150)
    plt.close()

    print("Median survival time by comorbidity group:")
    for group in df["comorbidity_group"].cat.categories:
        subset = df[df["comorbidity_group"] == group]
        kmf.fit(subset["duration_days"], subset["event_observed"], label=group)
        print(f"  {group}: {kmf.median_survival_time_} days")

def fit_cox_model(df):
    from sklearn.preprocessing import StandardScaler

    cox_df = df[[
        "duration_days", "event_observed",
        "age_at_first_therapy", "comorbidity_count",
        "prior_encounter_count", "switched_therapy",
    ]].copy()

    cox_df["switched_therapy"] = cox_df["switched_therapy"].astype(int)

    continuous_cols = ["age_at_first_therapy", "comorbidity_count", "prior_encounter_count"]
    scaler = StandardScaler()
    cox_df[continuous_cols] = scaler.fit_transform(cox_df[continuous_cols])

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="duration_days", event_col="event_observed")

    print("\nCox Proportional Hazards summary (standardized features):")
    print(cph.summary[["coef", "exp(coef)", "p"]])

    return cph


if __name__ == "__main__":
    df = load_survival_data()
    print(f"Patients in survival analysis: {len(df)}")

    plot_overall_km(df)
    plot_km_by_comorbidity(df)
    fit_cox_model(df)