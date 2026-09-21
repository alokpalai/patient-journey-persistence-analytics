import pandas as pd
import os

WAREHOUSE_DIR = "data/warehouse"
RAW_DIR = "data/synthea_raw"


def load_base_data():
    timelines = pd.read_parquet(f"{WAREHOUSE_DIR}/patient_timelines.parquet")
    dim_patient = pd.read_parquet(f"{WAREHOUSE_DIR}/dim_patient.parquet")
    dim_patient["birth_date"] = pd.to_datetime(dim_patient["birth_date"])
    return timelines, dim_patient


def add_age_feature(timelines, dim_patient):
    df = timelines.merge(dim_patient, on="patient_key", how="left")

    first_therapy = pd.to_datetime(df["first_therapy_date"]).dt.tz_localize(None)
    birth = pd.to_datetime(df["birth_date"]).dt.tz_localize(None)

    df["age_at_first_therapy"] = ((first_therapy - birth).dt.days / 365.25).round(1)

    return df


def add_comorbidity_count(df):
    CHRONIC_COMORBIDITIES = [
        "Essential hypertension (disorder)",
        "Metabolic syndrome X (disorder)",
        "Hypertriglyceridemia (disorder)",
        "Hyperglycemia (disorder)",
        "Anemia (disorder)",
        "Ischemic heart disease (disorder)",
        "Chronic kidney disease stage 1 (disorder)",
        "Chronic kidney disease stage 2 (disorder)",
        "Chronic kidney disease stage 3 (disorder)",
        "Chronic kidney disease stage 4 (disorder)",
        "Body mass index 30+ - obesity (finding)",
        "Prediabetes (finding)",
        "Chronic pain (finding)",
        "Chronic low back pain (finding)",
    ]

    conditions = pd.read_csv(f"{RAW_DIR}/conditions.csv")

    merged = conditions.merge(
        df[["patient_key", "first_therapy_date"]],
        left_on="PATIENT",
        right_on="patient_key",
        how="inner",
    )

    merged["START"] = pd.to_datetime(merged["START"]).dt.tz_localize(None)
    merged["first_therapy_date"] = pd.to_datetime(merged["first_therapy_date"]).dt.tz_localize(None)

    merged = merged[merged["DESCRIPTION"].isin(CHRONIC_COMORBIDITIES)]

    prior_conditions = merged[merged["START"] < merged["first_therapy_date"]]

    comorbidity_count = (
        prior_conditions.groupby("patient_key")["DESCRIPTION"]
        .nunique()
        .reset_index()
        .rename(columns={"DESCRIPTION": "comorbidity_count"})
    )

    df = df.merge(comorbidity_count, on="patient_key", how="left")
    df["comorbidity_count"] = df["comorbidity_count"].fillna(0)

    return df


def add_prior_encounter_count(df):
    encounters = pd.read_csv(f"{RAW_DIR}/encounters.csv")

    merged = encounters.merge(
        df[["patient_key", "first_therapy_date"]],
        left_on="PATIENT",
        right_on="patient_key",
        how="inner",
    )

    merged["START"] = pd.to_datetime(merged["START"]).dt.tz_localize(None)
    merged["first_therapy_date"] = pd.to_datetime(merged["first_therapy_date"]).dt.tz_localize(None)

    prior_encounters = merged[merged["START"] < merged["first_therapy_date"]]

    encounter_count = (
        prior_encounters.groupby("patient_key")
        .size()
        .reset_index(name="prior_encounter_count")
    )

    df = df.merge(encounter_count, on="patient_key", how="left")
    df["prior_encounter_count"] = df["prior_encounter_count"].fillna(0)

    return df


if __name__ == "__main__":
    timelines, dim_patient = load_base_data()

    df = add_age_feature(timelines, dim_patient)
    df = add_comorbidity_count(df)
    df = add_prior_encounter_count(df)

    df.to_parquet(f"{WAREHOUSE_DIR}/model_features.parquet", index=False)

    print(f"model_features: {len(df)} rows written")
    print(df[["age_at_first_therapy", "comorbidity_count", "prior_encounter_count"]].describe())