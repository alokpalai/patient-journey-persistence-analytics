import pandas as pd
import os

RAW_DIR = "data/synthea_raw"
WAREHOUSE_DIR = "data/warehouse"

os.makedirs(WAREHOUSE_DIR, exist_ok=True)


def build_cohort():
    conditions = pd.read_csv(f"{RAW_DIR}/conditions.csv")

    t2d = conditions[
        conditions["DESCRIPTION"] == "Diabetes mellitus type 2 (disorder)"
    ].copy()

    cohort = t2d[["PATIENT", "START"]].rename(columns={
        "PATIENT": "patient_key",
        "START": "diagnosis_date",
    })

    cohort = cohort.sort_values("diagnosis_date").drop_duplicates(
        subset="patient_key", keep="first"
    )

    cohort.to_parquet(f"{WAREHOUSE_DIR}/cohort_t2d_patients.parquet", index=False)
    print(f"cohort_t2d_patients: {len(cohort)} rows written")
    return cohort

def build_fact_medication_events(cohort):
    medications = pd.read_csv(f"{RAW_DIR}/medications.csv")
    dim_drug = pd.read_parquet(f"{WAREHOUSE_DIR}/dim_drug.parquet")

    diabetes_drug_names = dim_drug["drug_description"].tolist()

    fact = medications[medications["DESCRIPTION"].isin(diabetes_drug_names)].copy()

    fact = fact.merge(
        cohort[["patient_key"]],
        left_on="PATIENT",
        right_on="patient_key",
        how="inner",
    )

    fact = fact.merge(
        dim_drug[["drug_key", "drug_description"]],
        left_on="DESCRIPTION",
        right_on="drug_description",
        how="left",
    )

    fact = fact[[
        "patient_key", "drug_key", "START", "STOP", "DESCRIPTION"
    ]].rename(columns={
        "START": "start_date",
        "STOP": "stop_date",
        "DESCRIPTION": "drug_description",
    })

    fact.to_parquet(f"{WAREHOUSE_DIR}/fact_medication_events.parquet", index=False)
    print(f"fact_medication_events: {len(fact)} rows written")
    print(f"unique patients in fact table: {fact['patient_key'].nunique()}")
    return fact


if __name__ == "__main__":
    print("Building cohort...")
    cohort = build_cohort()
    print("Building fact table...")
    build_fact_medication_events(cohort)
    print("Done.")