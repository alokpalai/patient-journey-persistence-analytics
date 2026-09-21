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


if __name__ == "__main__":
    print("Building cohort...")
    build_cohort()
    print("Done.")