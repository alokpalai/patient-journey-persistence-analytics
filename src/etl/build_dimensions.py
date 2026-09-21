import pandas as pd
import os

RAW_DIR = "data/synthea_raw"
WAREHOUSE_DIR = "data/warehouse"

os.makedirs(WAREHOUSE_DIR, exist_ok=True)


def build_dim_patient():
    patients = pd.read_csv(f"{RAW_DIR}/patients.csv")

    dim_patient = patients[[
        "Id", "BIRTHDATE", "DEATHDATE", "GENDER", "CITY", "STATE"
    ]].copy()

    dim_patient = dim_patient.rename(columns={
        "Id": "patient_key",
        "BIRTHDATE": "birth_date",
        "DEATHDATE": "death_date",
        "GENDER": "gender",
        "CITY": "city",
        "STATE": "state",
    })

    dim_patient.to_parquet(f"{WAREHOUSE_DIR}/dim_patient.parquet", index=False)
    print(f"dim_patient: {len(dim_patient)} rows written")
    return dim_patient


if __name__ == "__main__":
    print("Building dimension tables...")
    build_dim_patient()
    print("Done.")