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

def build_dim_drug():
    drug_map = {
        "24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet": "Metformin",
        "insulin isophane  human 70 UNT/ML / insulin  regular  human 30 UNT/ML Injectable Suspension [Humulin]": "Insulin",
        "insulin  regular  human 100 UNT/ML Injectable Solution": "Insulin",
        "Insulin Lispro 100 UNT/ML Injectable Solution [Humalog]": "Insulin",
        "3 ML liraglutide 6 MG/ML Pen Injector": "GLP-1 Agonist",
    }

    dim_drug = pd.DataFrame([
        {"drug_key": i + 1, "drug_description": desc, "drug_class": cls}
        for i, (desc, cls) in enumerate(drug_map.items())
    ])

    dim_drug.to_parquet(f"{WAREHOUSE_DIR}/dim_drug.parquet", index=False)
    print(f"dim_drug: {len(dim_drug)} rows written")
    return dim_drug


if __name__ == "__main__":
    print("Building dimension tables...")
    build_dim_patient()
    build_dim_drug()
    print("Done.")