import pandas as pd

WAREHOUSE_DIR = "data/warehouse"
DASHBOARD_DIR = "dashboards"

import os
os.makedirs(DASHBOARD_DIR, exist_ok=True)


def export_dashboard_data():
    segments = pd.read_parquet(f"{WAREHOUSE_DIR}/patient_segments.parquet")

    # keep only what the dashboard needs — drop internal keys not useful for viz
    dashboard_df = segments[[
        "patient_key", "age_at_first_therapy", "comorbidity_count",
        "prior_encounter_count", "switched_therapy", "discontinued",
        "segment", "first_therapy_date",
    ]].copy()

    # human-readable segment names, matching what we defined together
    segment_names = {
        2: "Disengaged Younger Patients",
        1: "Older, Stable-but-Unmanaged",
        3: "High-Utilizer Outliers",
        0: "Actively Managed Patients",
    }
    dashboard_df["segment_name"] = dashboard_df["segment"].map(segment_names)

    dashboard_df["switched_therapy"] = dashboard_df["switched_therapy"].map({True: "Yes", False: "No"})
    dashboard_df["discontinued"] = dashboard_df["discontinued"].map({True: "Discontinued", False: "Active/Censored"})

    dashboard_df.to_csv(f"{DASHBOARD_DIR}/patient_segments_export.csv", index=False)
    print(f"Exported {len(dashboard_df)} rows to dashboards/patient_segments_export.csv")


if __name__ == "__main__":
    export_dashboard_data()