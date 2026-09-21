import pandas as pd
import os

WAREHOUSE_DIR = "data/warehouse"
DISCONTINUATION_GAP_DAYS = 90


def load_fact():
    fact = pd.read_parquet(f"{WAREHOUSE_DIR}/fact_medication_events.parquet")
    fact["start_date"] = pd.to_datetime(fact["start_date"])
    fact["stop_date"] = pd.to_datetime(fact["stop_date"])
    return fact

def build_patient_timelines(fact):
    fact = fact.sort_values(["patient_key", "start_date"]).copy()

    results = []

    for patient_key, group in fact.groupby("patient_key"):
        group = group.sort_values("start_date").reset_index(drop=True)

        first_therapy_date = group.loc[0, "start_date"]
        first_drug = group.loc[0, "drug_key"]

        # detect a switch: any later event using a different drug_key
        switched = (group["drug_key"] != first_drug).any()

        # detect discontinuation: gap > 90 days between consecutive events,
        # with nothing after the gap (i.e. the gap is at the END of the record)
        gaps = group["start_date"].diff().dt.days
        last_event_date = group["start_date"].iloc[-1]

        discontinued = False
        discontinuation_date = None

        for i in range(1, len(group)):
            if gaps.iloc[i] > DISCONTINUATION_GAP_DAYS:
                # check if this gap is the LAST gap in the sequence
                if i == len(group) - 1:
                    discontinued = True
                    discontinuation_date = group.loc[i - 1, "start_date"]

        results.append({
            "patient_key": patient_key,
            "first_therapy_date": first_therapy_date,
            "first_drug_key": first_drug,
            "switched_therapy": switched,
            "num_medication_events": len(group),
            "last_event_date": last_event_date,
            "discontinued": discontinued,
            "discontinuation_date": discontinuation_date,
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    fact = load_fact()
    timelines = build_patient_timelines(fact)
    timelines.to_parquet(f"{WAREHOUSE_DIR}/patient_timelines.parquet", index=False)
    print(f"patient_timelines: {len(timelines)} rows written")
    print(f"discontinued: {timelines['discontinued'].sum()}")
    print(f"switched therapy: {timelines['switched_therapy'].sum()}")