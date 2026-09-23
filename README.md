# Patient Journey Analytics & Treatment Persistence Prediction

Predicting Type 2 Diabetes medication discontinuation using synthetic patient data, survival analysis, and explainable machine learning.

## Problem

Pharmaceutical and healthcare analytics teams need to know which patients are at risk of stopping their prescribed therapy, so they can intervene before it happens. This project builds an end-to-end pipeline that:

- Reconstructs each patient's treatment journey from diagnosis to first therapy to discontinuation or ongoing treatment
- Predicts 12-month discontinuation risk using XGBoost, validated with a temporal (not random) train/test split
- Identifies the real drivers of discontinuation using Kaplan-Meier survival curves, Cox proportional hazards, and SHAP explainability
- Segments patients into risk archetypes with a distinct intervention strategy for each

## Data

Data is synthetic, generated with [Synthea](https://github.com/synthetichealth/synthea), MITRE's open-source synthetic patient generator. Synthetic data was used because real patient-level medical records are protected health information and not available for a personal portfolio project; Synthea produces realistic, clinically plausible longitudinal patient histories without any privacy concern.

**Generation command:**
```
./run_synthea.bat -p 50000 -s 20260913 Massachusetts
```

This produced 57,641 total patient records (50,000 living, 7,641 marked deceased - Synthea models mortality across a full simulated lifespan, not just a point-in-time snapshot). Raw CSVs are not committed to this repository (see `data/README.md` for reproduction instructions); only the processed, de-identified warehouse tables and code are included.

## Cohort definition

The analysis cohort is built in stages:

1. **4,419 patients** carry a `Diabetes mellitus type 2 (disorder)` diagnosis. Diabetes complications (nephropathy, retinopathy, neuropathy) and prediabetes were deliberately excluded from the cohort definition - they are downstream consequences or a distinct earlier condition, not the diagnosis itself.
2. Of those, **3,180 patients (72%)** have at least one recorded diabetes medication event (metformin, insulin, or a GLP-1 agonist). The remaining 28% are diagnosed but have no medication on record in the dataset - plausibly diet/lifestyle-managed, or a data generation artifact. Only the 3,180 medicated patients are used for persistence modeling, since discontinuation is undefined for a patient who was never on therapy.

## Methodology

### Discontinuation definition
A patient is flagged as discontinued if the gap between their last two medication events exceeds **90 days**, with no subsequent fill. This threshold is the standard convention in pharmacy persistence literature. It is a simplified heuristic - a more rigorous version would account for each prescription's expected days-of-supply rather than just the gap between start dates. Stated here as a known limitation.

### Comorbidity counting
An early version of this pipeline counted every diagnosis code as a "comorbidity," which produced an implausible average of ~20 comorbidities per patient. Inspection showed this was dominated by administrative entries (e.g. "Medication review due") and social-determinants-of-health findings (e.g. "Limited social contact," "Full-time employment") that Synthea logs alongside genuine clinical diagnoses. The final pipeline uses a fixed inclusion list of 14 clinically relevant chronic conditions (hypertension, obesity, chronic kidney disease stages, ischemic heart disease, etc.) - the same principle used by standard clinical comorbidity indices. This brought the average down to a clinically plausible 6.5 comorbidities per patient.

### Leakage discipline
Every feature (age, comorbidity count, prior encounter count) is computed using only data available **before** the patient's first therapy date. This was enforced explicitly in code, and a timezone bug (medication dates were timezone-aware, condition/encounter dates were not) had to be fixed to make the date comparisons work correctly at all.

### Temporal validation
The train/test split is chronological, not random: the earliest 80% of patients (by first therapy date) form the training set, the most recent 20% form the test set. This mirrors real deployment - a model trained on historical patients predicting for new patients going forward - and is a harder, more honest test than a random split would be.

## Results

### Persistence prediction (XGBoost)
| Metric | Value |
|---|---|
| AUC | 0.702 |
| Precision | 0.671 |
| Recall | 0.667 |
| F1 | 0.669 |
| Top-decile capture rate | 68.3% (vs. 50.0% baseline) |

The top-decile capture rate is the most business-relevant number: if an intervention program targeted the 10% of patients the model scores as highest risk, 68% of them would actually go on to discontinue therapy, versus a 50% baseline if patients were targeted at random.

### Survival analysis (Kaplan-Meier / Cox Proportional Hazards)
Overall median time to discontinuation: **3,710 days**. Cox regression (standardized features) found four statistically significant predictors (all p < 0.001):

| Feature | Hazard Ratio | Effect (per 1 SD) |
|---|---|---|
| Switched therapy | 0.310 | 69% lower hazard of discontinuation |
| Comorbidity count | 0.644 | 36% lower hazard |
| Age at first therapy | 0.854 | 15% lower hazard |
| Prior encounter count | 1.200 | 20% higher hazard |

### The counterintuitive finding
The strongest predictor of a patient **staying** on therapy is whether their treatment was ever switched (e.g. metformin to insulin) - not whether they were younger, healthier, or had fewer comorbidities. Patients who switched therapy had a 69% lower hazard of discontinuation. The most plausible explanation: a therapy switch signals active clinical management - a provider adjusting treatment is evidence the patient is engaged with the healthcare system, not evidence the patient is failing therapy. Quiet, unadjusted stability may actually precede disengagement. This finding was independently confirmed by SHAP feature importance on the XGBoost model, where `switched_therapy` was also the single most important feature (mean absolute SHAP value of 0.91, more than double the next-highest feature).

## Patient segments

KMeans clustering on the same feature set produced four risk archetypes:

| Segment | Patients | Discontinuation Rate | Defining characteristic |
|---|---|---|---|
| Disengaged Younger Patients | 1,242 | 69% | Youngest average age, almost never switched therapy |
| Older, Stable-but-Unmanaged | 1,087 | 50% | Highest comorbidity burden, but zero therapy switches |
| High-Utilizer Outliers | 13 | 31% | Extreme prior encounter count (avg. 226); too small a group to generalize from |
| Actively Managed Patients | 838 | 15% | 100% had their therapy switched at least once |

Suggested intervention per segment is documented in `reports/insight_memo.md`.

## Dashboard

An interactive Power BI dashboard (`dashboards/patient_persistence_dashboard.pbix`, PDF export also included) presents the cohort overview, segment comparison, and driver analysis for a non-technical audience.

## Limitations

- Data is synthetic. Findings are internally consistent and methodologically sound, but should not be read as real-world clinical fact.
- Synthea's medication model is simpler than real-world T2D pharmacology (no sulfonylureas, DPP-4 or SGLT2 inhibitors were generated in this run).
- The discontinuation definition (90-day gap) does not account for expected days-of-supply per prescription.
- The comorbidity inclusion list was built by manual inspection of the most frequent diagnosis codes in this dataset and is not exhaustive.
- The "High-Utilizer Outliers" segment (13 patients) is too small to generalize from and is reported as a distinct edge case rather than folded into broader conclusions.
- Synthea generates patient timelines into simulated future dates; this does not affect model validity, since the temporal split is purely relative within the dataset.

## Repository structure

```
data/
  synthea_raw/      - not committed, see data/README.md
  warehouse/        - not committed, parquet tables, regenerable
src/
  etl/              - dimension and fact table builders
  journey/          - therapy sequencing and discontinuation logic
  models/           - feature engineering, model training, survival analysis, SHAP, segmentation
dashboards/         - Power BI dashboard (.pbix and .pdf)
outputs/figures/    - exported chart images
reports/            - insight memo
```

## Reproducing this project

1. Generate synthetic data with Synthea (see `data/README.md` for the exact command)
2. `pip install -r requirements.txt`
3. Run scripts in order:
```
python src/etl/build_dimensions.py
python src/etl/build_fact_medication_events.py
python src/journey/discontinuation.py
python src/models/build_features.py
python src/models/train_model.py
python src/models/survival.py
python src/models/explain_shap.py
python src/models/segmentation.py
python src/models/export_for_dashboard.py
```