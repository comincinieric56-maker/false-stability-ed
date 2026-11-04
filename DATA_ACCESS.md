# Data access

The code needs three PhysioNet datasets. Each requires a PhysioNet credentialed account, completion of the required human-subjects research training (CITI "Data or Specimens Only Research"), and signing the dataset's data use agreement. Every person who runs the code needs their own access; files must not be passed between people.

| Dataset | Version | Link |
|---|---|---|
| MIMIC-IV-ED | 2.2 | https://physionet.org/content/mimic-iv-ed/2.2/ |
| MIMIC-IV | 3.1 | https://physionet.org/content/mimiciv/3.1/ |
| MC-MED | 1.0.0 | https://physionet.org/content/mc-med/1.0.0/ |

## Expected folder layout

Place the downloaded files in one Google Drive folder (the notebook variable `FOLDER`):

```
FOLDER/
├── mimic-iv-ed-2.2.zip          # full MIMIC-IV-ED 2.2 download (contains ed/edstays.csv.gz, ed/vitalsign.csv.gz, ed/triage.csv.gz, ...)
├── patients.csv.gz              # MIMIC-IV 3.1 hosp/
├── admissions.csv.gz            # MIMIC-IV 3.1 hosp/
├── transfers.csv.gz             # MIMIC-IV 3.1 hosp/
├── icustays.csv.gz              # MIMIC-IV 3.1 icu/
└── mcmed/
    ├── visits.csv               # MC-MED 1.0.0
    ├── numerics.csv
    └── orders.csv
```

The scripts create `salida/`, `salida_mcmed/` and `salida_fase7/` inside `FOLDER`. Those folders contain patient-level derived files. Keep them in your credentialed storage only.

## What may be shared

Only aggregate results (counts, estimates, figures, the CSV tables in `salida_fase7/` and the console output) may be shared. Row-level data, the derived cohort files and trained model weights must not be shared or uploaded to online services, including AI assistants.
