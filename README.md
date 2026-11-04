# Apparent normalization of vital signs after emergency department care

Analysis code for the two-center cohort study *"Apparent normalization of vital signs and residual risk after emergency department care"* (MIMIC-IV-ED 2.2 + MIMIC-IV 3.1, and MC-MED 1.0.0).

**This repository contains code only.** It contains no patient-level data, no derived cohort files and no trained models. To reproduce the analyses you need your own credentialed access to the source databases (see [`DATA_ACCESS.md`](DATA_ACCESS.md)).

## What the code does

| Step | Script | Purpose | Manuscript |
|---|---|---|---|
| 2 | `false_stability_cohort.py` | MIMIC-IV-ED cohorts A (discharge) and B (ward), vital-sign sets, modified 5-component NEWS2, exposure groups | Methods; Figure 1 |
| 2b | `fase2b_ajustado.py` | First adjusted and sensitivity analyses (MIMIC) | — (superseded by 7a) |
| 3 | `fase3_lineas_base.py` | Baseline prediction models M0–M2 (logistic, gradient boosting) | Supplement S5 |
| 4 | `fase4_interpretabilidad.py` | Ablations and TreeSHAP (LightGBM) | Supplement S5 |
| 5 / 5b / 5c | `fase5_red_secuencia.py`, `fase5b_robustez.py`, `fase5c_solo_ritmo.py` | Sequence model M3h (GRU + FT-Transformer), robustness, timing-channel ablation (GPU) | Supplement S5–S6 |
| 5c* | `fase5c_independiente.py` | Stand-alone version of 3 + 5 + 5b + 5c for a fresh GPU session | Supplement S5–S6 |
| 6a | `fase6a_explorar_mcmed.py` | Aggregate exploration of MC-MED | — |
| 6b | `fase6b_mcmed_fenomeno.py` | MC-MED cohorts A and B* with the frozen MIMIC definitions | Methods; Figure 1 |
| 6c | `fase6c_transporte.py` | Transport of MIMIC-trained models to MC-MED (GPU) | Supplement S6 |
| 6d | `fase6d_local_mcmed.py` | Models retrained within MC-MED (GPU) | Supplement S6 |
| 7e | `fase7e_desenlace_A_corregido.py` | Final cohort-A outcome: **any** ED return ≤72 h ending in admission (keeps the earlier first-return label as `outcome_primera_reconsulta`) | Methods |
| 7a | `fase7a_epidemiologia.py` | Primary association analysis: logistic GLM with patient-clustered SE, models M1–M3, interaction, sensitivity analyses | Table 2, Figure 2 |
| 7b | `fase7b_news2_incremental.py` | Incremental value of prior NEWS2 maximum (N0–N3), patient-grouped 5-fold CV and temporal split | Table 3, Figure 3 |
| 7c-1 | `fase7c_sensibilidad_calibracion.py` | N4 sensitivity, temporal calibration, relative calibration-in-the-large, clustered bootstrap (1,000 reps), Table 1 | Tables 1 & 3, Figure 4 |
| 7c-2 | `fase7c_tablas_consolidadas.py` | Consolidated tables and supplementary tables, traceability file | Supplement |
| 7d | `fase7d_verificacion.py` | Verification analyses V1–V4 (any return, recorded triage, 5/5 sets, fixed 10-min windows) | Supplement S4 |
| 7g | `fase7g_figuras_drive.py` | Final English figures (TIFF 600 dpi) | Figures 1–4 |

Code comments and console output are in Spanish; variable names map to the manuscript as follows: *normalizacion aparente* = apparent normalization; *estable desde triage* = stable from triage; *anormal al final* = abnormal final set; *cohorte* = cohort; *desenlace* = outcome.

### Execution order and the outcome correction

The order in the notebook is the order in which the analyses were run. The exploratory machine-learning analyses (steps 3–6d) were run **before** step 7e and therefore use the first-return cohort-A label, as stated in the manuscript. Step 7e then rewrites cohort A with the final outcome, and steps 7a–7g produce every estimate in the main text. Step 7e refuses to run twice (it stops if its backup already exists). Cohorts B and B* are not affected by 7e.

### Frozen definitions

- Vital-sign set valid with ≥4 of 5 NEWS2 parameters (respiratory rate, SpO₂, systolic BP, heart rate, temperature); a missing parameter scores 0. No supplemental-oxygen or consciousness components.
- Abnormal set: modified NEWS2 ≥3 or any single parameter scoring 3. Normal final set: modified NEWS2 ≤1 with no parameter ≥2.
- MC-MED: nurse-charted records ≤10 min apart are grouped into one set; a missing parameter is carried forward ≤60 min.
- Definitions were fixed in MIMIC-IV-ED before any MC-MED analysis.
- Random seed 20261103; 1,000 patient-clustered bootstrap replicates.

## How to run

Open [`false_stability_pipeline.ipynb`](false_stability_pipeline.ipynb) in Google Colab and run the cells in order. It mounts Google Drive, takes the code from this repository, and reads the source data from your own Drive folder. Approximate run times: steps 2–2b ≈15 min (CPU); 3–4 ≈40 min (CPU); 5–5c ≈30 min (GPU); 6a–6b ≈25 min (CPU); 6c–6d ≈50 min (GPU); 7e–7c ≈95 min (CPU); 7d ≈20 min per center (high-RAM runtime recommended).

Locally: `pip install -r requirements.txt`, then set `FOLDER` (and `dest = FOLDER + "/mcmed"`) and run the scripts with `%run -i` in IPython in the same order. Steps 2 and 7a–7g run on CPU.

## Output and privacy

All scripts write their results to subfolders of `FOLDER` (`salida/`, `salida_mcmed/`, `salida_fase7/`). Console output and the CSV/figure files in `salida_fase7/` are aggregate. The cohort files (`cohorte_*.parquet`, `mcmed_*.csv.gz`) are **patient-level**. Under the PhysioNet data use agreements they must not be shared, uploaded to this repository, or pasted into online services. `.gitignore` excludes them.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite MIMIC-IV, MIMIC-IV-ED, MC-MED and PhysioNet as required by their data use agreements.

## License

Code: MIT (see [`LICENSE`](LICENSE)). The licence covers the code only; the data remain under their PhysioNet licences.
