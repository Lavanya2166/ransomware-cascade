# Static stage (v0)

First stage of the trust-calibrated ransomware cascade: a base classifier that outputs class probabilities, evaluated on ransomware families it has never seen. The trust layer (`trust/`) reads only the predictions files written here.

## Data

Sgandurra ransomware dataset (1,524 files: 942 benign, 582 ransomware; 30,967 binary behaviour features from sandbox runs; 11 ransomware families). Benign files have no family labels, so `groups.py` clusters them into 20 groups (KMeans on an SVD of the features) to allow family-wise splits.

## Models

- **Logistic regression (primary, `--model lr`)**: L2, C = 1, balanced class weights, after a variance filter. Chosen over Random Forest, Extra Trees and Bernoulli naive Bayes because it had the best family-wise out-of-fold scores (selection used dev data only, never the test families).
- **Random Forest (comparison, `--model rf`)**: 300 trees, kept as a baseline arm.

## Run order

Run from the repo root with the virtual environment active.

1. Put `RansomwareData.csv` and `VariableNames.txt` in `data/raw/`.
2. `python -m common.build_master` builds `data/master.parquet`.
3. `python -m static.prepare_X` builds the sparse feature matrix `data/X_sparse.npz`.
4. `python -m static.run_static --seed 0` runs the primary model for one seed. Repeat for seeds 1 to 4.
5. Optional comparison: `python -m static.run_static --seed 0 --model rf` (also seeds 1 to 4).
6. `python -m static.summarize` writes `results/static_runs.csv` and `results/static_summary.md`.

Do not use `--tune` for v0. Mutual-information feature selection over 30k features is too slow.

## What a run does

1. Splits families into dev (about 75%) and a held-out test set (about 25%), per seed.
2. Scores every dev file out of fold: a 5-fold family-wise CV, so each file is scored by a model that never saw its family. A second, random-split CV is saved only for comparison.
3. Trains on all of dev and predicts the held-out test families.
4. Saves probabilities and a summary, then validates the file with `common/validate.py`.

## Outputs

`artifacts/preds_static_seed{s}.parquet` (primary model) and `artifacts/preds_static_rf_seed{s}.parquet` (Random Forest), with columns `seed, cv_mode, role, sample_id, family, y, p_benign, p_ransom`.
- `cv_mode = group`, `role = oof`: family-wise out-of-fold scores (calibration pool for the trust layer).
- `cv_mode = random`, `role = oof`: random-split scores (ablation only).
- `cv_mode = group`, `role = test`: held-out test families.

`artifacts/static_summary_seed{s}.json` (and `static_summary_rf_seed{s}.json`) hold accuracy, recall, false-positive rate and AUC for each setting, plus latency.

## Files

| File | Purpose |
| --- | --- |
| `prepare_X.py` | Builds the sparse feature matrix aligned with `master.parquet` |
| `groups.py` | Replaces the benign family labels with balanced groups |
| `model.py` | Variance filter plus logistic regression or Random Forest |
| `run_static.py` | Splits, OOF scoring, final model, outputs |
| `summarize.py` | Mean and spread over seeds, per model |

## Results (5 seeds, mean +/- std)

Logistic regression (primary):

| Setting | Accuracy | Recall | False-positive rate | AUC |
| --- | --- | --- | --- | --- |
| Random-split CV (families seen) | 0.977 +/- 0.009 | 0.972 +/- 0.015 | 0.020 +/- 0.007 | 0.997 +/- 0.001 |
| Family-wise OOF (unseen families) | 0.849 +/- 0.064 | 0.942 +/- 0.051 | 0.213 +/- 0.071 | 0.928 +/- 0.051 |
| Held-out test families | 0.821 +/- 0.146 | 0.927 +/- 0.053 | 0.211 +/- 0.229 | 0.935 +/- 0.083 |

Random Forest (comparison):

| Setting | Accuracy | Recall | False-positive rate | AUC |
| --- | --- | --- | --- | --- |
| Random-split CV (families seen) | 0.970 +/- 0.007 | 0.971 +/- 0.012 | 0.032 +/- 0.011 | 0.996 +/- 0.002 |
| Family-wise OOF (unseen families) | 0.752 +/- 0.091 | 0.930 +/- 0.051 | 0.368 +/- 0.120 | 0.843 +/- 0.108 |
| Held-out test families | 0.787 +/- 0.167 | 0.951 +/- 0.029 | 0.268 +/- 0.245 | 0.862 +/- 0.160 |

## Caveats

- Features come from sandbox runs, so this is a behavioural baseline, not a static one.
- Benign families are similarity clusters, not real families.
- Only 11 ransomware families, so results vary a lot between seeds.
- Latency covers model inference only (about 2 ms for logistic regression, about 21.5 ms for Random Forest; mean of 5 seeds).

A plain-language explainer of this stage, including why we moved from Random Forest to logistic regression, is in `docs/static_v0/` (rebuild its figures with `python docs/static_v0/make_figures.py`).
