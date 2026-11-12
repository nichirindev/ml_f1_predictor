# Pitwall · Pre-race F1 Podium Predictor

A local ML dashboard that estimates a driver's chance of a top-three finish **after
qualifying and before the race**. Three classifiers use the starting grid, previous
driver and team results, and prior circuit history. No current-race laps, finishing
status, fastest lap, points or finishing position enter the model.

The upgrade replaces the original retrospective model and its random split. Those
older scores measured a different task and are not comparable to the backtests below.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate            # Linux / macOS
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py      # http://localhost:8501
```

The bundled CSVs are sufficient; lap-time downloads and API credentials are not
required. scikit-learn 1.6+ is required for frozen-estimator calibration. Retrain
artifacts when upgrading ML libraries. Legacy joblib files cannot be used with the
new dashboard. Restart an already-running Streamlit server after this schema upgrade.

## Pre-race features

`features.py` is shared by training and inference. It processes races in date order,
reads the history snapshot **before** each event, then updates it after all teammates
have been scored. Even separate events on the same date cannot observe one another.

| Input | Definition |
|---|---|
| Starting position | Final grid slot; a recorded zero is represented as a non-grid/pit-lane flag and position 25 |
| Race round | Round number known before the race |
| Recent driver form | Mean classified finish and podium rate over the previous five starts |
| Driver history coverage | Number of prior starts in that window and experience capped at 100 starts |
| Recent team form | Mean finish and podium rate over the previous five Grands Prix, averaging teammates within each event |
| Team history coverage | Number of prior team races in the recent window |
| Driver circuit history | Mean finish, podium rate and count over the previous three visits |
| Team circuit history | Mean finish and podium rate over the previous three team visits |

There are 15 numeric features. Finish history uses `positionOrder`, including DNFs;
the target uses official finishing position 1–3. Entries are not discarded for
missing lap telemetry. No arbitrary driver, nationality or team label encoding is
used. Driver and constructor identity select the relevant historical form.

Cold starts use fixed priors: mean finish **15** and podium rate **0.10**, with zero
history counts. These are explicit assumptions, not statistics estimated from the
future. Rookies retain their available team history. Constructor names are mapped
explicitly (`Red Bull Racing` → `Red Bull`, `Kick Sauber` → `Sauber`, etc.). Different
constructor IDs are not automatically merged into an organizational lineage.
Roster-local synthetic driver IDs are ignored: names are normalized, including
`Hulkenberg` / `Hülkenberg`, to match real historical drivers.

## Chronological training and calibration

Training uses **2000–2024** results: 10,079 entries, including DNFs, rather than the
8,252 telemetry-complete entries used by the old model. The default backtest holds
out the latest three seasons:

| Test season | Base model training | Sigmoid calibration |
|---|---|---|
| 2022 | 2000–2020 | 2021 |
| 2023 | 2000–2021 | 2022 |
| 2024 | 2000–2022 | 2023 |

For each fold, base model fitting, calibration and testing use disjoint dates.
Calibration wraps an already fitted estimator in `FrozenEstimator`, following the
[scikit-learn calibration API](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).
No random cross-validation is used to fit the base models or choose seasons.
During a held-out season, form updates from earlier completed races, while base
model weights and calibration stay fixed. This is a sequential race forecast,
not a forecast of the whole season made before its first race.

The deployment bundle is separate: base models fit **2000–2023**, sigmoid calibration
uses **2024**, and the form snapshot ends **8 December 2024**. The base estimators
are not refitted on calibration labels afterward. Deployment forecasts must have a
race date after this cutoff; historical backtests use their own earlier models.

Fixed model configurations use seed 42. The test set is not used for hyperparameter
or threshold tuning. Evaluate alternative season sets explicitly with:

```bash
python train_models.py --test-years 2021 2022 2023 2024
```

## Backtest results

Initial upgraded run, pooled across 2022–2024: **1,359 test entries / 68 races**.
These numbers measure the pre-race task, not the older leakage-prone experiment.

| Model | Podium F1 | ROC-AUC | Brier ↓ | Top-three pick recall |
|---|---:|---:|---:|---:|
| Random Forest | 0.6562 | 0.9328 | 0.0713 | 0.6520 |
| Gradient Boosting | 0.6632 | 0.9358 | 0.0687 | 0.6520 |
| XGBoost | 0.6579 | 0.9314 | 0.0709 | 0.6471 |
| Model average | 0.6597 | 0.9358 | 0.0694 | 0.6520 |
| Grid positions 1–3 | 0.6486 | 0.7928 | 0.1052* | 0.6520 |
| Training-prevalence baseline | 0.0000 | 0.5003 | 0.1277 | — |

F1 uses a fixed **0.5** threshold. Brier measures probability error; lower is better.
Top-three pick recall measures how many actual podium finishers appear in each
race's three highest scores; ties use starting position, then normalized driver
name. The constant prior has no ranking information. *The grid baseline assigns
hard 0/1 scores, so it is a classification/ranking baseline rather than a calibrated
probability model. Results depend on the local library versions.

The gain over the grid baseline is modest. The models match its pooled top-three
pick recall in this run. Calibration is not guaranteed to improve every model;
reports show both raw and calibrated Brier error. The equal-weight model average
is evaluated as an ensemble but is not separately calibrated.

```bash
python compare_models.py            # Prints the locally generated backtest
python preprocess.py                # Refreshes the feature-importance PNG
python -m unittest discover -s tests -v
```

## Dashboard

- **Driver analysis:** select a 2025 driver and grid slot, compare model probabilities,
  and sweep all starting positions. Zero represents a pit-lane start in driver views.
- **Grid simulator:** edit every starting slot, rank contenders, and export scores
  with race context, training cutoff and model generation metadata. Invalid grids
  keep the edits and block grid scoring until corrected.
- **Head to head:** compare two drivers' podium probabilities at selected starting
  positions. This is not a model of which driver beats the other.
- **Model evidence:** pooled and season-by-season backtests, baselines, calibration
  error, selected driver's exact model inputs, and feature importance.

All views share the same model selector, circuit, date, round and score threshold.
The app reads the roster, history snapshot, models and report from one bundle, so
data files edited after training do not silently change the inference snapshot.
Retraining publishes the bundle atomically; its file timestamp invalidates the
Streamlit cache on the next rerun. Missing or old model artifacts show setup guidance.

## Data and generated artifacts

Historical CSVs come from the Ergast F1 dataset. `results.csv`, `races.csv`,
`drivers.csv` and `constructors.csv` form the training history; `circuits.csv`
provides display metadata. The bundled `f1_2025_grid.csv` is a static roster snapshot.
The separate `f1_2025_results.csv` contains only podium rows and is **excluded**:
training on it would omit all negative race entries. No 2025 or 2026 race results
are included in the deployed history.

Training generates these ignored, reproducible files:

- `models/prerace_bundle.joblib`: versioned models, rolling state, resolved roster,
  circuits, feature importances and matching report in one deployable artifact.
- `models/metrics.json`: backtest metrics, fold boundaries, deployment cutoff,
  feature schema, source hashes and training timestamp.
- `models/backtest_predictions.csv`: per-entry held-out predictions, including raw
  and calibrated scores, for independent inspection.

The original `build_ppt.py` is a legacy presentation generator describing the earlier
retrospective experiment; it is not a report of the upgraded pipeline.

## Tests and remaining limits

Tests cover current-outcome and future-outcome invariance, teammate and same-date
isolation, history window sizes, rookie defaults, training/inference parity,
chronological split boundaries, frozen base estimators during calibration, schema
rejection, score aggregation, grid editing and Streamlit controls. Dashboard tests
require trained artifacts; pure feature and training tests do not.

This remains a local prototype with stale data, a static 2025 roster, no weather,
qualifying times, tyres or current car updates. Forecast dates do not invent later
results: all deployment form stays at the saved cutoff until full new race results
are ingested and the pipeline reruns. Independent binary probabilities do not
sum to three or define a joint podium forecast. Calibration on one season does
not guarantee probability reliability in a different competitive era.
