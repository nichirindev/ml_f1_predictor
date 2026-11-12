"""Train pre-race podium models with chronological calibration and season backtests.

Usage: python train_models.py [--test-years 2022 2023 2024]
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

from features import FEATURES, SCHEMA_VERSION, START_YEAR, build_training_table, load_history, prepare_roster
from prediction import MODEL_LABELS

ROOT = Path(__file__).parent
DATA_DIR, MODEL_DIR = ROOT / "data", ROOT / "models"


def get_models() -> dict:
    """Fixed hyperparameters; no tuning or class reweighting on held-out seasons."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "gradient_boost": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, min_samples_leaf=10, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=5,
            random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=4),
    }


def temporal_split(table: pd.DataFrame, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve the preceding season exclusively for calibration."""
    train = table.loc[table.year < test_year - 1]
    calibration = table.loc[table.year == test_year - 1]
    test = table.loc[table.year == test_year]
    if any(frame.empty or frame.top3.nunique() < 2 for frame in (train, calibration, test)):
        raise ValueError(f"{test_year}: need earlier training seasons, a calibration season, and a test season with both classes.")
    if not train.date.max() < calibration.date.min() <= calibration.date.max() < test.date.min():
        raise ValueError("Training, calibration and test dates must be strictly separated.")
    return train, calibration, test


def fit_calibrated(train: pd.DataFrame, calibration: pd.DataFrame) -> tuple[dict, dict]:
    """Freeze each base estimator before fitting sigmoid calibration on later races."""
    fitted, importance = {}, {}
    for name, estimator in get_models().items():
        estimator.fit(train[FEATURES], train.top3)
        importance[MODEL_LABELS[name]] = estimator.feature_importances_.tolist()
        calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
        calibrated.fit(calibration[FEATURES], calibration.top3)
        fitted[name] = calibrated
    return fitted, importance


def evaluate(entries: pd.DataFrame, probabilities: np.ndarray) -> dict:
    """Report probability quality, fixed-threshold classification and top-three picks."""
    y = entries.top3.to_numpy()
    predictions = probabilities >= 0.5
    ranked = entries[["raceId", "driver_key", "grid", "top3"]].assign(probability=probabilities)
    picked = ranked.sort_values(["probability", "grid", "driver_key"], ascending=[False, True, True]).groupby("raceId").head(3)
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities),
        "brier": brier_score_loss(y, probabilities),
        "log_loss": log_loss(y, probabilities, labels=[0, 1]),
        "top3_recall": float(picked.top3.sum() / y.sum()),
    }


def run_backtests(table: pd.DataFrame, years: list[int]) -> tuple[dict, pd.DataFrame]:
    """Expanding season backtests, updating form only after preceding race results."""
    folds, predictions = [], []
    for year in years:
        train, calibration, test = temporal_split(table, year)
        print(f"Backtest {year}: fit {int(train.year.min())}–{int(train.year.max())}, calibrate {year - 1}, test {len(test)} entries", flush=True)
        start = time.monotonic()
        models, _ = fit_calibrated(train, calibration)
        output = test[["raceId", "date", "year", "driver_key", "constructorId", "grid", "top3"]].copy()
        for name, model in models.items():
            output[name] = model.predict_proba(test[FEATURES])[:, 1]
            output[f"{name}_raw"] = model.calibrated_classifiers_[0].estimator.predict_proba(test[FEATURES])[:, 1]
        output["model_average"] = output[list(MODEL_LABELS)].mean(axis=1)
        output["prior_rate"] = float(train.top3.mean())
        output["grid_top3"] = (test.grid.between(1, 3)).astype(float)
        metrics = {name: evaluate(output, output[name].to_numpy())
                   for name in [*MODEL_LABELS, "model_average", "prior_rate", "grid_top3"]}
        for name in MODEL_LABELS:
            metrics[name]["raw_brier"] = brier_score_loss(test.top3, output[f"{name}_raw"])
        folds.append({"test_year": year, "train_through": int(train.year.max()),
                      "calibration_year": year - 1, "train_entries": len(train),
                      "calibration_entries": len(calibration), "test_entries": len(test),
                      "test_races": int(test.raceId.nunique()), "models": metrics})
        predictions.append(output)
        print(f"  Model average: F1={metrics['model_average']['f1']:.3f}, Brier={metrics['model_average']['brier']:.3f}, top-3 recall={metrics['model_average']['top3_recall']:.3f} ({time.monotonic() - start:.1f}s)", flush=True)
    combined = pd.concat(predictions, ignore_index=True)
    summary = {name: evaluate(combined, combined[name].to_numpy())
               for name in [*MODEL_LABELS, "model_average", "prior_rate", "grid_top3"]}
    for name in MODEL_LABELS:
        summary[name]["raw_brier"] = brier_score_loss(combined.top3, combined[f"{name}_raw"])
    # A constant prior has no ranking information; do not credit the grid tie-break.
    summary["prior_rate"]["top3_recall"] = None
    for fold in folds:
        fold["models"]["prior_rate"]["top3_recall"] = None
    return {"folds": folds, "models": summary, "test_entries": len(combined),
            "test_races": int(combined.raceId.nunique())}, combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-years", type=int, nargs="+", help="Distinct held-out seasons; default: latest three seasons")
    args = parser.parse_args()
    history = load_history(DATA_DIR)
    table, state = build_training_table(history)
    latest_year = int(table.year.max())
    years = sorted(set(args.test_years or range(latest_year - 2, latest_year + 1)))
    report, predictions = run_backtests(table, years)

    # Deployment uses the latest season for calibration, without refitting the base on it.
    deployment_train = table.loc[table.year < latest_year]
    deployment_calibration = table.loc[table.year == latest_year]
    print(f"Deployment: fit through {latest_year - 1}, calibrate on {latest_year}", flush=True)
    models, importance = fit_calibrated(deployment_train, deployment_calibration)
    cutoff = history.date.max().date().isoformat()
    report.update({
        "schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES, "samples": len(table), "history_start_year": START_YEAR,
        "trained_through": cutoff, "backtest_years": years,
        "protocol": "Expanding seasons; preceding season reserved for sigmoid calibration; form updates after each earlier race; threshold 0.5.",
        "deployment": {"base_train_through": latest_year - 1, "calibration_year": latest_year,
                       "train_entries": len(deployment_train), "calibration_entries": len(deployment_calibration)},
        "sources_sha256": {filename: hashlib.sha256((DATA_DIR / filename).read_bytes()).hexdigest()
                           for filename in ["results.csv", "races.csv", "drivers.csv", "constructors.csv", "circuits.csv", "f1_2025_grid.csv"]},
        "sklearn_version": sklearn.__version__,
    })
    circuits = pd.read_csv(DATA_DIR / "circuits.csv")
    circuits = circuits.loc[circuits.circuitId.isin(history.circuitId.unique()), ["circuitId", "name", "country"]]
    bundle = {"schema_version": SCHEMA_VERSION, "features": FEATURES, "trained_through": cutoff,
              "models": models, "state": state, "roster": prepare_roster(DATA_DIR),
              "circuits": circuits, "metrics": report, "feature_importance": importance}
    MODEL_DIR.mkdir(exist_ok=True)
    # Publish one self-contained bundle atomically so live sessions cannot mix generations.
    with tempfile.TemporaryDirectory(dir=MODEL_DIR) as temporary:
        artifact = Path(temporary) / "prerace_bundle.joblib"
        joblib.dump(bundle, artifact)
        artifact.replace(MODEL_DIR / "prerace_bundle.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    predictions.to_csv(MODEL_DIR / "backtest_predictions.csv", index=False)
    print(pd.DataFrame(report["models"]).T[["f1", "roc_auc", "brier", "top3_recall"]].round(4).to_string())
    print(f"Saved pre-race model bundle and backtest report to {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    main()
