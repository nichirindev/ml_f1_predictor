"""Shared pre-race inference and validation for the Pitwall dashboard."""
import numpy as np
import pandas as pd

from features import FEATURES, SCHEMA_VERSION

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "gradient_boost": "Gradient Boosting",
    "xgboost": "XGBoost",
}


def validate_bundle(bundle: dict) -> None:
    """Reject retrospective or incompatible artifacts instead of mixing schemas."""
    if bundle.get("schema_version") != SCHEMA_VERSION or bundle.get("features") != FEATURES:
        raise ValueError("Incompatible model bundle. Run python train_models.py to train the pre-race models.")
    if set(bundle["models"]) != set(MODEL_LABELS):
        raise ValueError("Incomplete pre-race model bundle. Retrain all three models together.")


def build_features(drivers: pd.DataFrame, prep: dict, assumptions: dict) -> tuple[pd.DataFrame, list[str]]:
    """Use the training snapshot and the exact same feature function as backtesting."""
    cutoff = pd.Timestamp(assumptions["race_date"])
    if pd.isna(cutoff) or cutoff <= pd.Timestamp(prep["trained_through"]):
        raise ValueError("Choose a race date after the model's training cutoff; historical forecasts require a backtest model.")
    round_number, circuit = assumptions["round"], assumptions["circuitId"]
    if not np.isfinite(round_number) or round_number != int(round_number) or not 1 <= round_number <= 30:
        raise ValueError("Race round must be between 1 and 30.")
    if circuit not in prep["circuits"].circuitId.values:
        raise ValueError("Choose a circuit represented in the historical race data.")
    state = prep["state"]
    rows, notes = [], set()
    for driver in drivers.to_dict("records"):
        grid = driver["Grid"]
        if not np.isfinite(grid) or grid != int(grid) or not 0 <= grid <= 24:
            raise ValueError("Starting positions must be whole numbers from 0 (pit lane) to 24.")
        entry = {**driver, "grid": int(grid), "round": round_number, "circuitId": circuit}
        row = state.features(entry)
        rows.append(row)
        if row["driver_history_races"] == 0:
            notes.add(f"{driver['driver_name']}: no recorded F1 starts; using rookie priors and team history.")
        if row["driver_circuit_races"] == 0:
            notes.add(f"{driver['driver_name']}: no recorded starts at this circuit; using circuit priors.")
        if driver["team_name"] != driver["historical_team"]:
            notes.add(f"Team mapping: {driver['team_name']} → {driver['historical_team']}.")
    return pd.DataFrame(rows, columns=FEATURES), sorted(notes)


def predict_models(models: dict, features: pd.DataFrame) -> pd.DataFrame:
    """Return calibrated probabilities from each pre-race classifier."""
    return pd.DataFrame({MODEL_LABELS[key]: model.predict_proba(features)[:, 1]
                         for key, model in models.items()})


def select_scores(probabilities: pd.DataFrame, selection: str) -> pd.Series:
    """Use one scoring method consistently across all dashboard views."""
    return probabilities.mean(axis=1) if selection == "Model average" else probabilities[selection]


def validate_grid(grid: pd.DataFrame) -> None:
    """Require a complete grid permutation; driver comparisons may use pit-lane starts."""
    positions = pd.to_numeric(grid["Grid"], errors="coerce")
    if (positions.isna().any() or not np.isfinite(positions).all()
            or (positions % 1 != 0).any()
            or sorted(positions.tolist()) != list(range(1, len(grid) + 1))):
        raise ValueError(f"Assign each starting position from 1 to {len(grid)} exactly once.")
