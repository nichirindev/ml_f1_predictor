"""Shared inference helpers for the scenario dashboard."""
import numpy as np
import pandas as pd

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "gradient_boost": "Gradient Boosting",
    "xgboost": "XGBoost",
}
TEAM_CANDIDATES = {
    "Red Bull Racing": ["Red Bull"],
    "McLaren": ["McLaren", "McLaren-Mercedes"],
    "Alpine": ["Alpine F1 Team", "Alpine"],
    "Kick Sauber": ["Sauber", "Alfa Romeo", "Alfa Romeo Racing"],
    "RB": ["RB F1 Team", "AlphaTauri", "Toro Rosso"],
}


def encode_label(encoder, value, candidates=()):
    for candidate in [value, *candidates]:
        if candidate in encoder.classes_:
            return int(encoder.transform([candidate])[0]), candidate
    raise ValueError(f"'{value}' is absent from the training labels. Retrain with this label before predicting.")


def build_features(drivers, prep, assumptions):
    """Build rows in the artifact's feature order; report historical team aliases."""
    rows, aliases = [], set()
    for driver in drivers.to_dict("records"):
        nationality, _ = encode_label(prep["le_nat"], driver["nationality"])
        constructor, used = encode_label(
            prep["le_con"], driver["team_name"], TEAM_CANDIDATES.get(driver["team_name"], ())
        )
        if used != driver["team_name"]:
            aliases.add(f"{driver['team_name']} → {used}")
        rows.append({**assumptions, "grid": int(driver["Grid"]),
                     "nationality_encoded": nationality, "constructor_encoded": constructor})
    return pd.DataFrame(rows, columns=prep["features"]), sorted(aliases)


def predict_models(models, features):
    return pd.DataFrame({MODEL_LABELS[key]: model.predict_proba(features)[:, 1]
                         for key, model in models.items()})


def select_scores(probabilities, selection):
    return probabilities.mean(axis=1) if selection == "Model average" else probabilities[selection]


def validate_grid(grid):
    positions = pd.to_numeric(grid["Grid"], errors="coerce")
    if (positions.isna().any() or not np.isfinite(positions).all()
            or (positions % 1 != 0).any()
            or sorted(positions.tolist()) != list(range(1, len(grid) + 1))):
        raise ValueError(f"Assign each starting position from 1 to {len(grid)} exactly once.")
