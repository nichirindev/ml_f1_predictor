"""
F1 Podium Predictor - Streamlit front end.
Loads the models trained by train_models.py and lets you:
  1. Predict a single driver's podium probability (all 3 models)
  2. Simulate a full race grid and rank every driver's podium chance

Usage: .venv/Scripts/python.exe -m streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏁", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_all():
    models = {}
    for name in ["random_forest", "gradient_boost", "xgboost"]:
        models[name] = joblib.load(MODELS / f"f1_model_{name}.joblib")
    prep = joblib.load(MODELS / "preprocessors.joblib")
    features = prep["features"]
    le_nat, le_con = prep["le_nat"], prep["le_con"]

    grid = pd.read_csv(DATA / "f1_2025_grid.csv")
    races = pd.read_csv(DATA / "races.csv", usecols=["year"])
    constructors = pd.read_csv(DATA / "constructors.csv", usecols=["name"])
    status = pd.read_csv(DATA / "status.csv")

    return models, features, le_nat, le_con, grid, races, constructors, status


MODELS_DICT, FEATURES, LE_NAT, LE_CON, GRID, RACES, CONSTRUCTORS, STATUS = load_all()

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "gradient_boost": "Gradient Boosting",
    "xgboost": "XGBoost",
}

# 2025 team names -> historical names used in the training data.
# The first candidate found in the encoder's classes is used.
TEAM_CANDIDATES = {
    "Red Bull Racing": ["Red Bull"],
    "Ferrari": ["Ferrari"],
    "Mercedes": ["Mercedes"],
    "McLaren": ["McLaren", "McLaren-Mercedes"],
    "Aston Martin": ["Aston Martin"],
    "Alpine": ["Alpine F1 Team", "Alpine"],
    "Williams": ["Williams"],
    "Haas F1 Team": ["Haas F1 Team"],
    "Kick Sauber": ["Sauber", "Alfa Romeo", "Alfa Romeo Racing"],
    "RB": ["RB F1 Team", "AlphaTauri", "Toro Rosso"],
}


def safe_encode(le, value, fallbacks):
    """Encode a label; fall back to the first candidate the encoder knows."""
    for candidate in [value, *fallbacks]:
        if candidate in le.classes_:
            return int(le.transform([candidate])[0]), candidate
    return 0, None


def predict_podium(row: dict) -> dict:
    """Return podium probability per model for one feature row."""
    x = pd.DataFrame([row], columns=FEATURES)
    out = {}
    for name, model in MODELS_DICT.items():
        out[name] = float(model.predict_proba(x)[0, 1])
    return out


def build_row(driver: dict, grid_pos: int, year: int, laps: int,
              status_id: int, fastest_lap: int, rank: int, fl_speed: float) -> tuple[dict, list]:
    nat_enc, nat_used = safe_encode(LE_NAT, driver["nationality"], ["British"])
    con_enc, con_used = safe_encode(
        LE_CON, driver["team_name"], TEAM_CANDIDATES.get(driver["team_name"], []))
    warnings = []
    if nat_used is None:
        warnings.append(f"Nationality '{driver['nationality']}' unseen in training - used fallback encoding.")
    if con_used is None:
        warnings.append(f"Team '{driver['team_name']}' unseen in training - used fallback encoding.")
    row = {
        "grid": grid_pos,
        "laps": laps,
        "fastestLap": fastest_lap,
        "rank": rank,
        "fastestLapSpeed": fl_speed,
        "year": year,
        "statusId": status_id,
        "nationality_encoded": nat_enc,
        "constructor_encoded": con_enc,
    }
    return row, warnings


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Race setup")
min_year, max_year = int(RACES["year"].min()), int(RACES["year"].max())
year = st.sidebar.slider("Season", min_year, max_year, min(max_year, 2025))

grid_pos = st.sidebar.number_input("Grid position", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("Race assumptions")
with st.sidebar.expander("Post-race features (advanced)"):
    st.caption("The models also use these, which are only known "
               "during/after a race. Defaults assume a normal race finish.")
    laps = st.slider("Laps completed", 0, 80, 58)
    status_options = {int(r.statusId): r.status for r in STATUS.itertuples()
                      if r.status in ("Finished", "Accident", "Collision", "Engine",
                                      "Gearbox", "Hydraulics", "Disqualified", "Withdrew")}
    status_id = st.selectbox("Result status", list(status_options.keys()),
                             format_func=lambda i: f"{i} - {status_options[i]}", index=0)
    fastest_lap = st.slider("Fastest lap set on lap #", 1, 80, 40)
    rank = st.slider("Fastest-lap rank in race", 1, 24, 8)
    fl_speed = st.slider("Fastest lap speed (km/h)", 180.0, 260.0, 220.0, step=1.0)

# ---------------------------------------------------------------- tabs
tab_single, tab_grid, tab_about = st.tabs(["🏁 Single driver", "📊 Full grid simulation", "ℹ️ About the models"])

with tab_single:
    st.subheader("Pick a driver and grid slot")
    drivers = GRID.sort_values("team_name")
    choice = st.selectbox(
        "Driver (2025 grid)",
        drivers.itertuples(),
        format_func=lambda d: f"{d.driver_name} - {d.team_name}",
        index=list(drivers["driver_name"]).index("Max Verstappen"),
    )
    driver = {"nationality": choice.nationality, "team_name": choice.team_name,
              "name": choice.driver_name}

    row, warnings = build_row(driver, grid_pos, year, laps, status_id,
                              fastest_lap, rank, fl_speed)
    probs = predict_podium(row)

    for w in warnings:
        st.warning(w)

    st.markdown(f"### {choice.driver_name} ({choice.team_name}), P{grid_pos} - {year}")
    cols = st.columns(3)
    for col, (key, label) in zip(cols, MODEL_LABELS.items()):
        col.metric(label, f"{probs[key]*100:.1f}%",
                   delta="Podium" if probs[key] >= 0.5 else "Not projected",
                   delta_color="normal" if probs[key] >= 0.5 else "off")

    chart_df = pd.DataFrame(
        {"Podium probability": [probs[k] for k in MODEL_LABELS]},
        index=list(MODEL_LABELS.values()),
    )
    st.bar_chart(chart_df, horizontal=True, height=250)

    best = max(probs, key=probs.get)
    st.success(f"**{MODEL_LABELS[best]}** is most confident: "
               f"{probs[best]*100:.1f}% chance of a top-3 finish.")

with tab_grid:
    st.subheader(f"Full {year} grid simulation")
    st.caption("Uses every 2025 driver in the order of the grid file as their grid "
               "position, plus the race assumptions from the sidebar.")
    rows, names, teams = [], [], []
    for i, d in enumerate(GRID.itertuples(), start=1):
        row, _ = build_row({"nationality": d.nationality, "team_name": d.team_name,
                            "name": d.driver_name},
                           i, year, laps, status_id, fastest_lap, rank, fl_speed)
        probs = predict_podium(row)
        rows.append(max(probs.values()))  # most optimistic model
        names.append(d.driver_name)
        teams.append(d.team_name)

    df = (pd.DataFrame({"Driver": names, "Team": teams, "Podium probability": rows})
          .sort_values("Podium probability", ascending=False)
          .reset_index(drop=True))
    df.index = df.index + 1
    df.index.name = "Projected finish"

    def highlight_top3(r):
        return ["background-color: #1a7f37" if r.name <= 3 else "" for _ in r]

    st.dataframe(df.style.format({"Podium probability": "{:.1%}"}).apply(highlight_top3, axis=1),
                 width="stretch")
    st.bar_chart(df.set_index("Driver")["Podium probability"], height=450)

with tab_about:
    st.subheader("What this app uses")
    st.markdown(
        "Three classifiers trained in `train_models.py` on the Ergast F1 dataset "
        "to predict whether a driver finishes on the **podium (top 3)**:\n\n"
        "| Model | File |\n|---|---|\n"
        "| Random Forest | `models/f1_model_random_forest.joblib` |\n"
        "| Gradient Boosting | `models/f1_model_gradient_boost.joblib` |\n"
        "| XGBoost | `models/f1_model_xgboost.joblib` |\n\n"
        "Input features: grid position, laps, fastest lap, fastest-lap rank, "
        "fastest-lap speed, season year, result status, driver nationality, "
        "and constructor (both label-encoded)."
    )
    if (ROOT / "feature_importance.png").exists():
        st.image(str(ROOT / "feature_importance.png"), caption="Random Forest feature importance")
