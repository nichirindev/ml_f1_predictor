"""Pitwall: an interactive F1 podium scenario lab. Run: streamlit run app.py."""
from datetime import date, timedelta
from html import escape
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from prediction import (MODEL_LABELS, build_features, predict_models, select_scores,
                        validate_bundle, validate_grid)

ROOT = Path(__file__).parent
DATA, MODELS = ROOT / "data", ROOT / "models"
st.set_page_config(page_title="Pitwall · F1 Podium Lab", page_icon="🏁", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600;700&display=swap');
:root { --bg:oklch(19.5% .014 260); --surface:oklch(25% .018 260);
--fg:oklch(95% .006 260); --muted:oklch(77% .018 260);
--border:oklch(39% .02 260); --accent:oklch(69% .21 28);
--radius:4px; --radius-panel:8px; }
html, body, [data-testid="stApp"] {font-family:'Barlow',sans-serif; color:var(--fg);}
h1,h2,h3,[data-testid="stMetricValue"] {font-family:'Barlow Condensed',sans-serif;}
h1 {font-size:3rem !important; letter-spacing:-.02em;}
[data-testid="stMainBlockContainer"] {padding-top:2rem; padding-bottom:3rem; max-width:1500px;}
[data-testid="stSidebar"] {border-right:1px solid var(--border);}
[data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;}
[data-testid="stMetric"] {padding:16px; background:var(--surface); border-radius:var(--radius);}
[data-baseweb="tab-list"] {gap:24px; border-bottom:1px solid var(--border);}
button[data-baseweb="tab"] {font-size:16px; min-height:48px;}
.brand {font-family:'Barlow Condensed',sans-serif; font-size:24px; font-weight:700; letter-spacing:.08em;}
.brand span {color:var(--accent);}
.driver-name {font-family:'Barlow Condensed',sans-serif; font-size:28px; margin:4px 0;}
.detail {color:var(--muted); font-size:16px;}
.podium {padding:20px; background:var(--surface); border-radius:var(--radius-panel); min-height:160px;}
.podium strong {color:var(--accent); font-variant-numeric:tabular-nums;}
:focus-visible {outline:2px solid var(--accent) !important; outline-offset:3px;}
@media(max-width:640px) {h1 {font-size:2.3rem !important;} [data-testid="stMainBlockContainer"] {padding:1rem;}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle(modified_ns):
    # The file timestamp invalidates cached models when an atomic retrain finishes.
    bundle = joblib.load(MODELS / "prerace_bundle.joblib")
    validate_bundle(bundle)
    return bundle


st.sidebar.markdown('<div class="brand">PIT<span>WALL</span> / F1</div>', unsafe_allow_html=True)
st.sidebar.caption("PRE-RACE PODIUM LAB")
st.title("Every grid slot tells a story.")
st.write("Explore the grid. Compare the models. Build your race scenario.")
st.info("Pre-race models · Starting grid and strictly earlier race results only. Evaluated on later seasons, with a separate season for probability calibration.", icon="ℹ️")

bundle_path = MODELS / "prerace_bundle.joblib"
if not bundle_path.exists():
    st.warning("Train the upgraded pre-race models to open your analysis workspace.")
    st.code("python train_models.py\npython -m streamlit run app.py", language="bash")
    st.caption("Legacy retrospective artifacts are incompatible. Retraining builds one versioned pre-race bundle.")
    st.stop()
try:
    prep = load_bundle(bundle_path.stat().st_mtime_ns)
    models, grid, circuits = prep["models"], prep["roster"], prep["circuits"]
except Exception as exc:
    st.error(f"Could not load the pre-race model bundle: {exc}")
    st.caption("Retrain with python train_models.py using the installed dependencies.")
    st.stop()

st.sidebar.subheader("Race setup")
selection = st.sidebar.selectbox("Scoring model", ["Model average", *MODEL_LABELS.values()],
    help="The average gives each calibrated model equal weight. All views use the same scoring choice.")
cutoff = date.fromisoformat(prep["trained_through"])
race_date = st.sidebar.date_input("Race date", value=max(date(cutoff.year + 1, 3, 16), cutoff + timedelta(days=7)),
                                  min_value=cutoff + timedelta(days=1),
                                  help="The deployed models have seen data through the cutoff. Earlier dates require backtest models.")
round_number = st.sidebar.number_input("Race round", 1, 30, 1)
circuit_names = circuits.set_index("circuitId")["name"].to_dict()
circuit_id = st.sidebar.selectbox("Circuit", list(circuit_names), format_func=lambda key: circuit_names[key])
year = race_date.year
st.sidebar.caption(f"History and calibration end {cutoff:%d %b %Y}. No later results are included. Roster: bundled 2025 snapshot.")
threshold = st.sidebar.slider("Podium score threshold", 0.05, 0.95, 0.5, 0.05,
    help="Labels scores above this value; it does not change the model or force exactly three podiums.")
assumptions = {"race_date": race_date.isoformat(), "round": round_number, "circuitId": circuit_id}
st.sidebar.divider()
st.sidebar.caption("Form uses the last five driver starts and five team races; circuit history uses the last three visits. Rookie priors are disclosed below. Starting position is a post-qualifying input; use 0 for a pit-lane start in driver views.")
st.sidebar.caption("Independent classifiers estimate top-three membership. Scores across a grid need not sum to three; rankings are not predicted finishing positions.")


def score(entries):
    try:
        features, aliases = build_features(entries, prep, assumptions)
        probabilities = predict_models(models, features)
        return probabilities, select_scores(probabilities, selection), aliases
    except ValueError as exc:
        st.error(str(exc))
        st.stop()


def driver_entries(names, positions):
    entries = grid.set_index("driver_name").loc[names].reset_index()
    entries["Grid"] = positions
    return entries


def history_note(notes):
    for note in notes:
        if "no recorded F1 starts" in note:
            st.caption(note)
    if notes:
        with st.expander("History coverage and team mappings"):
            for note in notes:
                st.write(note)
    st.caption(f"Form snapshot: {cutoff:%d %b %Y}. Changing the race date does not invent later race results.")


def probability_columns():
    return {name: st.column_config.ProgressColumn(name, min_value=0.0, max_value=1.0, format="percent")
            for name in ["Podium score", *MODEL_LABELS.values()]}


def score_chart(data, category, height, group=None):
    """Use a fixed percentage scale so scores compare honestly across scenarios."""
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("Podium score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y(f"{category}:N", sort=None, title=None),
        tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip("Podium score:Q", format=".1%")],
    )
    if group:
        chart = chart.encode(color=alt.Color(f"{group}:N", scale=alt.Scale(range=["#ff5349", "#8abed1"])),
                             yOffset=f"{group}:N",
                             tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{group}:N"),
                                      alt.Tooltip("Podium score:Q", format=".1%")])
    else:
        chart = chart.mark_bar(color="#ff5349")
    st.altair_chart(chart.properties(height=height), width="stretch")


single, full_grid, compare, evidence = st.tabs(["Driver analysis", "Grid simulator", "Head to head", "Model evidence"])
names = grid.driver_name.tolist()
with single:
    left, right = st.columns([2, 1])
    name = left.selectbox("Driver · 2025 roster", names)
    position = right.number_input("Starting position", 0, len(grid), 5)
    analysis_entries = driver_entries([name], [position])
    probabilities, scores, aliases = score(analysis_entries)
    starting_label = f"P{position:02}" if position else "Pit lane"
    st.subheader(f"{name} / {starting_label}")
    st.caption(f"{analysis_entries.iloc[0].team_name} · {circuit_names[circuit_id]} · {race_date:%d %b %Y} · {selection}")
    a, b, c = st.columns([2, 1, 1])
    a.metric("Podium score", f"{scores.iloc[0]:.1%}")
    b.metric("Model spread", f"{(probabilities.iloc[0].max() - probabilities.iloc[0].min()) * 100:.1f} pp")
    c.metric("Threshold status", "Above" if scores.iloc[0] >= threshold else "Below")
    st.caption("Model spread is the difference between the highest and lowest model score, not a confidence interval.")
    st.subheader("Three models, one scenario")
    score_chart(probabilities.iloc[0].rename_axis("Model").reset_index(name="Podium score"), "Model", 200)
    st.subheader("What if you started elsewhere?")
    sensitivity = driver_entries([name] * len(grid), list(range(1, len(grid) + 1)))
    _, slot_scores, _ = score(sensitivity)
    sweep = pd.DataFrame({"Starting position": range(1, len(grid) + 1), "Podium score": slot_scores})
    st.altair_chart(alt.Chart(sweep).mark_line(color="#ff5349", point=True).encode(
        x=alt.X("Starting position:Q", axis=alt.Axis(tickMinStep=1), scale=alt.Scale(domain=[1, len(grid)])),
        y=alt.Y("Podium score:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=["Starting position", alt.Tooltip("Podium score:Q", format=".1%")],
    ).properties(height=230), width="stretch")
    st.caption("Only starting position changes in this sweep; driver form, team form and circuit history stay fixed.")
    history_note(aliases)

with full_grid:
    st.subheader("Set the grid. See the contenders.")
    st.caption("Edit starting positions below. Each slot must be used once. Initial order follows the roster file, not qualifying results.")
    initial = grid[["driver_name", "team_name"]].rename(columns={"driver_name": "Driver", "team_name": "Team"})
    initial.insert(0, "Grid", range(1, len(grid) + 1))
    edited = st.data_editor(initial, hide_index=True, width="stretch", disabled=["Driver", "Team"],
        column_config={"Grid": st.column_config.NumberColumn("Grid", min_value=1, max_value=len(grid), step=1, required=True)},
        key="starting_grid")
    try:
        validate_grid(edited)
    except ValueError as exc:
        st.error(str(exc))
    else:
        entries = driver_entries(edited.Driver.tolist(), edited.Grid.tolist())
        probabilities, scores, aliases = score(entries)
        result = edited.copy()
        result["Podium score"] = scores.to_numpy()
        result["Threshold"] = ["Above" if value >= threshold else "Below" for value in scores]
        for column in probabilities:
            result[column] = probabilities[column].to_numpy()
        result = result.sort_values(["Podium score", "Grid"], ascending=[False, True]).reset_index(drop=True)
        result.insert(0, "Score rank", range(1, len(result) + 1))
        st.subheader("Highest-scoring contenders")
        for col, (_, driver) in zip(st.columns(3), result.head(3).iterrows()):
            with col:
                st.markdown(f'<div class="podium"><strong>RANK {driver["Score rank"]:02} / {driver["Podium score"]:.1%}</strong>'
                    f'<div class="driver-name">{escape(driver.Driver)}</div>'
                    f'<div class="detail">{escape(driver.Team)} · starts P{driver.Grid:02}</div></div>', unsafe_allow_html=True)
        st.caption("Top three by score; this is not a joint podium forecast. Ties use starting position.")
        st.dataframe(result, hide_index=True, width="stretch", column_config=probability_columns())
        score_chart(result[["Driver", "Podium score"]], "Driver", 560)
        export = result.assign(**assumptions, circuit=circuit_names[circuit_id],
            scoring_model=selection, threshold=threshold, trained_through=prep["trained_through"],
            model_version=prep["schema_version"], model_created_at=prep["metrics"]["created_at"])
        st.download_button("Download scenario CSV", export.to_csv(index=False).encode("utf-8"),
                           "pitwall_scenario.csv", "text/csv")
        history_note(aliases)

with compare:
    st.subheader("Two drivers. Shared race context.")
    col_a, col_b = st.columns(2)
    first = col_a.selectbox("Driver A", names, key="driver_a")
    second = col_b.selectbox("Driver B", [n for n in names if n != first], key="driver_b")
    slot_a = col_a.number_input("Starting position A", 0, len(grid), 1)
    slot_b = col_b.number_input("Starting position B", 0, len(grid), 2)
    probabilities, scores, aliases = score(driver_entries([first, second], [slot_a, slot_b]))
    col_a.metric(first, f"{scores.iloc[0]:.1%}")
    col_b.metric(second, f"{scores.iloc[1]:.1%}")
    gap = (scores.iloc[0] - scores.iloc[1]) * 100
    st.write(f"{first}'s podium score is {abs(gap):.1f} percentage points {'higher' if gap >= 0 else 'lower'} than {second}'s.")
    comparison = probabilities.set_axis([first, second]).rename_axis("Driver").reset_index().melt(
        id_vars="Driver", var_name="Model", value_name="Podium score")
    score_chart(comparison, "Model", 300, group="Driver")
    st.caption("Compares top-three scores, not the probability that one driver beats the other.")
    history_note(aliases)

with evidence:
    st.subheader("Tested on seasons the models had not seen.")
    st.write("Random Forest, Gradient Boosting and XGBoost use starting position, race round, prior driver and team form, and prior circuit results. Current-race finishing data never enters the feature set.")
    report = prep["metrics"]
    labels = {**MODEL_LABELS, "model_average": "Model average", "prior_rate": "Prior-rate baseline", "grid_top3": "Grid top-three baseline"}
    columns = ["f1", "roc_auc", "average_precision", "brier", "raw_brier", "top3_recall"]
    metric_labels = {"f1": "Podium F1", "roc_auc": "ROC-AUC", "average_precision": "Avg. precision",
                     "brier": "Brier (lower is better)", "raw_brier": "Brier before calibration",
                     "top3_recall": "Top-three pick recall"}
    metrics = pd.DataFrame(report["models"]).T.rename(index=labels)
    st.dataframe(metrics[columns].rename(columns=metric_labels).round(4), width="stretch")
    st.caption(f"Walk-forward backtests: {', '.join(map(str, report['backtest_years']))} · {report['test_entries']:,} test entries across {report['test_races']} races. F1 uses a fixed 0.5 threshold. Top-three pick recall is the fraction of actual podium finishers found among the three highest scores per race.")
    st.caption("The prior-rate baseline uses training-set podium prevalence. The grid baseline selects starting slots 1–3, with hard 0/1 scores. Its Brier and log loss are not calibrated probability benchmarks.")
    st.subheader("Season-by-season backtest")
    folds = [{"Test season": fold["test_year"], "Base fit through": fold["train_through"],
              "Calibration season": fold["calibration_year"], "Model": labels[key],
              "Podium F1": values["f1"], "Brier": values["brier"], "Top-three pick recall": values["top3_recall"]}
             for fold in report["folds"] for key, values in fold["models"].items()]
    st.dataframe(pd.DataFrame(folds).round(4), hide_index=True, width="stretch")
    deployment = report["deployment"]
    st.caption(f"Deployment bundle: base models fitted through {deployment['base_train_through']}; sigmoid calibration on {deployment['calibration_year']}; form history through {prep['trained_through']}. Backtest models are separate from this final deployment fit.")
    st.markdown("""
**How the time boundary works**

- Each entry reads history from strictly earlier race dates. Team form updates after all teammates have been scored, so there is no within-race leakage.
- Base model fitting, probability calibration and test seasons are separate. During a test season, form can use already-completed races, while model weights stay fixed.
- The individual models use sigmoid calibration. Their equal-weight average is not separately calibrated; Brier scores report observed held-out probability error.
- The static 2025 roster and data ending in 2024 are not a live forecast service. Rookies use fixed priors plus their team's available history. Team changes follow the selected roster's constructor.
- Qualifying times, weather, tyres and current car upgrades are absent. Scores are independent and do not enforce exactly three podium places.
""")
    with st.expander("Inspect the selected driver's pre-race inputs"):
        selected_features, _ = build_features(analysis_entries, prep, assumptions)
        st.dataframe(selected_features.T.rename(columns={0: "Feature value"}), width="stretch")
    importance = pd.DataFrame(prep["feature_importance"], index=prep["features"])
    st.subheader("Feature importance")
    st.bar_chart(importance.sort_values("Random Forest"), horizontal=True, stack=False, height=480)
    st.caption("Base-estimator importance describes model dependence, not a causal effect or a calibrated contribution.")
    st.download_button("Download backtest report", json.dumps(report, indent=2),
                       "pitwall_backtest.json", "application/json")

st.divider()
st.caption("PITWALL / Pre-race F1 podium analysis · Chronological backtests, explicit data cutoff.")
