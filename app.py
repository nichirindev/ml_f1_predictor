"""Pitwall: an interactive F1 podium scenario lab. Run: streamlit run app.py."""
from html import escape
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from prediction import MODEL_LABELS, build_features, predict_models, select_scores, validate_grid

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
def load_models():
    models = {key: joblib.load(MODELS / f"f1_model_{key}.joblib") for key in MODEL_LABELS}
    return models, joblib.load(MODELS / "preprocessors.joblib")


@st.cache_data
def load_data():
    return (pd.read_csv(DATA / "f1_2025_grid.csv"), pd.read_csv(DATA / "races.csv"),
            pd.read_csv(DATA / "status.csv"))


st.sidebar.markdown('<div class="brand">PIT<span>WALL</span> / F1</div>', unsafe_allow_html=True)
st.sidebar.caption("PODIUM SCENARIO LAB")
st.title("Every grid slot tells a story.")
st.write("Explore the grid. Compare the models. Build your race scenario.")
st.info("Scenario analysis · These models use post-race features. Estimates depend on your assumptions and are not validated pre-race forecasts.", icon="ℹ️")

missing = [f"f1_model_{key}.joblib" for key in MODEL_LABELS] + ["preprocessors.joblib"]
if any(not (MODELS / name).exists() for name in missing):
    st.warning("Train the models to open your analysis workspace.")
    st.code("python train_models.py\npython -m streamlit run app.py", language="bash")
    st.caption("Training uses the historical CSV files in data/ and saves the model artifacts locally.")
    st.stop()
try:
    models, prep = load_models()
    grid, races, statuses = load_data()
except Exception as exc:
    st.error(f"Could not load the local data or model artifacts: {exc}")
    st.caption("Check data/ and retrain the models with the installed dependencies.")
    st.stop()

st.sidebar.subheader("Race setup")
selection = st.sidebar.selectbox("Scoring model", ["Model average", *MODEL_LABELS.values()],
    help="The average gives each model equal weight. All views use the same scoring choice.")
min_year, max_year = int(races.year.min()), int(races.year.max())
year = st.sidebar.number_input("Scenario season", min_year, max(max_year, 2025), 2025)
st.sidebar.caption(f"Historical race data: {min_year}–{max_year}. Driver roster: static 2025 snapshot.")
threshold = st.sidebar.slider("Podium score threshold", 0.05, 0.95, 0.5, 0.05,
    help="Labels scores above this value; it does not change the model or force exactly three podiums.")
with st.sidebar.expander("Race assumptions", expanded=False):
    st.caption("Shared across drivers. Change these to explore a hypothetical finish.")
    laps = st.slider("Laps completed", 0, 100, 58)
    status_options = dict(zip(statuses.statusId.astype(int), statuses.status))
    status_id = st.selectbox("Result status", list(status_options),
                            format_func=lambda key: status_options[key])
    fastest_lap = st.slider("Fastest lap set on lap", 1, max(laps, 1), min(40, max(laps, 1)))
    rank = st.slider("Fastest-lap rank", 1, 24, 8)
    speed = st.slider("Fastest-lap speed (km/h)", 100.0, 300.0, 220.0, 1.0)
assumptions = {"year": year, "laps": laps, "statusId": status_id, "fastestLap": fastest_lap,
               "rank": rank, "fastestLapSpeed": speed}
st.sidebar.divider()
st.sidebar.caption("Independent binary classifiers estimate top-three membership. Scores across a grid need not sum to three; rankings are not predicted finishing positions.")


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


def alias_note(aliases):
    if aliases:
        st.caption("Historical team mapping: " + " · ".join(aliases))
    st.caption("Driver identity is not a model feature. Drivers with the same team and nationality receive identical scores at equal grid positions.")


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
    position = right.number_input("Starting position", 1, len(grid), 5)
    entries = driver_entries([name], [position])
    probabilities, scores, aliases = score(entries)
    st.subheader(f"{name} / P{position:02}")
    st.caption(f"{entries.iloc[0].team_name} · {year} scenario · {selection}")
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
    st.caption("Only starting position changes in this sweep; all race assumptions stay fixed.")
    alias_note(aliases)

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
        export = result.assign(**assumptions, scoring_model=selection, threshold=threshold)
        st.download_button("Download scenario CSV", export.to_csv(index=False).encode("utf-8"),
                           "pitwall_scenario.csv", "text/csv")
        alias_note(aliases)

with compare:
    st.subheader("Two drivers. Shared assumptions.")
    col_a, col_b = st.columns(2)
    first = col_a.selectbox("Driver A", names, key="driver_a")
    second = col_b.selectbox("Driver B", [n for n in names if n != first], key="driver_b")
    slot_a = col_a.number_input("Starting position A", 1, len(grid), 1)
    slot_b = col_b.number_input("Starting position B", 1, len(grid), 2)
    probabilities, scores, aliases = score(driver_entries([first, second], [slot_a, slot_b]))
    col_a.metric(first, f"{scores.iloc[0]:.1%}")
    col_b.metric(second, f"{scores.iloc[1]:.1%}")
    gap = (scores.iloc[0] - scores.iloc[1]) * 100
    st.write(f"{first}'s podium score is {abs(gap):.1f} percentage points {'higher' if gap >= 0 else 'lower'} than {second}'s.")
    comparison = probabilities.set_axis([first, second]).rename_axis("Driver").reset_index().melt(
        id_vars="Driver", var_name="Model", value_name="Podium score")
    score_chart(comparison, "Model", 300, group="Driver")
    st.caption("Compares top-three scores, not the probability that one driver beats the other.")
    alias_note(aliases)

with evidence:
    st.subheader("Know what sits behind the score.")
    st.write("Random Forest, Gradient Boosting and XGBoost are trained on historical Ergast race entries. Each predicts podium membership using the same nine features.")
    metrics_path = MODELS / "metrics.json"
    if metrics_path.exists():
        try:
            report = json.loads(metrics_path.read_text())
            metrics = pd.DataFrame(report["models"]).T.rename(index=MODEL_LABELS)
            st.dataframe(metrics.rename(columns={"f1": "Podium F1", "roc_auc": "ROC-AUC", "time": "Train seconds"}),
                         width="stretch")
            st.caption(f"Local training run · {report['samples']:,} entries · {report['train_samples']:,} train / {report['test_samples']:,} test · 80/20 stratified random split, seed 42.")
        except (ValueError, KeyError, TypeError):
            st.warning("The training report could not be read. Retrain to regenerate it.")
    else:
        st.caption("Retrain with python train_models.py to generate metrics for your local artifacts.")
    st.markdown("""
**Read the results with these limits in mind**

- **Post-race inputs:** completed laps, fastest lap, lap rank, speed and result status leak race outcomes. Holdout metrics do not measure pre-race forecasting skill.
- **Random holdout:** the split mixes seasons; it does not test performance on a future season.
- **Uncalibrated estimates:** averaging model outputs does not calibrate them or enforce three podium places.
- **Static roster:** this is the bundled 2025 roster, with historical constructor aliases. Driver form, circuit, weather and team upgrades are not features.

A true pre-race model needs features available before lights out, rolling history built without future results, and season-based backtesting.
""")
    with st.expander("Inspect the current model inputs"):
        st.code("\n".join(prep["features"]), language="text")
    importance = pd.DataFrame({MODEL_LABELS[key]: model.feature_importances_ for key, model in models.items()},
                              index=prep["features"])
    st.subheader("Feature importance")
    st.bar_chart(importance.sort_values("Random Forest"), horizontal=True, stack=False, height=400)
    st.caption("Built-in model importance describes model dependence, not a causal effect.")

st.divider()
st.caption("PITWALL / Local F1 scenario analysis · Historical data, explicit assumptions.")
