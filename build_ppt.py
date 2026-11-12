"""
Builds F1_Podium_Prediction.pptx - a complete project presentation.
Usage: .venv/Scripts/python.exe build_ppt.py
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = Path(__file__).parent

# ---- theme -------------------------------------------------------------
RED   = RGBColor(0xE1, 0x06, 0x00)   # F1 red
DARK  = RGBColor(0x15, 0x15, 0x1E)   # F1 black
GRAY  = RGBColor(0x5A, 0x5A, 0x6A)
LIGHT = RGBColor(0xF2, 0xF2, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x1A, 0x7F, 0x37)

SW, SH = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


# ---- helpers -----------------------------------------------------------
def add_slide():
    return prs.slides.add_slide(BLANK)


def rect(slide, x, y, w, h, color, line=False):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = color
    if not line:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def textbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    return tb, tf


def set_run(r, text, size, color=DARK, bold=False, italic=False, font="Calibri"):
    r.text = text
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = font


def bullets(tf, items, size=16, color=DARK, spacing=6):
    """items: list of (text, level, bold, color_override) or plain strings."""
    first = True
    for it in items:
        text, level, bold, col = (it + (0, False, None))[:4] if isinstance(it, tuple) else (it, 0, False, None)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(spacing)
        r = p.add_run()
        set_run(r, ("•  " if level == 0 else "–  ") + text if text else "",
                size - (1 if level else 0), col or color, bold)


def content_slide(title, footer_num):
    s = add_slide()
    rect(s, Inches(0.55), Inches(0.42), Inches(0.12), Inches(0.62), RED)
    _, tf = textbox(s, Inches(0.8), Inches(0.3), Inches(12), Inches(0.9))
    p = tf.paragraphs[0]
    set_run(p.add_run(), title, 28, DARK, bold=True)
    rect(s, Inches(0.55), Inches(1.22), Inches(12.23), Pt(1.6), LIGHT)
    # footer
    _, tf = textbox(s, Inches(0.55), Inches(7.05), Inches(6), Inches(0.35))
    set_run(tf.paragraphs[0].add_run(), "F1 Podium Prediction with Machine Learning", 10, GRAY)
    _, tf = textbox(s, Inches(12.2), Inches(7.05), Inches(0.7), Inches(0.35))
    tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
    set_run(tf.paragraphs[0].add_run(), str(footer_num), 10, GRAY)
    return s


def style_table(table, header_fill=DARK, font_size=13):
    for ci in range(len(table.columns)):
        cell = table.cell(0, ci)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(font_size)
                r.font.bold = True
                r.font.color.rgb = WHITE
    for ri in range(1, len(table.rows)):
        for ci in range(len(table.columns)):
            cell = table.cell(ri, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if ri % 2 else LIGHT
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                for r in p.runs:
                    r.font.size = Pt(font_size)
                    r.font.color.rgb = DARK


def fill_table(table, data):
    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            table.cell(ri, ci).text = str(val)


# ======================================================================
# 1 - TITLE
# ======================================================================
s = add_slide()
rect(s, 0, 0, SW, SH, DARK)
rect(s, Inches(0.9), Inches(2.35), Inches(2.2), Inches(0.09), RED)
_, tf = textbox(s, Inches(0.9), Inches(2.55), Inches(11.5), Inches(2.2))
p = tf.paragraphs[0]
set_run(p.add_run(), "F1 Podium Prediction", 48, WHITE, bold=True)
p = tf.add_paragraph()
set_run(p.add_run(), "with Machine Learning", 48, RED, bold=True)
p = tf.add_paragraph()
p.space_before = Pt(14)
set_run(p.add_run(), "End-to-end ML project  •  Data ingestion → Preprocessing → Model training → "
                     "Evaluation → Deployment", 18, GRAY)
_, tf = textbox(s, Inches(0.9), Inches(6.35), Inches(11), Inches(0.7))
set_run(tf.paragraphs[0].add_run(), "Predicting Formula 1 top-3 finishes with Random Forest, "
        "Gradient Boosting & XGBoost", 14, GRAY, italic=True)

# ======================================================================
# 2 - AGENDA
# ======================================================================
s = content_slide("Agenda", 2)
_, tf = textbox(s, Inches(0.8), Inches(1.6), Inches(6.6), Inches(5.3))
bullets(tf, [
    ("Problem Statement", 0, True),
    ("Dataset & Data Sources", 0, True),
    ("Preprocessing & Feature Engineering", 0, True),
    ("Models & Methodology", 0, True),
    ("Results & Model Comparison", 0, True),
    ("Deployment - Streamlit Front End", 0, True),
    ("Limitations & Future Work", 0, True),
    ("Conclusion", 0, True),
], size=20, spacing=14)
_, tf = textbox(s, Inches(8.2), Inches(2.1), Inches(4.3), Inches(3.6))
bullets(tf, [
    "3 models trained & compared",
    "8,252 historical race entries",
    "75 years of F1 data (1950-2024)",
    "Full 2025 grid simulation",
], size=14, color=GRAY, spacing=10)
for i, val in enumerate(["0.90", "0.96", "0.72"]):
    _, tf = textbox(s, Inches(8.2), Inches(4.35) + Inches(i * 0.78), Inches(4.3), Inches(0.75))
    p = tf.paragraphs[0]
    set_run(p.add_run(), val + "  ", 26, RED, bold=True)
    set_run(p.add_run(), ["Accuracy (best)", "ROC-AUC", "F1 - podium class"][i], 15, DARK)

# ======================================================================
# 3 - PROBLEM STATEMENT
# ======================================================================
s = content_slide("Problem Statement", 3)
_, tf = textbox(s, Inches(0.8), Inches(1.55), Inches(6.9), Inches(5.2))
bullets(tf, [
    ("Goal", 0, True),
    ("Predict whether a Formula 1 driver will finish a race on the podium (position 1-3).", 1),
    ("", 0),
    ("Why it matters", 0, True),
    ("Race-strategy support and what-if analysis before and during a race weekend.", 1),
    ("Fan engagement and prediction games.", 1),
    ("A reproducible framework that can extend to winner prediction and championship forecasts.", 1),
    ("", 0),
    ("Why it is hard", 0, True),
    ("Podiums are rare - only ~15% of all race entries finish top-3 (heavily imbalanced classes).", 1),
    ("Performance depends on driver skill, car performance, team strategy and reliability combined.", 1),
    ("75 years of historical data with rule changes, team renames and evolving competition.", 1),
], size=15, spacing=5)
rect(s, Inches(8.1), Inches(1.7), Inches(4.6), Inches(4.6), LIGHT)
_, tf = textbox(s, Inches(8.35), Inches(1.95), Inches(4.1), Inches(4.2))
p = tf.paragraphs[0]
set_run(p.add_run(), "Framed as", 14, GRAY, bold=True)
p = tf.add_paragraph()
set_run(p.add_run(), "Binary Classification", 24, RED, bold=True)
p = tf.add_paragraph()
p.space_before = Pt(8)
bullets(tf, [
    ("Class 1  →  finishing position 1-3 (podium)", 0),
    ("Class 0  →  position 4+ or DNF", 0),
    ("", 0),
    ("Metric focus: F1-score of the podium class (accuracy alone is misleading at 85/15 imbalance).", 0),
], size=14, spacing=6)

# ======================================================================
# 4 - DATASET
# ======================================================================
s = content_slide("Dataset - Ergast Formula 1 Database (1950-2024)", 4)
tbl = s.shapes.add_table(8, 3, Inches(0.8), Inches(1.55), Inches(7.3), Inches(4.4)).table
tbl.columns[0].width, tbl.columns[1].width, tbl.columns[2].width = Inches(3.1), Inches(2.2), Inches(2.0)
fill_table(tbl, [
    ["File", "Rows", "Role"],
    ["results.csv", "26,759", "Target + core features"],
    ["races.csv", "1,125", "Season, circuit context"],
    ["drivers.csv", "861", "Driver, nationality"],
    ["constructors.csv", "212", "Team identity"],
    ["qualifying.csv", "10,494", "Grid source (future use)"],
    ["lap_times.csv", "589,081", "Telemetry-level detail"],
    ["status.csv", "139", "Result status labels"],
])
style_table(tbl, font_size=12)
rect(s, Inches(8.4), Inches(1.55), Inches(4.35), Inches(2.5), LIGHT)
_, tf = textbox(s, Inches(8.62), Inches(1.75), Inches(3.95), Inches(2.2))
p = tf.paragraphs[0]
set_run(p.add_run(), "2025 supplement", 16, RED, bold=True)
bullets(tf, [
    ("f1_2025_grid.csv - 20 drivers, 10 teams", 0),
    ("f1_2025_results.csv - 66 race results", 0),
    ("Used by the front end for live predictions", 0),
], size=13, spacing=5)
rect(s, Inches(8.4), Inches(4.25), Inches(4.35), Inches(1.7), DARK)
_, tf = textbox(s, Inches(8.62), Inches(4.45), Inches(3.95), Inches(1.4))
p = tf.paragraphs[0]
set_run(p.add_run(), "1,125 races  •  26,759 entries\n", 15, WHITE, bold=True)
set_run(p.add_run(), "77 circuits, 861 drivers, 212 constructors\n", 12, GRAY)
set_run(p.add_run(), "Seasons 1950 → 2024 (+2025 live data)", 12, GRAY)

# ======================================================================
# 5 - PREPROCESSING
# ======================================================================
s = content_slide("Preprocessing & Feature Engineering", 5)
_, tf = textbox(s, Inches(0.8), Inches(1.5), Inches(6.7), Inches(5.3))
bullets(tf, [
    ("Data preparation (preprocess.py / train_models.py)", 0, True),
    ("Merged results with races (year, circuitId), drivers (nationality) and constructors (name).", 1),
    ("Casted 8 columns to numeric; '\\N' placeholders and DNFs became NaN.", 1),
    ("Dropped rows with missing feature values → 8,252 clean race entries.", 1),
    ("", 0),
    ("Target construction", 0, True),
    ("top3 = 1 if finishing position ≤ 3, else 0 (DNF counts as non-podium).", 1),
    ("Result: 1,230 podiums of 8,252 entries = 14.9% positive class.", 1),
    ("", 0),
    ("Encoding categorical features", 0, True),
    ("LabelEncoder for driver nationality and constructor name.", 1),
    ("Encoders persisted in models/preprocessors.joblib for consistent inference.", 1),
], size=15, spacing=5)
s.shapes.add_picture(str(ROOT / "slide_class_balance.png"), Inches(7.75), Inches(1.9), width=Inches(4.9))
_, tf = textbox(s, Inches(7.75), Inches(6.0), Inches(4.9), Inches(0.9))
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
set_run(p.add_run(), "Strong class imbalance → stratified split + class weights", 13, GRAY, italic=True)

# ======================================================================
# 6 - FEATURES
# ======================================================================
s = content_slide("Feature Set & Target", 6)
tbl = s.shapes.add_table(10, 3, Inches(0.8), Inches(1.5), Inches(7.6), Inches(4.9)).table
tbl.columns[0].width, tbl.columns[1].width, tbl.columns[2].width = Inches(3.0), Inches(2.1), Inches(2.5)
fill_table(tbl, [
    ["Feature", "Type", "Meaning"],
    ["grid", "Numeric", "Starting grid position"],
    ["laps", "Numeric", "Laps completed"],
    ["fastestLap", "Numeric", "Lap no. of fastest lap"],
    ["rank", "Numeric", "Fastest-lap rank in race"],
    ["fastestLapSpeed", "Numeric", "Fastest lap speed (km/h)"],
    ["year", "Numeric", "Season year"],
    ["statusId", "Categorical", "Result status (finished / DNF reason)"],
    ["nationality_encoded", "Categorical", "Driver nationality (label-encoded)"],
    ["constructor_encoded", "Categorical", "Constructor name (label-encoded)"],
])
style_table(tbl, font_size=12)
rect(s, Inches(8.7), Inches(1.5), Inches(4.05), Inches(2.15), LIGHT)
_, tf = textbox(s, Inches(8.92), Inches(1.7), Inches(3.6), Inches(1.85))
p = tf.paragraphs[0]
set_run(p.add_run(), "Target variable", 15, RED, bold=True)
bullets(tf, [
    ("top3 = 1  →  podium finish (pos 1-3)", 0),
    ("top3 = 0  →  position 4+ or DNF", 0),
], size=13, spacing=5)
rect(s, Inches(8.7), Inches(3.85), Inches(4.05), Inches(2.55), LIGHT)
_, tf = textbox(s, Inches(8.92), Inches(4.05), Inches(3.6), Inches(2.25))
p = tf.paragraphs[0]
set_run(p.add_run(), "Modeling note", 15, RED, bold=True)
bullets(tf, [
    ("laps, fastestLap, rank, speed, status are post-race values - they make this a "
     "retrospective / explanatory model, not a pre-race forecast.", 0),
    ("Documented as the key limitation and first item of future work.", 0),
], size=12, spacing=5)

# ======================================================================
# 7 - METHODOLOGY / PIPELINE
# ======================================================================
s = content_slide("Methodology - End-to-End Pipeline", 7)
steps = [
    ("1. Ingest", "Load 13 CSVs from\nErgast database"),
    ("2. Merge & Clean", "Join races / drivers /\nconstructors, cast types"),
    ("3. Feature Eng.", "top3 target, label\ncode nationality & team"),
    ("4. Split", "80/20 stratified\n6,601 / 1,651 samples"),
    ("5. Train", "RF, Gradient Boosting,\nXGBoost (train_models.py)"),
    ("6. Evaluate", "Accuracy, Precision,\nRecall, F1, ROC-AUC"),
    ("7. Persist", "joblib models +\nencoders → models/"),
    ("8. Deploy", "Streamlit app\n(app.py)"),
]
from pptx.enum.shapes import MSO_SHAPE
bw, gap = Inches(2.85), Inches(0.30)
x0, y0 = Inches(0.8), Inches(1.9)
for i, (t, d) in enumerate(steps):
    r_, c_ = divmod(i, 4)
    x = x0 + c_ * (bw + gap)
    y = y0 + r_ * Inches(2.35)
    card = rect(s, x, y, bw, Inches(1.95), DARK if r_ == 0 else LIGHT)
    _, tf = textbox(s, x + Inches(0.18), y + Inches(0.14), bw - Inches(0.36), Inches(1.7))
    p = tf.paragraphs[0]
    set_run(p.add_run(), t, 17, RED if r_ else WHITE, bold=True)
    p = tf.add_paragraph()
    p.space_before = Pt(4)
    set_run(p.add_run(), d, 13, WHITE if r_ else DARK)
    if c_ < 3:
        arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + bw + Inches(0.02),
                                   y + Inches(0.78), Inches(0.26), Inches(0.38))
        arrow.fill.solid(); arrow.fill.fore_color.rgb = RED; arrow.line.fill.background()
_, tf = textbox(s, Inches(0.8), Inches(6.35), Inches(12), Inches(0.6))
set_run(tf.paragraphs[0].add_run(),
        "All artifacts versioned on disk: f1_model_*.joblib + preprocessors.joblib → loaded directly "
        "by the front end for inference.", 13, GRAY, italic=True)

# ======================================================================
# 8 - MODELS USED
# ======================================================================
s = content_slide("Models & Hyperparameters", 8)
tbl = s.shapes.add_table(4, 4, Inches(0.8), Inches(1.6), Inches(11.7), Inches(2.6)).table
tbl.columns[0].width, tbl.columns[1].width, tbl.columns[2].width, tbl.columns[3].width = \
    Inches(2.4), Inches(4.6), Inches(3.0), Inches(1.7)
fill_table(tbl, [
    ["Model", "Hyperparameters", "Imbalance handling", "Train time"],
    ["Random Forest", "200 trees, max_depth 15, random_state 42", "class_weight='balanced'", "0.23 s"],
    ["Gradient Boosting", "200 estimators, depth 5, lr 0.1", "n/a", "1.75 s"],
    ["XGBoost", "200 estimators, depth 6, lr 0.1, logloss", "scale_pos_weight = neg/pos", "2.06 s"],
])
style_table(tbl, font_size=13)
_, tf = textbox(s, Inches(0.8), Inches(4.55), Inches(11.7), Inches(2.2))
bullets(tf, [
    ("Why these three?", 0, True),
    ("Random Forest - robust bagging baseline, fast to train, low variance.", 1),
    ("Gradient Boosting - sequential boosting, strongest raw accuracy on tabular data.", 1),
    ("XGBoost - optimized gradient boosting, native handling of class weighting.", 1),
    ("All three share identical features and split → fair, like-for-like comparison.", 1),
], size=15, spacing=5)

# ======================================================================
# 9 - RESULTS TABLE
# ======================================================================
s = content_slide("Results - Test-Set Performance (1,651 samples)", 9)
tbl = s.shapes.add_table(4, 7, Inches(0.8), Inches(1.7), Inches(11.7), Inches(2.7)).table
tbl.columns[0].width = Inches(2.7)
for i, w in enumerate([1.6, 1.7, 1.5, 1.4, 1.6, 1.3], start=1):
    tbl.columns[i].width = Inches(w)
fill_table(tbl, [
    ["Model", "Accuracy", "Precision", "Recall", "F1 (podium)", "ROC-AUC", "Time"],
    ["XGBoost  ★ best F1", "0.8988", "0.6119", "0.8780", "0.7212", "0.9557", "2.06 s"],
    ["Gradient Boosting", "0.9140", "0.7000", "0.7398", "0.7194", "0.9570", "1.75 s"],
    ["Random Forest", "0.8964", "0.6068", "0.8659", "0.7136", "0.9568", "0.23 s"],
])
style_table(tbl, font_size=13)
_, tf = textbox(s, Inches(0.8), Inches(4.6), Inches(11.7), Inches(2.3))
bullets(tf, [
    ("Key observations", 0, True),
    ("All three models land within 0.8 F1 points of each other - the feature signal dominates the algorithm.", 1),
    ("XGBoost wins on F1 (0.7212) with the highest recall (0.878) - it catches the most podiums.", 1),
    ("Gradient Boosting is the most precise (0.700) - when it predicts podium, it is right 70% of the time.", 1),
    ("Random Forest trains ~8x faster - practical for frequent re-training.", 1),
], size=15, spacing=5)

# ======================================================================
# 10 - RESULTS CHART
# ======================================================================
s = content_slide("Results - Metric Comparison", 10)
s.shapes.add_picture(str(ROOT / "slide_model_comparison.png"), Inches(0.8), Inches(1.55), width=Inches(7.9))
rect(s, Inches(9.0), Inches(1.7), Inches(3.75), Inches(4.9), LIGHT)
_, tf = textbox(s, Inches(9.22), Inches(1.95), Inches(3.3), Inches(4.5))
p = tf.paragraphs[0]
set_run(p.add_run(), "Reading the chart", 16, RED, bold=True)
bullets(tf, [
    ("Accuracy ~0.90 everywhere - driven by the 85% majority class.", 0),
    ("Recall favors XGBoost & RF (class weights push toward the minority class).", 0),
    ("Precision favors Gradient Boosting (fewer false podium calls).", 0),
    ("ROC-AUC ≈ 0.956 for all - excellent ranking ability.", 0),
], size=13, spacing=8)

# ======================================================================
# 11 - FEATURE IMPORTANCE
# ======================================================================
s = content_slide("Feature Importance & Insights", 11)
s.shapes.add_picture(str(ROOT / "feature_importance.png"), Inches(0.8), Inches(1.6), width=Inches(6.6))
_, tf = textbox(s, Inches(7.8), Inches(1.7), Inches(4.9), Inches(5.0))
bullets(tf, [
    ("What drives a podium?", 0, True),
    ("Grid position is the dominant signal - where you start largely decides where you finish.", 1),
    ("Constructor identity matters more than driver nationality - the car is the biggest performance lever.", 1),
    ("Laps completed & result status capture race survival - mechanical reliability matters.", 1),
    ("", 0),
    ("Interpretation for the app", 0, True),
    ("Starting a 2025 driver P1 vs P12 swings the podium probability dramatically - visible live in the front end.", 1),
], size=15, spacing=6)

# ======================================================================
# 12 - DEPLOYMENT
# ======================================================================
s = content_slide("Deployment - Streamlit Front End (app.py)", 12)
_, tf = textbox(s, Inches(0.8), Inches(1.5), Inches(6.9), Inches(5.4))
bullets(tf, [
    ("Three tabs", 0, True),
    ("Single Driver - pick any 2025 driver, set grid slot, see podium probability from all 3 models.", 1),
    ("Full Grid Simulation - ranks all 20 drivers, highlights projected top-3.", 1),
    ("About - model docs + feature-importance chart.", 1),
    ("", 0),
    ("Engineering details", 0, True),
    ("Loads joblib models + LabelEncoders once, cached with @st.cache_resource.", 1),
    ("Maps 2025 team names to historical training labels (e.g. 'RB' → 'Toro Rosso' lineage).", 1),
    ("Safe fallback encoding for unseen labels - app never crashes on new data.", 1),
    ("Sidebar controls expose race assumptions (laps, status, fastest lap).", 1),
], size=15, spacing=6)
rect(s, Inches(8.0), Inches(1.7), Inches(4.75), Inches(2.4), DARK)
_, tf = textbox(s, Inches(8.25), Inches(1.9), Inches(4.3), Inches(2.0))
p = tf.paragraphs[0]
set_run(p.add_run(), "Run it", 15, RED, bold=True)
p = tf.add_paragraph()
set_run(p.add_run(), ".\\venv\\Scripts\\python.exe\n  -m streamlit run app.py", 14, WHITE, bold=True)
p = tf.add_paragraph()
set_run(p.add_run(), "→ opens at http://localhost:8501", 12, GRAY)
rect(s, Inches(8.0), Inches(4.35), Inches(4.75), Inches(2.35), LIGHT)
_, tf = textbox(s, Inches(8.25), Inches(4.55), Inches(4.3), Inches(2.0))
p = tf.paragraphs[0]
set_run(p.add_run(), "Stack", 15, RED, bold=True)
bullets(tf, [
    ("Python 3.12 virtual environment", 0),
    ("pandas, scikit-learn, xgboost, joblib", 0),
    ("Streamlit for the interactive UI", 0),
], size=13, spacing=5)

# ======================================================================
# 13 - LIMITATIONS & FUTURE WORK
# ======================================================================
s = content_slide("Limitations & Future Work", 13)
rect(s, Inches(0.8), Inches(1.55), Inches(5.9), Inches(5.15), LIGHT)
_, tf = textbox(s, Inches(1.05), Inches(1.75), Inches(5.4), Inches(4.8))
p = tf.paragraphs[0]
set_run(p.add_run(), "Limitations", 18, RED, bold=True)
bullets(tf, [
    ("Data leakage: laps, fastest-lap and status features exist only after the race - "
     "the model is explanatory, not a pre-race forecast.", 0),
    ("Random split ignores time - no true out-of-sample season validation yet.", 0),
    ("No qualifying times, weather, tyre or telemetry features.", 0),
    ("Label encoders are static - new teams/drivers need retraining.", 0),
    ("Precision ~0.6-0.7 → still 3-4 false podium calls in 10 predictions.", 0),
], size=14, spacing=7)
rect(s, Inches(7.0), Inches(1.55), Inches(5.9), Inches(5.15), LIGHT)
_, tf = textbox(s, Inches(7.25), Inches(1.75), Inches(5.4), Inches(4.8))
p = tf.paragraphs[0]
set_run(p.add_run(), "Roadmap", 18, GREEN, bold=True)
bullets(tf, [
    ("Leakage-free retrain using pre-race features: grid, rolling driver/constructor form, circuit history.", 0),
    ("Time-based backtesting: train ≤ year X, predict season X+1.", 0),
    ("Head-to-head and race-winner prediction modes.", 0),
    ("Automated pipeline to ingest new 2025 results and retrain on schedule.", 0),
    ("Probability calibration + richer app: charts, historical lookup, export.", 0),
], size=14, spacing=7)

# ======================================================================
# 14 - CONCLUSION
# ======================================================================
s = content_slide("Conclusion", 14)
_, tf = textbox(s, Inches(0.8), Inches(1.55), Inches(11.9), Inches(4.4))
bullets(tf, [
    ("Built a complete, reproducible ML system - from raw Ergast CSVs to a live interactive "
     "dashboard, with every model and encoder persisted for reuse.", 0, True),
    ("Trained and fairly compared three algorithms on identical data - all reach ~0.90 accuracy "
     "and ~0.96 ROC-AUC for podium classification.", 0, True),
    ("XGBoost delivers the best F1 (0.7212) and recall (0.878); Gradient Boosting the best "
     "precision (0.700) and accuracy (0.9140).", 0, True),
    ("Starting grid position and constructor identity are the strongest podium drivers - "
     "confirming F1 domain knowledge with data.", 0, True),
    ("The Streamlit front end turns the models into a usable product: single-driver forecasts "
     "and full-grid race simulations on the 2025 season.", 0, True),
    ("The project has a clear, honest path forward: removing post-race leakage to become a "
     "genuine pre-race prediction engine.", 0, True),
], size=17, spacing=12)
rect(s, Inches(0.8), Inches(6.15), Inches(11.9), Inches(0.75), DARK)
_, tf = textbox(s, Inches(1.05), Inches(6.28), Inches(11.4), Inches(0.5))
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
set_run(p.add_run(), "Final models: XGBoost (best F1)  •  Gradient Boosting (best precision)  •  "
                     "Random Forest (fastest)", 15, WHITE, bold=True)

# ======================================================================
# 15 - THANK YOU
# ======================================================================
s = add_slide()
rect(s, 0, 0, SW, SH, DARK)
rect(s, Inches(0.9), Inches(3.05), Inches(1.6), Inches(0.09), RED)
_, tf = textbox(s, Inches(0.9), Inches(3.25), Inches(11), Inches(1.6))
p = tf.paragraphs[0]
set_run(p.add_run(), "Thank You", 44, WHITE, bold=True)
p = tf.add_paragraph()
p.space_before = Pt(10)
set_run(p.add_run(), "Questions & Discussion", 20, GRAY)
_, tf = textbox(s, Inches(0.9), Inches(6.4), Inches(11), Inches(0.6))
set_run(tf.paragraphs[0].add_run(),
        "Project files: preprocess.py  •  train_models.py  •  compare_models.py  •  app.py  •  models/", 13, GRAY)

# ----------------------------------------------------------------------
out = ROOT / "F1_Podium_Prediction.pptx"
prs.save(out)
print(f"Saved -> {out} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
