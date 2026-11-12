"""Display the canonical chronological backtest; run train_models.py to refresh it."""
import json
from pathlib import Path

import pandas as pd

from prediction import MODEL_LABELS


def main() -> None:
    report_path = Path(__file__).parent / "models" / "metrics.json"
    if not report_path.exists():
        raise SystemExit("No backtest report. Run python train_models.py first.")
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != 2:
        raise SystemExit("Legacy retrospective report. Run python train_models.py to upgrade.")
    labels = {**MODEL_LABELS, "model_average": "Model average", "prior_rate": "Prior-rate baseline",
              "grid_top3": "Grid top-three baseline"}
    print(f"Chronological test seasons: {report['backtest_years']}")
    print(pd.DataFrame(report["models"]).T.rename(index=labels).round(4).to_string())
    print("F1 uses threshold 0.5. Brier/log loss: lower is better. Constant prior has no ranking score.")


if __name__ == "__main__":
    main()
