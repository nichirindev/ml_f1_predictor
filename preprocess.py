"""Export feature importance from the canonical pre-race training bundle."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from prediction import validate_bundle


def main() -> None:
    root = Path(__file__).parent
    bundle_path = root / "models" / "prerace_bundle.joblib"
    if not bundle_path.exists():
        raise SystemExit("Run python train_models.py to build the pre-race models first.")
    bundle = joblib.load(bundle_path)
    validate_bundle(bundle)
    importance = pd.Series(bundle["feature_importance"]["Random Forest"], index=bundle["features"]).sort_values()
    _, axis = plt.subplots(figsize=(10, 7))
    importance.plot.barh(ax=axis, color="#d94a43")
    axis.set(title="Pre-race feature importance · Random Forest", xlabel="Importance", ylabel="")
    plt.tight_layout()
    plt.savefig(root / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved feature_importance.png from the deployed pre-race base estimator.")


if __name__ == "__main__":
    main()
