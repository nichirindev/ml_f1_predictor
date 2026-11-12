"""Run with python -m unittest discover -s tests."""
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from prediction import build_features, predict_models, select_scores, validate_grid


class PredictionTests(unittest.TestCase):
    def test_grid_requires_a_permutation(self):
        validate_grid(pd.DataFrame({"Grid": [3, 1, 2]}))
        for values in ([1, 1, 3], [0, 2, 3], [1, None, 3], [1, 2.5, 3], [1, np.inf, 3]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_grid(pd.DataFrame({"Grid": values}))

    def test_features_follow_artifact_order_and_report_aliases(self):
        prep = {"features": ["grid", "year", "constructor_encoded", "nationality_encoded"],
                "le_nat": LabelEncoder().fit(["Dutch"]),
                "le_con": LabelEncoder().fit(["Ferrari", "Red Bull"])}
        drivers = pd.DataFrame([{"nationality": "Dutch", "team_name": "Red Bull Racing", "Grid": 2}])
        features, aliases = build_features(drivers, prep, {"year": 2025})
        self.assertEqual(features.columns.tolist(), prep["features"])
        self.assertEqual(features.iloc[0].tolist(), [2, 2025, 1, 0])
        self.assertEqual(aliases, ["Red Bull Racing → Red Bull"])
        drivers.loc[0, "team_name"] = "Unknown Team"
        with self.assertRaisesRegex(ValueError, "Unknown Team"):
            build_features(drivers, prep, {"year": 2025})
        drivers.loc[0, "team_name"] = "Ferrari"
        drivers.loc[0, "nationality"] = "Unknown nationality"
        with self.assertRaisesRegex(ValueError, "Unknown nationality"):
            build_features(drivers, prep, {"year": 2025})

    def test_batch_scoring_uses_average_instead_of_optimistic_maximum(self):
        class Model:
            def __init__(self, scores):
                self.scores = np.array(scores)

            def predict_proba(self, features):
                assert len(features) == len(self.scores)
                return np.column_stack([1 - self.scores, self.scores])

        scores = predict_models({"random_forest": Model([.9, .2]),
                                 "xgboost": Model([.3, .6])}, pd.DataFrame({"grid": [1, 2]}))
        np.testing.assert_allclose(select_scores(scores, "Model average"), [.6, .4])
        np.testing.assert_allclose(select_scores(scores, "XGBoost"), [.3, .6])


if __name__ == "__main__":
    unittest.main()
