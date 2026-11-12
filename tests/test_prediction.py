"""Pre-race inference validation and model scoring tests."""
import unittest

import numpy as np
import pandas as pd

from features import FEATURES, SCHEMA_VERSION, build_training_table
from prediction import build_features, predict_models, select_scores, validate_bundle, validate_grid
from test_features import sample_history


class PredictionTests(unittest.TestCase):
    def test_grid_requires_a_permutation(self):
        validate_grid(pd.DataFrame({"Grid": [3, 1, 2]}))
        for values in ([1, 1, 3], [0, 2, 3], [1, None, 3], [1, 2.5, 3], [1, np.inf, 3]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_grid(pd.DataFrame({"Grid": values}))

    def test_snapshot_features_dates_and_cold_start_notes(self):
        history = sample_history()
        _, state = build_training_table(history)
        prep = {"state": state, "trained_through": "2022-03-08", "circuits": pd.DataFrame({"circuitId": [1, 2]})}
        drivers = pd.DataFrame([{"driver_key": "rookie", "driver_name": "Rookie", "constructorId": 10,
                                 "historical_team": "Team", "team_name": "Team Racing", "Grid": 2}])
        assumptions = {"race_date": "2023-03-01", "round": 1, "circuitId": 2}
        features, notes = build_features(drivers, prep, assumptions)
        self.assertEqual(features.columns.tolist(), FEATURES)
        self.assertEqual(features.iloc[0].grid, 2)
        self.assertEqual(features.iloc[0].driver_experience, 0)
        self.assertTrue(any("rookie priors" in note for note in notes))
        self.assertTrue(any("Team Racing → Team" in note for note in notes))
        for bad in ("2022-03-08", "2020-01-01", None):
            with self.subTest(date=bad), self.assertRaises(ValueError):
                build_features(drivers, prep, {**assumptions, "race_date": bad})
        for bad in (-1, 1.5, np.nan, 25):
            with self.subTest(grid=bad), self.assertRaises(ValueError):
                build_features(drivers.assign(Grid=bad), prep, assumptions)
        with self.assertRaisesRegex(ValueError, "circuit"):
            build_features(drivers, prep, {**assumptions, "circuitId": 999})
        with self.assertRaises(ValueError):
            build_features(drivers, prep, {**assumptions, "round": 1.5})

    def test_legacy_artifacts_are_rejected(self):
        for bundle in ({}, {"schema_version": 1}, {"schema_version": SCHEMA_VERSION, "features": ["laps"]}):
            with self.subTest(bundle=bundle), self.assertRaises(ValueError):
                validate_bundle(bundle)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_bundle({"schema_version": SCHEMA_VERSION, "features": FEATURES, "models": {}})

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
