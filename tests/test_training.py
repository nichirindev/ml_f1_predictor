"""Verify temporal splits, calibration isolation and ranking metrics."""
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from features import FEATURES, build_training_table
from test_features import sample_history
from train_models import evaluate, fit_calibrated, temporal_split


class TrainingTests(unittest.TestCase):
    def test_training_calibration_and_test_are_strictly_separated(self):
        table, _ = build_training_table(sample_history())
        train, calibration, test = temporal_split(table, 2022)
        self.assertLess(train.date.max(), calibration.date.min())
        self.assertLess(calibration.date.max(), test.date.min())
        self.assertEqual(set(train.year), {2020})
        self.assertEqual(set(calibration.year), {2021})
        self.assertEqual(set(test.year), {2022})
        with self.assertRaises(ValueError):
            temporal_split(table, 2021)

    def test_calibration_does_not_refit_base_model(self):
        table, _ = build_training_table(sample_history())
        train, calibration, test = temporal_split(table, 2022)
        with patch("train_models.get_models", return_value={"random_forest": DecisionTreeClassifier(max_depth=2, random_state=42)}):
            models, importance = fit_calibrated(train, calibration)
        model = models["random_forest"]
        frozen = model.calibrated_classifiers_[0].estimator
        self.assertEqual(frozen.tree_.n_node_samples[0], len(train))
        probabilities = model.predict_proba(test[FEATURES])
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertEqual(len(importance["Random Forest"]), len(FEATURES))

    def test_metrics_keep_probability_and_ranking_quality_separate(self):
        table, _ = build_training_table(sample_history())
        metrics = evaluate(table, np.where(table.top3, .9, .1))
        self.assertEqual(metrics["f1"], 1)
        self.assertEqual(metrics["top3_recall"], 1)
        self.assertAlmostEqual(metrics["brier"], .01)


if __name__ == "__main__":
    unittest.main()
