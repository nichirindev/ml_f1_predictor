"""Streamlit smoke tests using locally trained artifacts."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "models" / "prerace_bundle.joblib").exists(), "Train models first")
class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error)

    def test_model_and_scenario_controls(self):
        app = self.app
        initial = app.metric[0].value
        model = next(s for s in app.selectbox if s.label == "Scoring model")
        model.select("XGBoost").run()
        self.assertFalse(app.exception)
        self.assertNotEqual(initial, app.metric[0].value)
        next(n for n in app.number_input if n.label == "Starting position").set_value(20).run()
        self.assertFalse(app.exception)
        self.assertNotEqual(initial, app.metric[0].value)
        labels = [slider.label for slider in app.slider]
        self.assertNotIn("Laps completed", labels)
        self.assertNotIn("Result status", [s.label for s in app.selectbox])
        next(n for n in app.number_input if n.label == "Race round").set_value(5).run()
        self.assertFalse(app.exception)
        next(n for n in app.number_input if n.label == "Starting position").set_value(0).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Pit lane" in h.value for h in app.subheader))

    def test_rookie_and_circuit_context(self):
        app = self.app
        next(s for s in app.selectbox if s.label == "Circuit").set_value(9).run()
        next(s for s in app.selectbox if s.label == "Driver · 2025 roster").select("Andrea Kimi Antonelli").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any("Silverstone" in caption.value for caption in app.caption))
        self.assertTrue(any("rookie priors" in caption.value for caption in app.caption))
        inputs = next(frame.value for frame in app.dataframe if "Feature value" in frame.value.columns)
        self.assertEqual(inputs.loc["driver_history_races", "Feature value"], 0)
        self.assertEqual(inputs.loc["driver_circuit_races", "Feature value"], 0)

    def test_comparison_and_invalid_grid_preserve_other_views(self):
        app = self.app
        app.selectbox(key="driver_a").select("Charles Leclerc").run()
        self.assertNotEqual(app.selectbox(key="driver_a").value, app.selectbox(key="driver_b").value)
        app.session_state["starting_grid"] = {
            "edited_rows": {0: {"Grid": 2}}, "added_rows": [], "deleted_rows": []}
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("exactly once" in e.value for e in app.error))
        self.assertEqual(len(app.metric), 5)
        app.session_state["starting_grid"] = {
            "edited_rows": {0: {"Grid": 2}, 1: {"Grid": 1}}, "added_rows": [], "deleted_rows": []}
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)


if __name__ == "__main__":
    unittest.main()
