"""Streamlit smoke tests using locally trained artifacts."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "models" / "f1_model_xgboost.joblib").exists(), "Train models first")
class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(self.app.exception)

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
        next(s for s in app.slider if s.label == "Laps completed").set_value(10).run()
        fastest = next(s for s in app.slider if s.label == "Fastest lap set on lap")
        self.assertLessEqual(fastest.value, 10)
        self.assertFalse(app.exception)

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
