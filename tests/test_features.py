"""Adversarial chronology tests: current/future outcomes cannot enter pre-race inputs."""
from pathlib import Path
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from features import FEATURES, build_training_table, driver_key, load_history, prepare_roster

ROOT = Path(__file__).resolve().parents[1]


def sample_history():
    rows = []
    for race_id, day in enumerate(["2020-03-01", "2020-03-08", "2021-03-01", "2021-03-08", "2022-03-01", "2022-03-08"], 1):
        for driver, team, grid, finish in [("a", 10, 1, 1), ("b", 10, 2, 4), ("c", 20, 3, 2),
                                          ("d", 20, 4, 3), ("e", 30, 5, 5), ("f", 30, 6, 6)]:
            rows.append({"raceId": race_id, "date": pd.Timestamp(day), "year": int(day[:4]),
                         "round": 1 + (race_id - 1) % 2, "circuitId": 1, "driver_key": driver,
                         "constructorId": team, "grid": grid, "finish": finish, "top3": int(finish <= 3)})
    return pd.DataFrame(rows)


class FeatureTests(unittest.TestCase):
    def test_current_outcomes_and_teammates_do_not_change_current_features(self):
        history = sample_history()
        before, _ = build_training_table(history)
        changed = history.copy()
        changed.loc[changed.raceId == 3, ["finish", "top3"]] = [20, 0]
        after, _ = build_training_table(changed)
        mask = before.raceId <= 3
        assert_frame_equal(before.loc[mask, FEATURES], after.loc[mask, FEATURES])
        self.assertNotEqual(before.loc[before.raceId == 4, "team_recent_finish"].iloc[0],
                            after.loc[after.raceId == 4, "team_recent_finish"].iloc[0])
        shuffled, _ = build_training_table(history.sample(frac=1, random_state=4))
        key = ["raceId", "driver_key"]
        assert_frame_equal(before.set_index(key)[FEATURES].sort_index(), shuffled.set_index(key)[FEATURES].sort_index())

    def test_future_results_do_not_change_the_past(self):
        history = sample_history()
        full, _ = build_training_table(history)
        prefix, _ = build_training_table(history.loc[history.raceId < 5])
        key = ["raceId", "driver_key"]
        assert_frame_equal(full.loc[full.raceId < 5].set_index(key)[FEATURES], prefix.set_index(key)[FEATURES])

    def test_same_date_races_do_not_observe_each_other(self):
        history = sample_history()
        history.loc[history.raceId == 2, "date"] = history.date.min()
        table, _ = build_training_table(history)
        first = table.loc[table.raceId == 1, FEATURES].reset_index(drop=True)
        second = table.loc[table.raceId == 2, FEATURES].reset_index(drop=True)
        first["round"] = second["round"]
        assert_frame_equal(first, second)

    def test_snapshot_inference_matches_training_before_each_race(self):
        history = sample_history()
        table, _ = build_training_table(history)
        for day, race in history.groupby("date"):
            _, snapshot = build_training_table(history.loc[history.date < day])
            actual = pd.DataFrame([snapshot.features(entry) for entry in race.to_dict("records")], columns=FEATURES)
            expected = table.loc[table.date == day, FEATURES].reset_index(drop=True)
            assert_frame_equal(actual, expected)

    def test_windows_rookie_defaults_and_no_future_derived_columns(self):
        history = sample_history()
        table, snapshot = build_training_table(history)
        self.assertEqual(snapshot.features(history.iloc[0].to_dict())["driver_history_races"], 5)
        self.assertEqual(snapshot.features(history.iloc[0].to_dict())["driver_circuit_races"], 3)
        rookie = {**history.iloc[0].to_dict(), "driver_key": "rookie", "grid": 0}
        row = snapshot.features(rookie)
        self.assertEqual(row["driver_history_races"], 0)
        self.assertEqual(row["driver_recent_finish"], 15)
        self.assertEqual(row["driver_recent_podium_rate"], .1)
        self.assertEqual(row["grid"], 25)
        self.assertEqual(row["pit_lane_start"], 1)
        self.assertGreater(row["team_history_races"], 0)
        self.assertTrue(table[FEATURES].notna().all().all())
        self.assertTrue(set(FEATURES).isdisjoint({"top3", "finish", "laps", "fastestLap", "rank", "fastestLapSpeed", "statusId", "position", "points"}))

    def test_raw_loader_keeps_dnfs_and_resolves_real_roster_names(self):
        history = load_history(ROOT / "data")
        raw = pd.read_csv(ROOT / "data" / "results.csv")
        races = pd.read_csv(ROOT / "data" / "races.csv")
        self.assertEqual(len(history), len(raw.loc[raw.raceId.isin(races.loc[races.year >= 2000, "raceId"])]))
        self.assertEqual(history.loc[history.top3 == 1].groupby("raceId").size().min(), 3)
        roster = prepare_roster(ROOT / "data")
        nico = roster.loc[roster.driver_name == "Nico Hulkenberg"].iloc[0]
        self.assertIn(nico.driver_key, history.driver_key.values)
        self.assertEqual(nico.constructorId, 15)
        self.assertEqual(driver_key("Nico Hülkenberg"), driver_key("Nico Hulkenberg"))


if __name__ == "__main__":
    unittest.main()
