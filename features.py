"""Pre-race features built from strictly earlier Grands Prix."""
from collections import defaultdict, deque
from pathlib import Path
import unicodedata

import pandas as pd

SCHEMA_VERSION = 2
START_YEAR = 2000
FEATURES = [
    "grid", "pit_lane_start", "round",
    "driver_recent_finish", "driver_recent_podium_rate", "driver_history_races", "driver_experience",
    "team_recent_finish", "team_recent_podium_rate", "team_history_races",
    "driver_circuit_finish", "driver_circuit_podium_rate", "driver_circuit_races",
    "team_circuit_finish", "team_circuit_podium_rate",
]
# Fixed cold-start priors; never estimated from a future season or the full dataset.
DEFAULT_FINISH = 15.0
DEFAULT_PODIUM_RATE = 0.10
TEAM_ALIASES = {
    "Red Bull Racing": "Red Bull", "Alpine": "Alpine F1 Team",
    "Kick Sauber": "Sauber", "RB": "RB F1 Team",
}


def driver_key(name: str) -> str:
    """Match names independently of accents, punctuation and roster-local IDs."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return "".join(char for char in plain.lower() if char.isalnum())


def load_history(data_dir: Path) -> pd.DataFrame:
    """Load modern-era race results without dropping DNFs or missing lap telemetry."""
    results = pd.read_csv(data_dir / "results.csv")
    races = pd.read_csv(data_dir / "races.csv")
    drivers = pd.read_csv(data_dir / "drivers.csv")
    drivers["driver_key"] = (drivers.forename + " " + drivers.surname).map(driver_key)
    history = results.merge(races[["raceId", "year", "round", "circuitId", "date"]],
                            on="raceId", validate="many_to_one")
    history = history.loc[history.year >= START_YEAR].merge(
        drivers[["driverId", "driver_key"]], on="driverId", validate="many_to_one")
    history["date"] = pd.to_datetime(history.date, errors="raise")
    history["finish"] = pd.to_numeric(history.positionOrder, errors="raise")
    history["top3"] = pd.to_numeric(history.position, errors="coerce").le(3).astype(int)
    history["grid"] = pd.to_numeric(history.grid, errors="raise")
    columns = ["raceId", "year", "round", "circuitId", "date", "driver_key", "constructorId", "grid", "finish", "top3"]
    history = history[columns].sort_values(["date", "raceId", "driver_key"]).reset_index(drop=True)
    if history.empty or history[columns].isna().any().any():
        raise ValueError("Historical race entries contain missing required pre-race metadata or results.")
    if history.duplicated(["raceId", "driver_key"]).any():
        raise ValueError("Duplicate driver entries in a Grand Prix; resolve them before training.")
    return history


def prepare_roster(data_dir: Path) -> pd.DataFrame:
    """Resolve the bundled roster's synthetic IDs using names and explicit team aliases."""
    roster = pd.read_csv(data_dir / "f1_2025_grid.csv")
    constructors = pd.read_csv(data_dir / "constructors.csv").set_index("name").constructorId
    roster["driver_key"] = roster.driver_name.map(driver_key)
    roster["historical_team"] = roster.team_name.map(lambda name: TEAM_ALIASES.get(name, name))
    roster["constructorId"] = roster.historical_team.map(constructors)
    if roster.constructorId.isna().any() or roster.driver_key.duplicated().any():
        raise ValueError("Roster needs unique driver names and recognized team mappings.")
    roster["constructorId"] = roster.constructorId.astype(int)
    return roster


def _summary(history: deque | None) -> tuple[float, float, int]:
    if not history:
        return DEFAULT_FINISH, DEFAULT_PODIUM_RATE, 0
    return (sum(item[0] for item in history) / len(history),
            sum(item[1] for item in history) / len(history), len(history))


class RaceHistory:
    """Rolling state updated only after all entries in an event have been scored."""

    def __init__(self) -> None:
        self.drivers = {}
        self.teams = {}
        self.driver_circuits = {}
        self.team_circuits = {}
        self.experience = defaultdict(int)

    def features(self, entry: dict) -> dict:
        """Read a snapshot without mutating it, including safe rookie defaults."""
        driver, team, circuit = entry["driver_key"], entry["constructorId"], entry["circuitId"]
        d_finish, d_podium, d_count = _summary(self.drivers.get(driver))
        t_finish, t_podium, t_count = _summary(self.teams.get(team))
        dc_finish, dc_podium, dc_count = _summary(self.driver_circuits.get((driver, circuit)))
        tc_finish, tc_podium, _ = _summary(self.team_circuits.get((team, circuit)))
        return {
            "grid": entry["grid"] if entry["grid"] > 0 else 25,
            "pit_lane_start": int(entry["grid"] == 0), "round": entry["round"],
            "driver_recent_finish": d_finish, "driver_recent_podium_rate": d_podium,
            "driver_history_races": d_count, "driver_experience": min(self.experience.get(driver, 0), 100),
            "team_recent_finish": t_finish, "team_recent_podium_rate": t_podium, "team_history_races": t_count,
            "driver_circuit_finish": dc_finish, "driver_circuit_podium_rate": dc_podium,
            "driver_circuit_races": dc_count, "team_circuit_finish": tc_finish, "team_circuit_podium_rate": tc_podium,
        }

    def update(self, race: pd.DataFrame) -> None:
        """Add one completed race; constructor form uses race-level teammate averages."""
        for row in race.to_dict("records"):
            driver = row["driver_key"]
            outcome = (row["finish"], row["top3"])
            self.drivers.setdefault(driver, deque(maxlen=5)).append(outcome)
            self.driver_circuits.setdefault((driver, row["circuitId"]), deque(maxlen=3)).append(outcome)
            self.experience[driver] += 1
        for (team, circuit), entries in race.groupby(["constructorId", "circuitId"]):
            outcome = (float(entries.finish.mean()), float(entries.top3.mean()))
            self.teams.setdefault(team, deque(maxlen=5)).append(outcome)
            self.team_circuits.setdefault((team, circuit), deque(maxlen=3)).append(outcome)


def build_training_table(history: pd.DataFrame) -> tuple[pd.DataFrame, RaceHistory]:
    """Score each race before updating state; even same-date events cannot see each other."""
    state, rows = RaceHistory(), []
    for _, day in history.sort_values(["date", "raceId"]).groupby("date", sort=True):
        for entry in day.to_dict("records"):
            rows.append({**entry, **state.features(entry)})
        for _, race in day.groupby("raceId", sort=False):
            state.update(race)
    return pd.DataFrame(rows), state
