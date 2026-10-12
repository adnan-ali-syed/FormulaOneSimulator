"""Data loading and leakage-safe pre-race feature construction."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

REQUIRED_FILES = {
    "race": "f1_2024_race_results.csv",
    "qualifying": "f1_qualifying_results_2024.csv",
    "circuits": "f1_circuits_metadata.csv",
    "driver_standings": "f1_2024_driver_standings.csv",
    "constructor_standings": "f1_2024_constructor_standings.csv",
}


def load_raw_data(data_dir: Path | str = DEFAULT_RAW_DIR) -> Dict[str, pd.DataFrame]:
    """Load the datasets used by Grid Battle.

    The function fails loudly when a required file is missing so deployment problems are
    caught immediately rather than surfaced later as cryptic model errors.
    """
    data_dir = Path(data_dir)
    frames: Dict[str, pd.DataFrame] = {}
    for key, filename in REQUIRED_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found: {path}")
        frames[key] = pd.read_csv(path)
    return frames


def _shifted_expanding_mean(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    default: float,
) -> pd.Series:
    """Mean of all *previous* rows within a group.

    For driver-level features there is one row per driver per race, so this is
    leakage-safe once the frame is sorted by race_id.
    """
    out = (
        df.groupby(group_cols, sort=False)[value_col]
        .transform(lambda s: s.shift(1).expanding().mean())
        .astype(float)
    )
    return out.fillna(float(default))


def build_driver_race_features(
    frames: Dict[str, pd.DataFrame] | None = None,
    data_dir: Path | str = DEFAULT_RAW_DIR,
) -> pd.DataFrame:
    """Create one leakage-safe feature row per driver/race.

    Features describing form and constructor performance use only races that occurred
    before the current race. The actual finishing position is retained solely as the
    training target and is never included in inference features.
    """
    if frames is None:
        frames = load_raw_data(data_dir)

    race = frames["race"].copy()
    qualifying = frames["qualifying"].copy()
    circuits = frames["circuits"].copy()

    qualifying_cols = [
        "race_id",
        "driver_name",
        "qualifying_position",
        "gap_to_pole",
        "best_time",
    ]
    df = race.merge(
        qualifying[qualifying_cols],
        on=["race_id", "driver_name"],
        how="inner",
        validate="one_to_one",
    )
    df = df.merge(
        circuits,
        left_on="circuit",
        right_on="circuit_name",
        how="left",
        validate="many_to_one",
        suffixes=("", "_metadata"),
    )

    df["dnf"] = df["race_time"].astype(str).str.upper().eq("DNF").astype(int)
    df["positions_gained"] = df["qualifying_position"] - df["position"]
    df["top5"] = (df["position"] <= 5).astype(int)
    df["street_flag"] = (
        df["circuit_type"].astype(str).str.contains("street", case=False, na=False).astype(int)
    )
    df = df.sort_values(["race_id", "driver_name"]).reset_index(drop=True)

    # Neutral priors for the first race. They are fixed constants rather than values
    # calculated from future races.
    neutral_finish = 10.5
    neutral_qual = 10.5
    neutral_points = 0.0
    neutral_positions_gained = 0.0
    neutral_dnf = 0.05
    neutral_top5 = 0.25

    # Driver history: one driver observation per race, so shifted expanding means are safe.
    df["driver_avg_finish"] = _shifted_expanding_mean(
        df, ["driver_name"], "position", neutral_finish
    )
    df["driver_avg_qual"] = _shifted_expanding_mean(
        df, ["driver_name"], "qualifying_position", neutral_qual
    )
    df["driver_avg_points"] = _shifted_expanding_mean(
        df, ["driver_name"], "points", neutral_points
    )
    df["driver_avg_pos_gained"] = _shifted_expanding_mean(
        df, ["driver_name"], "positions_gained", neutral_positions_gained
    )
    df["driver_dnf_rate"] = _shifted_expanding_mean(
        df, ["driver_name"], "dnf", neutral_dnf
    )
    df["driver_top5_rate"] = _shifted_expanding_mean(
        df, ["driver_name"], "top5", neutral_top5
    )
    df["driver_type_finish"] = _shifted_expanding_mean(
        df, ["driver_name", "circuit_type"], "position", neutral_finish
    )
    df["driver_races_prior"] = df.groupby("driver_name", sort=False).cumcount().astype(float)

    # Team features need a race-level aggregation first. Otherwise the second team driver
    # in a race would accidentally see the first driver's current-race result.
    team_race = (
        df.groupby(["race_id", "team", "circuit_type"], as_index=False)
        .agg(
            team_race_finish=("position", "mean"),
            team_race_qual=("qualifying_position", "mean"),
            team_race_points=("points", "mean"),
            team_race_dnf=("dnf", "mean"),
        )
        .sort_values(["race_id", "team"])
    )
    team_race["team_avg_finish"] = _shifted_expanding_mean(
        team_race, ["team"], "team_race_finish", neutral_finish
    )
    team_race["team_avg_qual"] = _shifted_expanding_mean(
        team_race, ["team"], "team_race_qual", neutral_qual
    )
    team_race["team_avg_points"] = _shifted_expanding_mean(
        team_race, ["team"], "team_race_points", neutral_points
    )
    team_race["team_dnf_rate"] = _shifted_expanding_mean(
        team_race, ["team"], "team_race_dnf", neutral_dnf
    )
    team_race["team_type_finish"] = _shifted_expanding_mean(
        team_race, ["team", "circuit_type"], "team_race_finish", neutral_finish
    )

    df = df.merge(
        team_race[
            [
                "race_id",
                "team",
                "team_avg_finish",
                "team_avg_qual",
                "team_avg_points",
                "team_dnf_rate",
                "team_type_finish",
            ]
        ],
        on=["race_id", "team"],
        how="left",
        validate="many_to_one",
    )

    numeric_required = [
        "qualifying_position",
        "driver_avg_finish",
        "driver_avg_qual",
        "driver_avg_points",
        "driver_avg_pos_gained",
        "driver_dnf_rate",
        "driver_top5_rate",
        "team_avg_finish",
        "team_avg_qual",
        "team_avg_points",
        "team_dnf_rate",
        "driver_type_finish",
        "team_type_finish",
        "driver_races_prior",
        "circuit_length_km",
        "number_of_turns",
        "drs_zones",
        "elevation_change_m",
    ]
    if df[numeric_required].isna().any().any():
        bad = df[numeric_required].columns[df[numeric_required].isna().any()].tolist()
        raise ValueError(f"Unexpected missing values in engineered columns: {bad}")

    return df.sort_values(["race_id", "position"]).reset_index(drop=True)
