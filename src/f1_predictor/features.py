"""Pairwise feature engineering for Driver-vs-Driver battles."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

import pandas as pd

RELATIVE_FEATURES = {
    "grid_diff": "qualifying_position",
    "driver_finish_diff": "driver_avg_finish",
    "driver_qual_diff": "driver_avg_qual",
    "driver_points_diff": "driver_avg_points",
    "driver_pos_gained_diff": "driver_avg_pos_gained",
    "driver_dnf_diff": "driver_dnf_rate",
    "driver_top5_diff": "driver_top5_rate",
    "team_finish_diff": "team_avg_finish",
    "team_qual_diff": "team_avg_qual",
    "team_points_diff": "team_avg_points",
    "team_dnf_diff": "team_dnf_rate",
    "driver_type_finish_diff": "driver_type_finish",
    "team_type_finish_diff": "team_type_finish",
    "experience_diff": "driver_races_prior",
}

MODEL_FEATURES = [
    *RELATIVE_FEATURES.keys(),
    "grid_x_street",
    "form_x_street",
    "grid_x_turns",
    "grid_x_drs",
]

FEATURE_LABELS = {
    "grid_diff": "Starting grid",
    "driver_finish_diff": "Driver recent finish form",
    "driver_qual_diff": "Driver qualifying form",
    "driver_points_diff": "Driver points form",
    "driver_pos_gained_diff": "Race-position gain tendency",
    "driver_dnf_diff": "Driver reliability",
    "driver_top5_diff": "Top-5 consistency",
    "team_finish_diff": "Constructor race form",
    "team_qual_diff": "Constructor qualifying form",
    "team_points_diff": "Constructor points form",
    "team_dnf_diff": "Constructor reliability",
    "driver_type_finish_diff": "Driver circuit-type form",
    "team_type_finish_diff": "Constructor circuit-type form",
    "experience_diff": "Season experience",
    "grid_x_street": "Grid × street-circuit context",
    "form_x_street": "Form × street-circuit context",
    "grid_x_turns": "Grid × corner-count context",
    "grid_x_drs": "Grid × DRS context",
}


def _pair_features(a: dict, b: dict) -> dict:
    row = {name: float(a[source] - b[source]) for name, source in RELATIVE_FEATURES.items()}
    street = float(a["street_flag"])
    row["grid_x_street"] = row["grid_diff"] * street
    row["form_x_street"] = row["driver_finish_diff"] * street
    row["grid_x_turns"] = row["grid_diff"] * float(a["number_of_turns"])
    row["grid_x_drs"] = row["grid_diff"] * float(a["drs_zones"])
    return row


def build_pairwise_training_data(driver_race: pd.DataFrame) -> pd.DataFrame:
    """Generate symmetric pairwise matchup examples from every race.

    Each unordered driver pair produces two rows (A vs B and B vs A). Races, not rows,
    should be used as the validation grouping boundary.
    """
    rows: list[dict] = []
    for race_id, race_df in driver_race.groupby("race_id", sort=True):
        records = race_df.to_dict("records")
        for left, right in combinations(records, 2):
            for a, b in ((left, right), (right, left)):
                record = {
                    "race_id": int(race_id),
                    "driver_a": a["driver_name"],
                    "driver_b": b["driver_name"],
                    "target": int(a["position"] < b["position"]),
                }
                record.update(_pair_features(a, b))
                rows.append(record)
    pairwise = pd.DataFrame(rows)
    if pairwise[MODEL_FEATURES].isna().any().any():
        raise ValueError("NaNs detected in pairwise model features")
    return pairwise


def build_latest_context(driver_race: pd.DataFrame) -> dict:
    """Build the state used for hypothetical post-season matchups.

    Unlike the training table, the deployment context intentionally summarizes all
    currently available races because it represents the information available *after*
    the supplied 2024 season snapshot.
    """
    df = driver_race.copy()

    driver = (
        df.groupby(["driver_name", "team"], as_index=False)
        .agg(
            qualifying_position=("qualifying_position", "mean"),
            driver_avg_finish=("position", "mean"),
            driver_avg_qual=("qualifying_position", "mean"),
            driver_avg_points=("points", "mean"),
            driver_avg_pos_gained=("positions_gained", "mean"),
            driver_dnf_rate=("dnf", "mean"),
            driver_top5_rate=("top5", "mean"),
            driver_races_prior=("race_id", "nunique"),
        )
        .sort_values("driver_name")
    )

    # If a driver ever changed team in the supplied snapshot, use their most recent team.
    latest_team = (
        df.sort_values("race_id")
        .groupby("driver_name", as_index=False)
        .tail(1)[["driver_name", "team"]]
        .set_index("driver_name")["team"]
        .to_dict()
    )
    driver["team"] = driver["driver_name"].map(latest_team)
    driver = driver.drop_duplicates("driver_name", keep="last")

    driver_type = (
        df.groupby(["driver_name", "circuit_type"], as_index=False)
        .agg(driver_type_finish=("position", "mean"))
    )
    team = (
        df.groupby("team", as_index=False)
        .agg(
            team_avg_finish=("position", "mean"),
            team_avg_qual=("qualifying_position", "mean"),
            team_avg_points=("points", "mean"),
            team_dnf_rate=("dnf", "mean"),
        )
    )
    team_type = (
        df.groupby(["team", "circuit_type"], as_index=False)
        .agg(team_type_finish=("position", "mean"))
    )
    circuits = (
        df[
            [
                "circuit",
                "city",
                "country",
                "circuit_type",
                "street_flag",
                "circuit_length_km",
                "number_of_turns",
                "drs_zones",
                "elevation_change_m",
                "race_distance_km",
                "total_laps",
            ]
        ]
        .drop_duplicates("circuit")
        .sort_values("circuit")
        .reset_index(drop=True)
    )
    return {
        "driver": driver,
        "driver_type": driver_type,
        "team": team,
        "team_type": team_type,
        "circuits": circuits,
    }


def build_scenario_features(
    context: dict,
    driver_a: str,
    driver_b: str,
    circuit: str,
    grid_a: int,
    grid_b: int,
) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Create the one-row model matrix for an interactive matchup."""
    if driver_a == driver_b:
        raise ValueError("Choose two different drivers")
    if not 1 <= int(grid_a) <= 20 or not 1 <= int(grid_b) <= 20:
        raise ValueError("Grid positions must be between 1 and 20")

    driver_df = context["driver"]
    circuit_df = context["circuits"]
    if driver_a not in set(driver_df["driver_name"]):
        raise ValueError(f"Unknown driver: {driver_a}")
    if driver_b not in set(driver_df["driver_name"]):
        raise ValueError(f"Unknown driver: {driver_b}")
    if circuit not in set(circuit_df["circuit"]):
        raise ValueError(f"Unknown circuit: {circuit}")

    a = driver_df.loc[driver_df["driver_name"].eq(driver_a)].iloc[0].to_dict()
    b = driver_df.loc[driver_df["driver_name"].eq(driver_b)].iloc[0].to_dict()
    c = circuit_df.loc[circuit_df["circuit"].eq(circuit)].iloc[0].to_dict()
    a["qualifying_position"] = float(grid_a)
    b["qualifying_position"] = float(grid_b)

    for record in (a, b):
        driver_type = context["driver_type"]
        mask = driver_type["driver_name"].eq(record["driver_name"]) & driver_type[
            "circuit_type"
        ].eq(c["circuit_type"])
        record["driver_type_finish"] = (
            float(driver_type.loc[mask, "driver_type_finish"].iloc[0]) if mask.any() else 10.5
        )

        team_df = context["team"]
        team_row = team_df.loc[team_df["team"].eq(record["team"])]
        if team_row.empty:
            record.update(
                team_avg_finish=10.5,
                team_avg_qual=10.5,
                team_avg_points=0.0,
                team_dnf_rate=0.05,
            )
        else:
            record.update(team_row.iloc[0].to_dict())

        team_type = context["team_type"]
        tmask = team_type["team"].eq(record["team"]) & team_type["circuit_type"].eq(
            c["circuit_type"]
        )
        record["team_type_finish"] = (
            float(team_type.loc[tmask, "team_type_finish"].iloc[0]) if tmask.any() else 10.5
        )

        record.update(c)

    features = _pair_features(a, b)
    X = pd.DataFrame([features], columns=MODEL_FEATURES)
    return X, a, b, c
