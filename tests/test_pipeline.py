from __future__ import annotations

import numpy as np

from f1_predictor.features import (
    MODEL_FEATURES,
    build_latest_context,
    build_pairwise_training_data,
    build_scenario_features,
)
from f1_predictor.model import load_model, predict_probability
from f1_predictor.preprocessing import build_driver_race_features


def _runtime():
    driver_race = build_driver_race_features()
    context = build_latest_context(driver_race)
    model = load_model()
    return driver_race, context, model


def test_engineered_training_data_is_complete():
    driver_race = build_driver_race_features()
    pairwise = build_pairwise_training_data(driver_race)
    assert len(pairwise) > 8_000
    assert not pairwise[MODEL_FEATURES].isna().any().any()
    assert set(pairwise["target"].unique()) == {0, 1}


def test_pairwise_prediction_is_symmetric():
    _, context, model = _runtime()
    X_ab, *_ = build_scenario_features(
        context, "Lewis Hamilton", "George Russell", "Silverstone Circuit", 3, 5
    )
    X_ba, *_ = build_scenario_features(
        context, "George Russell", "Lewis Hamilton", "Silverstone Circuit", 5, 3
    )
    p_ab = predict_probability(model, X_ab)
    p_ba = predict_probability(model, X_ba)
    assert np.isclose(p_ab + p_ba, 1.0, atol=1e-6)


def test_grid_position_changes_probability_in_expected_direction():
    _, context, model = _runtime()
    X_front, *_ = build_scenario_features(
        context, "Lewis Hamilton", "George Russell", "Silverstone Circuit", 1, 5
    )
    X_back, *_ = build_scenario_features(
        context, "Lewis Hamilton", "George Russell", "Silverstone Circuit", 20, 5
    )
    assert predict_probability(model, X_front) > predict_probability(model, X_back)
