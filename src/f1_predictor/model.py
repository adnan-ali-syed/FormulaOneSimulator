"""Model training, inference, explanations, and evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_LABELS, MODEL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = PROJECT_ROOT / "artifacts" / "pairwise_model.joblib"
DEFAULT_METADATA = PROJECT_ROOT / "artifacts" / "model_metadata.json"


def make_model() -> Pipeline:
    """Interpretable, fast, calibrated-enough baseline for pairwise probabilities."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, test: pd.DataFrame) -> dict[str, float]:
    X = test[MODEL_FEATURES]
    y = test["target"].astype(int)
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "log_loss": float(log_loss(y, prob)),
        "brier_score": float(brier_score_loss(y, prob)),
    }


def train_time_holdout(pairwise: pd.DataFrame, holdout_races: int = 5) -> tuple[dict, Pipeline]:
    """Evaluate on the latest races, then refit the production model on all races."""
    races = sorted(pairwise["race_id"].unique())
    if len(races) <= holdout_races:
        raise ValueError("Not enough races for the requested holdout")
    split_race = races[-holdout_races]
    train = pairwise[pairwise["race_id"] < split_race]
    test = pairwise[pairwise["race_id"] >= split_race]

    evaluation_model = make_model()
    evaluation_model.fit(train[MODEL_FEATURES], train["target"])
    metrics = evaluate_model(evaluation_model, test)
    metrics.update(
        {
            "train_races": int(train["race_id"].nunique()),
            "holdout_races": int(test["race_id"].nunique()),
            "train_examples": int(len(train)),
            "holdout_examples": int(len(test)),
            "first_holdout_race_id": int(split_race),
        }
    )

    production_model = make_model()
    production_model.fit(pairwise[MODEL_FEATURES], pairwise["target"])
    return metrics, production_model


def save_model(
    model: Pipeline,
    metrics: dict[str, Any],
    artifact_path: Path | str = DEFAULT_ARTIFACT,
    metadata_path: Path | str = DEFAULT_METADATA,
) -> None:
    artifact_path = Path(artifact_path)
    metadata_path = Path(metadata_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    metadata = {
        "model_type": "StandardScaler + LogisticRegression",
        "scikit_learn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "target": "Probability that Driver A finishes ahead of Driver B",
        "features": MODEL_FEATURES,
        "metrics": metrics,
        "notes": [
            "Validation is time-based: the latest five races are held out.",
            "Production artifact is refit on all supplied races after evaluation.",
            "Current model uses the repository's 2024 Kaggle snapshot only.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model(path: Path | str = DEFAULT_ARTIFACT) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `uv run python -m f1_predictor.train`."
        )
    return joblib.load(path)


def predict_probability(model: Pipeline, X: pd.DataFrame) -> float:
    return float(model.predict_proba(X[MODEL_FEATURES])[:, 1][0])


def explain_linear_prediction(model: Pipeline, X: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Return per-feature log-odds contributions for the logistic pipeline.

    These are model influences, not causal effects. Positive values favour Driver A;
    negative values favour Driver B.
    """
    scaler: StandardScaler = model.named_steps["scale"]
    classifier: LogisticRegression = model.named_steps["model"]
    transformed = scaler.transform(X[MODEL_FEATURES])[0]
    coef = classifier.coef_[0]
    contribution = transformed * coef
    table = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "label": [FEATURE_LABELS.get(f, f) for f in MODEL_FEATURES],
            "contribution": contribution,
            "raw_value": X[MODEL_FEATURES].iloc[0].to_numpy(dtype=float),
        }
    )
    table["absolute_contribution"] = table["contribution"].abs()
    return table.sort_values("absolute_contribution", ascending=False).head(top_n).reset_index(drop=True)
