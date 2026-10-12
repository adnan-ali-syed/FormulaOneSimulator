"""Train and persist the Grid Battle pairwise model."""
from __future__ import annotations

from pathlib import Path

from .features import build_pairwise_training_data
from .model import save_model, train_time_holdout
from .preprocessing import build_driver_race_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    driver_race = build_driver_race_features()
    pairwise = build_pairwise_training_data(driver_race)
    metrics, model = train_time_holdout(pairwise, holdout_races=5)
    save_model(model, metrics)

    processed = PROJECT_ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(processed / "pairwise_training_data.csv", index=False)

    print("Grid Battle model trained successfully.")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
