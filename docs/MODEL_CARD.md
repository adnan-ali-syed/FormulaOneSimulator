# Grid Battle Model Card

## Purpose

Grid Battle estimates the probability that **Driver A finishes ahead of Driver B** under a user-selected circuit and starting-grid scenario. It is designed as a Datathon fan-engagement prototype, not as an official motorsport forecast or betting model.

## Current training data

The model uses the Formula One CSV snapshot included in `data/raw/`, focused on the supplied 2024 season data. The relevant sources are race results, qualifying results, and circuit metadata.

## Target

For each driver pair within the same race:

- `1` if Driver A finished ahead of Driver B
- `0` otherwise

Each unordered matchup is represented in both orientations so the learned comparison remains symmetric.

## Feature families

The training pipeline constructs pre-race features using only races that occurred before the race being predicted:

- starting-grid difference
- driver average finishing form
- driver average qualifying form
- driver average points
- driver average positions gained
- driver DNF rate
- driver top-5 rate
- constructor finishing and qualifying form
- constructor points and DNF rate
- driver and constructor performance by circuit type
- season experience
- a small number of grid × circuit-context interactions

## Leakage protection

Historical form is calculated with shifted expanding statistics. Team-level statistics are aggregated by team/race before shifting so one teammate's current-race result cannot leak into the other teammate's feature row.

## Validation

The latest five races are held out as a chronological validation set. This is deliberately stricter than a random row split because pairwise rows from the same race are highly related.

Current artifact metrics are stored in `artifacts/model_metadata.json`.

## Model

`StandardScaler + LogisticRegression`

This model was selected for the first product because it is:

- fast at inference
- lightweight for deployment
- probabilistic
- naturally suited to pairwise comparison
- directly explainable through feature contributions

## Limitations

1. The product currently learns from the supplied Kaggle snapshot only.
2. Weather, tyres, pit strategy, safety cars, and telemetry are not yet represented.
3. The current dataset should be validated against authoritative timing data before making real-world performance claims.
4. Probabilities represent model belief under the supplied data and assumptions, not true physical race probabilities.
5. The model does not predict DNFs or full-grid interactions as separate stochastic events yet.

## Intended next steps

- authoritative historical timing/lap data
- weather and track-temperature integration
- FastF1 lap/telemetry features
- tyre and pit strategy information
- reliability sub-model
- full-grid race simulation
