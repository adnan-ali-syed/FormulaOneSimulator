# 🏎️ Formula One Simulator

> **Change the conditions. Predict the race. Explore the “what if?”**

Formula One is one of the most data-rich sports in the world, but most race predictions are still consumed as fixed answers: *Who will win? Who has the fastest car? Who is likely to start on pole?*

Our Datathon microproduct takes a different approach.

**Formula One Simulator** is an interactive race-outcome simulator designed to improve fan engagement by allowing users to modify selected race conditions and explore how those changes could affect the predicted result.

Instead of only asking:

> **Who will win the race?**

we want users to ask:

> **What happens if the race conditions change?**

---

## 🎯 Project Idea

A Formula One race outcome depends on many interacting factors:

- driver performance
- constructor/team performance
- qualifying position
- circuit characteristics
- recent form
- reliability
- fastest-lap pace
- time of day
- temperature and weather
- telemetry-derived variables
- race-specific conditions

Our goal is to combine these signals into a predictive and simulation pipeline that generates a **predicted finishing grid** and **estimated lap-performance outputs**.

The final product will allow users to tweak a selected set of variables and immediately see how the predicted race outcome changes.

---

## 🧠 Product Vision

A user could select a race scenario such as:

```text
Circuit: Silverstone
Weather: Light rain
Track temperature: 18°C
Starting position: P4
Driver: Lando Norris
```

The application would then generate an updated prediction such as:

```text
P1  Driver A
P2  Driver B
P3  Driver C
...
P20 Driver T
```

along with lap-performance estimates and explanations of which factors influenced the prediction.

The product is intended to be an **interactive fan-engagement experience**, not just a static forecasting model.

---

## 🔁 Planned Modeling Architecture

Rather than forcing the entire problem into one model, we plan to break the race outcome into related prediction tasks.

```text
Historical race data
        +
Qualifying performance
        +
Driver performance
        +
Constructor performance
        +
Circuit characteristics
        +
Weather / race conditions
        +
Telemetry-derived features
        ↓
Feature engineering
        ↓
Predictive models
        ↓
Race simulation / ranking layer
        ↓
Predicted finishing grid
+
Lap-performance predictions
```

Potential modeling components include:

1. **Race pace / lap-performance model**
2. **Finishing-position or ranking model**
3. **Reliability / DNF component**
4. **Fastest-lap model**
5. **Simulation layer** that combines model outputs into a complete race outcome

The exact architecture will evolve as we better understand the available data and validate baseline approaches.

---

## 📊 Current Data

The initial dataset is the **Comprehensive Formula 1 Dataset (2020–2025)** from Kaggle:

**Source:** `vshreekamalesh/comprehensive-formula-1-dataset-2020-2025`

The current repository contains the following raw data files:

| File | Shape | Purpose |
|---|---:|---|
| `f1_2024_race_results.csv` | 480 × 16 | 2024 race outcomes |
| `f1_qualifying_results_2024.csv` | 480 × 14 | 2024 qualifying results |
| `f1_2024_driver_standings.csv` | 480 × 13 | Driver standings by round |
| `f1_2024_constructor_standings.csv` | 240 × 13 | Constructor standings by round |
| `f1_circuits_metadata.csv` | 24 × 14 | Circuit characteristics |
| `f1_historical_drivers.csv` | 30 × 16 | Historical driver metadata |

The data is stored under:

```text
data/raw/
```

Processed modeling tables will later be written to:

```text
data/processed/
```

---

## 🔍 Initial EDA

Initial exploratory analysis is available in:

```text
notebooks/01_initial_eda.ipynb
```

The first pass focuses on:

- dataset dimensions and schema
- missing values
- duplicate rows
- race and driver coverage
- qualifying versus finishing position
- positions gained or lost during races
- driver-level performance summaries
- constructor-level performance
- circuit metadata
- identifying variables that can become future model features

### Early observation

In the current 2024 data, qualifying position and finishing position have a correlation of approximately **0.89**.

This makes qualifying performance an obvious baseline predictor, while the remaining variance gives us room to explore additional signals such as team strength, circuit characteristics, reliability, weather, and telemetry.

The qualifying dataset contains missing Q2 and Q3 times because drivers eliminated in earlier sessions do not participate in later sessions. These values should therefore be treated as **structural missingness**, not automatically as data-quality errors.

---

## 🧪 Candidate Features

### Driver features

- qualifying position
- qualifying time / gap to pole
- recent finishing position
- championship points
- average positions gained/lost
- podium frequency
- fastest-lap frequency
- recent form
- circuit-specific historical performance

### Constructor features

- constructor championship position
- recent points
- wins / podiums
- reliability / DNFs
- pole positions
- fastest laps
- engine / constructor metadata

### Circuit features

- circuit length
- number of turns
- DRS zones
- elevation change
- race distance
- circuit type
- lap record

### Planned external features

As the project progresses, we plan to evaluate additional sources for:

- weather
- track and air temperature
- rainfall
- wind
- tyre information
- pit-stop strategy
- sector times
- telemetry
- speed traces
- lap-by-lap race progression

---

## ⚠️ Important Modeling Principle

We must avoid **data leakage**.

Some variables in the current datasets describe information known only *after* or *during* the race, such as final championship totals or race outcomes. These cannot be used directly to predict that same race unless they are transformed into values that were genuinely available **before the prediction point**.

A major part of our next phase will therefore be constructing **pre-race features** using only historical information available up to each race.

---

## 📁 Repository Structure

```text
FormulaOneSimulator/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_initial_eda.ipynb
│
├── src/
│   └── f1_predictor/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── features.py
│       └── model.py
│
├── app/
│   └── app.py
│
└── docs/
    └── project_scope.md
```

---

## 🛠️ Tech Stack

- **Python**
- **uv** for dependency and environment management
- **pandas** / **NumPy** for data processing
- **Matplotlib** / **Plotly** for exploration and visualization
- **scikit-learn** for baseline predictive modeling
- **Streamlit** for the planned interactive microproduct

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd FormulaOneSimulator
```

### 2. Recreate the environment

```bash
uv sync
```

### 3. Launch Jupyter

```bash
uv run jupyter lab
```

### 4. Open the initial EDA

```text
notebooks/01_initial_eda.ipynb
```

---

## 🚧 Current Progress

- [x] Define the fan-engagement problem
- [x] Create shared GitHub repository
- [x] Set up a reproducible `uv` environment
- [x] Identify and add the initial Formula One dataset
- [x] Organize raw and processed data directories
- [x] Begin exploratory data analysis
- [x] Investigate qualifying vs. race-result relationships
- [ ] Design leakage-safe pre-race feature pipeline
- [ ] Build baseline race-position model
- [ ] Add circuit and constructor features
- [ ] Integrate weather data
- [ ] Investigate telemetry data sources
- [ ] Build fastest-lap / pace model
- [ ] Create race simulation layer
- [ ] Build interactive Streamlit interface
- [ ] Evaluate prediction quality
- [ ] Prepare final Datathon demo

---

## 🗺️ Roadmap

### Phase 1 — Data understanding
Explore the supplied datasets, validate data quality, identify useful fields, and document limitations.

### Phase 2 — Feature engineering
Create leakage-safe driver, constructor, circuit, form, and qualifying features.

### Phase 3 — Baseline modeling
Build simple interpretable baselines before moving to more complex models.

### Phase 4 — Data enrichment
Incorporate weather, telemetry, tyre, pit-stop, and lap-level information where feasible.

### Phase 5 — Simulation
Combine predictive outputs to generate a full race ranking under a selected set of conditions.

### Phase 6 — Interactive product
Allow fans to change race conditions and immediately explore alternative predicted outcomes.

---

## 🌟 Why This Matters

Formula One fans already consume huge amounts of race data.

Our goal is to let them **interact with it**.

Instead of treating a race prediction as a single fixed answer, Formula One Simulator turns prediction into a sandbox:

> **Change the scenario. Re-run the race. Understand what changed.**

That is the experience we want to build.
