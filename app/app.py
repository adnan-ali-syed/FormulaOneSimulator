from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1_predictor.features import (  # noqa: E402
    FEATURE_LABELS,
    build_latest_context,
    build_scenario_features,
)
from f1_predictor.model import (  # noqa: E402
    DEFAULT_METADATA,
    explain_linear_prediction,
    load_model,
    predict_probability,
)
from f1_predictor.preprocessing import build_driver_race_features, load_raw_data  # noqa: E402

st.set_page_config(
    page_title="Grid Battle | Formula One Simulator",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px;}
    .hero {padding: 1.25rem 1.4rem; border: 1px solid rgba(255,255,255,.12); border-radius: 18px;
           background: linear-gradient(135deg, rgba(225,6,0,.16), rgba(255,255,255,.025)); margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 2.25rem;}
    .hero p {margin: .45rem 0 0 0; color: #c9c9c9; font-size: 1.02rem;}
    .metric-card {padding: 1rem 1.1rem; border: 1px solid rgba(255,255,255,.11); border-radius: 14px;
                  background: rgba(255,255,255,.025); min-height: 115px;}
    .metric-card .kicker {color:#a7a7a7; font-size:.8rem; text-transform:uppercase; letter-spacing:.08em;}
    .metric-card .value {font-size:1.65rem; font-weight:700; margin-top:.2rem;}
    .metric-card .sub {color:#b7b7b7; font-size:.9rem; margin-top:.2rem;}
    .winner {padding: 1.15rem; border:1px solid rgba(225,6,0,.45); border-radius:16px; background:rgba(225,6,0,.09);}
    .small-note {font-size:.84rem; color:#a7a7a7;}
    div[data-testid="stMetric"] {border:1px solid rgba(255,255,255,.09); padding:.75rem; border-radius:12px;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading race model…")
def load_runtime():
    model = load_model()
    driver_race = build_driver_race_features()
    context = build_latest_context(driver_race)
    raw = load_raw_data()
    metadata = {}
    if DEFAULT_METADATA.exists():
        metadata = json.loads(DEFAULT_METADATA.read_text(encoding="utf-8"))
    return model, driver_race, context, raw, metadata


@st.cache_data(show_spinner=False)
def head_to_head(driver_race: pd.DataFrame, driver_a: str, driver_b: str) -> pd.DataFrame:
    cols = ["race_id", "race_name", "circuit", "driver_name", "position", "qualifying_position"]
    sub = driver_race.loc[driver_race["driver_name"].isin([driver_a, driver_b]), cols].copy()
    a = sub[sub["driver_name"].eq(driver_a)].rename(
        columns={"position": "finish_a", "qualifying_position": "grid_a"}
    )
    b = sub[sub["driver_name"].eq(driver_b)].rename(
        columns={"position": "finish_b", "qualifying_position": "grid_b"}
    )
    joined = a[["race_id", "race_name", "circuit", "finish_a", "grid_a"]].merge(
        b[["race_id", "finish_b", "grid_b"]], on="race_id", how="inner"
    )
    joined["winner"] = joined.apply(
        lambda r: driver_a if r["finish_a"] < r["finish_b"] else driver_b, axis=1
    )
    return joined


def probability_bar(driver_a: str, driver_b: str, prob_a: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[prob_a * 100],
            y=["Battle"],
            orientation="h",
            name=driver_a,
            hovertemplate=f"{driver_a}: %{{x:.1f}}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[(1 - prob_a) * 100],
            y=["Battle"],
            orientation="h",
            name=driver_b,
            hovertemplate=f"{driver_b}: %{{x:.1f}}%<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="stack",
        height=180,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis=dict(range=[0, 100], ticksuffix="%", title="Model probability"),
        yaxis=dict(showticklabels=False, title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def influence_chart(explanation: pd.DataFrame, driver_a: str, driver_b: str) -> go.Figure:
    table = explanation.sort_values("contribution")
    fig = go.Figure(
        go.Bar(
            x=table["contribution"],
            y=table["label"],
            orientation="h",
            customdata=table[["raw_value"]],
            hovertemplate="%{y}<br>Influence: %{x:.3f}<br>Raw difference: %{customdata[0]:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, opacity=0.5)
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title=f"Model influence  ← favours {driver_b} | favours {driver_a} →",
        yaxis_title="",
    )
    return fig


def sensitivity_chart(model, context, driver_a, driver_b, circuit, grid_b) -> go.Figure:
    rows = []
    for grid_a in range(1, 21):
        X, *_ = build_scenario_features(context, driver_a, driver_b, circuit, grid_a, grid_b)
        rows.append({"grid_a": grid_a, "prob": 100 * predict_probability(model, X)})
    data = pd.DataFrame(rows)
    fig = go.Figure(
        go.Scatter(
            x=data["grid_a"],
            y=data["prob"],
            mode="lines+markers",
            hovertemplate=f"{driver_a} starts P%{{x}}<br>Win-ahead probability: %{{y:.1f}}%<extra></extra>",
        )
    )
    fig.add_hline(y=50, line_dash="dash", opacity=0.45)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title=f"{driver_a} starting position", dtick=1, autorange="reversed"),
        yaxis=dict(title=f"P({driver_a} finishes ahead)", ticksuffix="%", range=[0, 100]),
    )
    return fig


def fmt(value, digits=2):
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


try:
    model, driver_race, context, raw, metadata = load_runtime()
except Exception as exc:
    st.error("The Grid Battle runtime could not be loaded.")
    st.code(str(exc))
    st.info("From the project root run: `uv run python -m f1_predictor.train`")
    st.stop()

st.markdown(
    """
<div class="hero">
  <h1>🏎️ Grid Battle</h1>
  <p><b>Two drivers. One circuit. Your starting grid.</b> Explore who the model favours — and what is driving the matchup.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Build the battle")
    drivers = context["driver"].sort_values("driver_name")["driver_name"].tolist()
    circuits = context["circuits"]["circuit"].tolist()

    default_a = drivers.index("Lando Norris") if "Lando Norris" in drivers else 0
    default_b = drivers.index("Charles Leclerc") if "Charles Leclerc" in drivers else min(1, len(drivers) - 1)
    driver_a = st.selectbox("Driver A", drivers, index=default_a)
    driver_b = st.selectbox("Driver B", drivers, index=default_b)
    circuit = st.selectbox("Circuit", circuits)

    st.divider()
    equal_grid = st.toggle("Equal-grid mode", value=False, help="Give both drivers the same starting position to isolate the model's non-grid signals.")
    if equal_grid:
        shared_grid = st.slider("Starting position", 1, 20, 5)
        grid_a = shared_grid
        grid_b = shared_grid
    else:
        grid_a = st.slider(f"{driver_a} starting position", 1, 20, 3)
        grid_b = st.slider(f"{driver_b} starting position", 1, 20, 5)

    if not equal_grid and grid_a == grid_b:
        st.caption("Real grids cannot share a position; this is allowed here as a controlled what-if scenario.")

    run = st.button("⚡ Run battle", type="primary", use_container_width=True)

    st.divider()
    st.caption("Current V1 uses the supplied 2024 dataset. Weather, tyres, and telemetry are planned enrichment layers and are intentionally not fabricated in this build.")

# Freeze the active scenario until the user presses Run Battle. On the first render,
# initialize it from the default controls so the page is never empty.
if "active_scenario" not in st.session_state or run:
    st.session_state["active_scenario"] = {
        "driver_a": driver_a,
        "driver_b": driver_b,
        "circuit": circuit,
        "grid_a": int(grid_a),
        "grid_b": int(grid_b),
    }

scenario = st.session_state["active_scenario"]
driver_a = scenario["driver_a"]
driver_b = scenario["driver_b"]
circuit = scenario["circuit"]
grid_a = scenario["grid_a"]
grid_b = scenario["grid_b"]

if driver_a == driver_b:
    st.warning("Choose two different drivers and press Run battle.")
    st.stop()

X, a, b, c = build_scenario_features(context, driver_a, driver_b, circuit, grid_a, grid_b)
prob_a = predict_probability(model, X)
prob_b = 1.0 - prob_a
winner = driver_a if prob_a >= 0.5 else driver_b
winner_prob = max(prob_a, prob_b)

left, right = st.columns([1.15, 1], gap="large")
with left:
    st.subheader(f"{driver_a}  ⚔️  {driver_b}")
    st.plotly_chart(probability_bar(driver_a, driver_b, prob_a), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f"""
<div class="winner">
  <div class="small-note">MODEL FAVOURITE</div>
  <div style="font-size:1.9rem;font-weight:800;margin-top:.15rem;">🏁 {winner}</div>
  <div style="margin-top:.25rem;">{winner_prob*100:.1f}% probability of finishing ahead in this scenario</div>
</div>
""",
        unsafe_allow_html=True,
    )

with right:
    st.subheader("Scenario")
    c1, c2 = st.columns(2)
    c1.metric(f"{driver_a} grid", f"P{grid_a}")
    c2.metric(f"{driver_b} grid", f"P{grid_b}")
    c1.metric("Circuit type", c["circuit_type"])
    c2.metric("DRS zones", int(c["drs_zones"]))
    c1.metric("Circuit length", f"{c['circuit_length_km']:.3f} km")
    c2.metric("Turns", int(c["number_of_turns"]))
    st.caption(f"{c['city']}, {c['country']} · {int(c['total_laps'])} laps · race distance {c['race_distance_km']:.1f} km")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Why this result", "Driver comparison", "Grid sensitivity", "Model & limitations"])

with tab1:
    explanation = explain_linear_prediction(model, X, top_n=8)
    st.markdown(
        "The chart below shows the strongest **model influences** for this prediction. Positive values favour Driver A; negative values favour Driver B. These are not causal effects."
    )
    st.plotly_chart(influence_chart(explanation, driver_a, driver_b), use_container_width=True, config={"displayModeBar": False})

    strongest = explanation.iloc[0]
    direction = driver_a if strongest["contribution"] > 0 else driver_b
    st.info(f"Largest current influence: **{strongest['label']}**, which favours **{direction}** in this matchup.")

with tab2:
    compare = pd.DataFrame(
        {
            "Metric": [
                "Team",
                "Starting grid",
                "2024 avg finish",
                "2024 avg qualifying",
                "2024 avg points / race",
                "Avg positions gained",
                "Top-5 rate",
                "DNF rate",
                f"Avg finish on {c['circuit_type'].lower()}s",
            ],
            driver_a: [
                a["team"],
                int(grid_a),
                fmt(a["driver_avg_finish"]),
                fmt(a["driver_avg_qual"]),
                fmt(a["driver_avg_points"]),
                fmt(a["driver_avg_pos_gained"]),
                f"{100*a['driver_top5_rate']:.1f}%",
                f"{100*a['driver_dnf_rate']:.1f}%",
                fmt(a["driver_type_finish"]),
            ],
            driver_b: [
                b["team"],
                int(grid_b),
                fmt(b["driver_avg_finish"]),
                fmt(b["driver_avg_qual"]),
                fmt(b["driver_avg_points"]),
                fmt(b["driver_avg_pos_gained"]),
                f"{100*b['driver_top5_rate']:.1f}%",
                f"{100*b['driver_dnf_rate']:.1f}%",
                fmt(b["driver_type_finish"]),
            ],
        }
    )
    st.dataframe(compare, hide_index=True, use_container_width=True)

    h2h = head_to_head(driver_race, driver_a, driver_b)
    wins_a = int((h2h["winner"] == driver_a).sum())
    wins_b = int((h2h["winner"] == driver_b).sum())
    st.subheader("Observed 2024 head-to-head in the supplied dataset")
    h1, h2, h3 = st.columns(3)
    h1.metric(driver_a, wins_a)
    h2.metric(driver_b, wins_b)
    h3.metric("Shared races", len(h2h))
    with st.expander("See race-by-race head-to-head"):
        st.dataframe(h2h.sort_values("race_id"), hide_index=True, use_container_width=True)

with tab3:
    st.markdown(
        f"Hold **{driver_b} at P{grid_b}** and move {driver_a} through the grid to see how sensitive the model is to starting position."
    )
    st.plotly_chart(
        sensitivity_chart(model, context, driver_a, driver_b, circuit, grid_b),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with tab4:
    metrics = metadata.get("metrics", {})
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Time-holdout accuracy", f"{100*metrics.get('accuracy', 0):.1f}%")
        m2.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
        m3.metric("Brier score", f"{metrics.get('brier_score', 0):.3f}")
        m4.metric("Holdout races", int(metrics.get("holdout_races", 0)))

    st.markdown(
        """
### What this V1 does
- Predicts the probability that **Driver A finishes ahead of Driver B**.
- Uses leakage-safe historical form features for training.
- Uses starting-grid position, driver form, constructor form/reliability proxies, and circuit-type context.
- Evaluates on the latest races as a time-based holdout before refitting the deployment artifact.

### What it does *not* yet do
- It does not model live weather, tyre compounds, pit strategy, safety cars, or telemetry.
- It is not an official F1 forecast or betting tool.
- The supplied Kaggle snapshot is suitable for a Datathon prototype, but it should be replaced/validated against authoritative historical timing data before making real-world performance claims.

### Product direction
The next enrichment layer is **FastF1/official timing + weather + lap/telemetry features**, followed by a richer scenario engine and eventually full-grid simulation.
"""
    )

st.divider()
st.caption("Grid Battle · Formula One Simulator Datathon prototype · Built for interactive fan exploration")
