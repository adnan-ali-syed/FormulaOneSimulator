# Deployment Guide

Grid Battle is intentionally stateless and lightweight. The trained model is loaded once per process and cached by Streamlit, while inference is a single logistic-regression call. This is appropriate for a small-to-medium Datathon/demo workload on one or a few application instances.

## Local development with uv

```bash
uv sync
uv run python -m f1_predictor.train
uv run streamlit run app/app.py
```

The app will be available at `http://localhost:8501`.

## Tests

```bash
uv run pytest
```

## Docker

Build:

```bash
docker build -t grid-battle .
```

Run:

```bash
docker run --rm -p 8501:8501 grid-battle
```

Then open `http://localhost:8501`.

## Generic container-host deployment

Any service capable of running a Docker container can deploy this project. Expose port `8501` and keep at least one instance warm for best interactive latency.

The application requires no database, user account, API key, or writable persistent storage in V1.

## Scaling notes

- The model and source tables are cached with Streamlit resource/data caching.
- No model fitting occurs on each request.
- Inference is CPU-light and takes negligible memory compared with the Streamlit process itself.
- Sessions contain only UI state; there is no shared mutable user state.
- For materially larger traffic, split inference into a FastAPI service and place multiple stateless app/API replicas behind a load balancer.

## Model refresh

When the raw data or feature engineering changes:

```bash
uv run python -m f1_predictor.train
```

Commit the updated files under `artifacts/` along with the code/data change so deployment uses the matching model.
