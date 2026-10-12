.PHONY: setup train run test

setup:
	uv sync

train:
	uv run python -m f1_predictor.train

run:
	uv run streamlit run app/app.py

test:
	uv run pytest
