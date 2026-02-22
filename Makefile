.PHONY: db kernel api preprocess train test

db:
	duckdb data/duckdb/datathon.db -readonly

kernel:
	uv run python -m ipykernel install --user --name datathon

api:
	uv run uvicorn datathon.api.main:app --host 0.0.0.0 --port 8000

preprocess:
	uv run python -m datathon.preprocessing.pipeline

train:
	uv run python -m datathon.modeling.train

test:
	uv run pytest tests/ -v
