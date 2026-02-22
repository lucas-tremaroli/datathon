.PHONY: db kernel api

db:
	duckdb data/duckdb/datathon.db -readonly

kernel:
	uv run python -m ipykernel install --user --name datathon

api:
	uv run uvicorn datathon.api.main:app --host 0.0.0.0 --port 8000
