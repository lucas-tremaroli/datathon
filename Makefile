.PHONY: db kernel

db:
	duckdb data/duckdb/datathon.db -readonly

kernel:
	uv run python -m ipykernel install --user --name datathon
