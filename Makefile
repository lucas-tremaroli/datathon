.PHONY: db kernel dash

db:
	duckdb data/raw/datathon.db -readonly

kernel:
	uv run python -m ipykernel install --user --name datathon

dash:
	uv run streamlit run datathon/dashboard/main.py
