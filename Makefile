.PHONY: db kernel dash

db:
	duckdb data/datathon.db

kernel:
	uv run python -m ipykernel install --user --name datathon

dash:
	uv run streamlit run datathon/dashboard/main.py
