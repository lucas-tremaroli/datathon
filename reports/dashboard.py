import altair as alt
import streamlit as st

from datathon.db import DuckDBClient

db = DuckDBClient("data/duckdb/datathon.db")

school_query = open("data/queries/school.sql").read()
school_df = db.execute_query(school_query)

st.altair_chart(
    alt.Chart(school_df)
    .mark_arc()
    .encode(
        alt.Theta("students", stack=True),
        alt.Color("name"),
    )
)
