"""Validation analysis: study the results of a validation. Not implemented yet."""

import streamlit as st

from evolver_studio.app_state import render_sidebar

st.title("Validation analysis")

render_sidebar()

st.info(
    "🔍 Not implemented yet — see ROADMAP.md.\n\n"
    "The last step of meta-optimization: after Training, Training analysis and Validation, the "
    "analysis of a validation study. Meanwhile, the Validation page shows the results of its "
    "studies."
)
