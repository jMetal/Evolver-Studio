"""Tutorials: interactive, step-by-step tutorials. Not implemented yet."""

import streamlit as st

from evolver_studio.app_state import render_sidebar

TUTORIALS_CATALOGUE_URL = (
    "https://github.com/jMetal/Evolver/blob/develop/docs/proposals/tutorials.md"
)

st.title("Tutorials")

render_sidebar()

st.info(
    "🔍 Not implemented yet — see ROADMAP.md, Next up.\n\n"
    "Planned: interactive tutorials for both tracks — solving problems with Evolver's "
    "configurable algorithms, and meta-optimization (training, analysis, validation) — each "
    "paired with its counterpart in Evolver's documentation. The planned tutorials are listed in "
    f"the [tutorials catalogue]({TUTORIALS_CATALOGUE_URL})."
)
