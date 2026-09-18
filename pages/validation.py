"""Validation: run a tuned configuration against a validation set. Not implemented yet."""

import streamlit as st

from evolver_studio.app_state import render_sidebar

st.title("Validation")

render_sidebar()

st.info(
    "🔍 Not implemented yet — see ROADMAP.md, Phase 4.\n\n"
    "Planned: launch a training run's winning configuration against a validation set and "
    "compare its indicator distribution against the algorithm's untuned default "
    "configuration."
)
