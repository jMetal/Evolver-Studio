"""Run algorithm: configure and run a base-level algorithm on a problem. Not implemented yet."""

import streamlit as st

from evolver_studio.app_state import render_sidebar

st.title("Run algorithm")

render_sidebar()

st.info(
    "🔍 Not implemented yet — see ROADMAP.md, Next up (solving track).\n\n"
    "Planned: configure and run one of Evolver's configurable algorithms on a concrete problem, "
    "in the style of jMetal's runners — choose the problem, the algorithm and its encoding, and a "
    "configuration (default, tuned or edited in the guided form); run it; inspect the front and "
    "its quality indicators; export VAR/FUN. Needs an Evolver-side entry point for single "
    "algorithm runs, analogous to cli.training."
)
