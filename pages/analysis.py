"""Analysis: statistical comparison of training results. Not implemented yet."""

import streamlit as st

from evolver_studio.app_state import render_sidebar

st.title("Analysis")

render_sidebar()

st.info(
    "🔍 Not implemented yet — see ROADMAP.md, Phase 3.\n\n"
    "Planned: structured parsing of CONFIGURATIONS.csv/VAR_CONF.txt, a history of past runs "
    "(not just the latest), and statistical comparison across runs/configurations "
    "(Wilcoxon, comparison tables)."
)
