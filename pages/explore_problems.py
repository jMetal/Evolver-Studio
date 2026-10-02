"""Explore › Problems: the problems available for training and solving. Not implemented yet."""

import streamlit as st

from evolver_studio.app_state import render_sidebar

st.title("Problems")

render_sidebar()

st.info(
    "🔍 Not implemented yet — see ROADMAP.md.\n\n"
    "Planned: browse the problems Evolver can train and solve on (the families of its "
    "ProblemRegistry: ZDT, DTLZ, WFG, UF, LZ09, LSMOP, ZCAT, and the real-world RE and RWA "
    "problems), with their number of objectives and variables, their reference front, and the "
    "training sets they belong to."
)
