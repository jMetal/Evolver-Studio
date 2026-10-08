"""Explore › Problems: the problems Evolver can resolve by name, for training and solving.

Read from DescribeMain's `problemCatalogue` (evolver_studio/problem_catalogue.py): each problem's
family, encoding, number of objectives and variables, the arguments its constructor takes, and the
reference fronts found among the resources. Evolver 2.3 and older describe the problems by name
only, so with them the page lists the names.
"""

import pandas as pd
import streamlit as st

from evolver_studio.app_state import (
    registered_problem_names,
    registered_problems,
    require_evolver_jar,
)
from evolver_studio.catalogue import BASE_ALGORITHMS
from evolver_studio.problem_browser import ALL, render_filters, render_table

KEY_PREFIX = "explore_problems"


def _algorithms_for(encoding: str) -> list[str]:
    """The runnable algorithms that can solve problems of an encoding."""
    return [
        algorithm.name
        for algorithm in BASE_ALGORITHMS
        if algorithm.runnable_today and encoding in algorithm.runnable_encodings
    ]


st.title("Problems")

jar = require_evolver_jar()
problems = registered_problems(str(jar))
if problems is None:
    names = registered_problem_names(str(jar))
    if names is None:
        st.warning("Could not list the problems (is Java installed?).")
        st.stop()
    st.info(
        "This Evolver jar lists the problems by name only: their encoding, dimensions and "
        "arguments need Evolver 2.4 or later."
    )
    st.dataframe(pd.DataFrame({"Problem": names}), hide_index=True)
    st.stop()

st.markdown(
    "The problems Evolver resolves by name, in Run algorithm and in a training set. An "
    "algorithm solves the problems of the **encodings** it supports. **Objectives** and "
    "**variables** are those of the problem built with no arguments; a problem with "
    "**arguments** (the number of objectives and variables of DTLZ, WFG or ZCAT, for instance) "
    "takes all of them, in that order, or none, and a value is shown after each one's name when "
    "it is its default. Indicators need a **reference front** with the problem's number of "
    "objectives: the dimension suffix (`3D`) says which one."
)

encoding_filter = st.session_state.get(f"{KEY_PREFIX}_encoding")
shown = render_filters(problems, KEY_PREFIX)
if encoding_filter not in (None, ALL):
    st.caption(
        f"Algorithms that solve {encoding_filter} problems: "
        f"{', '.join(_algorithms_for(encoding_filter))}."
    )
render_table(shown, len(problems), f"{KEY_PREFIX}_table")
