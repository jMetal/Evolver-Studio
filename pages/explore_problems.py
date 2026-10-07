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
from evolver_studio.problem_catalogue import Problem, format_arguments
from evolver_studio.problems import reference_front_candidates

ALL = "All"
ENCODING_KEY = "explore_problems_encoding"
FAMILY_KEY = "explore_problems_family"
FILTER_KEY = "explore_problems_filter"


def _problem_table(problems: list[Problem]) -> pd.DataFrame:
    """One row per problem, with its reference fronts' file names.

    Args:
        problems: The problems to show.

    Returns:
        The table.
    """
    rows = [
        {
            "Problem": problem.name,
            "Family": problem.family,
            "Encoding": problem.encoding,
            "Objectives": problem.number_of_objectives,
            "Variables": problem.number_of_variables,
            "Arguments": format_arguments(problem),
            "Reference fronts": ", ".join(
                path.rsplit("/", 1)[-1] for path in reference_front_candidates(problem.name)
            ),
        }
        for problem in problems
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Problem",
            "Family",
            "Encoding",
            "Objectives",
            "Variables",
            "Arguments",
            "Reference fronts",
        ],
    )


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
        "arguments need a newer Evolver."
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

encodings = sorted({problem.encoding for problem in problems.values()})
families = sorted({problem.family for problem in problems.values()})
columns = st.columns(3)
encoding = columns[0].selectbox("Encoding", [ALL, *encodings], key=ENCODING_KEY)
family = columns[1].selectbox("Family", [ALL, *families], key=FAMILY_KEY)
text = columns[2].text_input("Filter by name", key=FILTER_KEY, placeholder="e.g. DTLZ")

shown = [
    problem
    for problem in sorted(problems.values(), key=lambda problem: (problem.family, problem.name))
    if encoding in (ALL, problem.encoding)
    and family in (ALL, problem.family)
    and text.strip().lower() in problem.name.lower()
]
if encoding != ALL:
    st.caption(
        f"Algorithms that solve {encoding} problems: {', '.join(_algorithms_for(encoding))}."
    )
st.caption(f"{len(shown)} of {len(problems)} problems.")
st.dataframe(_problem_table(shown), hide_index=True)
