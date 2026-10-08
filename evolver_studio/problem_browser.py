"""The listing of Evolver's problems, with its filters: shared by Explore and by Training.

Explore › Problems shows it to browse the problems; Training shows it, with the encoding fixed to
the base algorithm's and rows that can be selected, to add problems to the training set.
"""

from collections.abc import Mapping

import pandas as pd
import streamlit as st

from evolver_studio.problem_catalogue import Problem, format_arguments
from evolver_studio.problems import reference_front_candidates

ALL = "All"
COLUMNS = (
    "Problem",
    "Family",
    "Encoding",
    "Objectives",
    "Variables",
    "Arguments",
    "Reference fronts",
)


def problem_frame(problems: list[Problem]) -> pd.DataFrame:
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
    return pd.DataFrame(rows, columns=list(COLUMNS))


def render_filters(
    catalogue: Mapping[str, Problem], key_prefix: str, fixed_encoding: str | None = None
) -> list[Problem]:
    """Show the filters (encoding, family, name) and apply them.

    Args:
        catalogue: The problem catalogue.
        key_prefix: Prefixes the filters' widget keys: `<prefix>_encoding`, `_family`, `_filter`.
        fixed_encoding: The encoding to show the problems of, without offering to choose it; None
            to offer it.

    Returns:
        The problems that pass the filters, sorted by family and name.
    """
    candidates = [p for p in catalogue.values() if fixed_encoding in (None, p.encoding)]
    families = sorted({problem.family for problem in candidates})
    columns = st.columns(2 if fixed_encoding else 3)
    encoding = fixed_encoding or ALL
    if fixed_encoding is None:
        encodings = sorted({problem.encoding for problem in catalogue.values()})
        encoding = columns[0].selectbox("Encoding", [ALL, *encodings], key=f"{key_prefix}_encoding")
    family = columns[-2].selectbox("Family", [ALL, *families], key=f"{key_prefix}_family")
    text = columns[-1].text_input(
        "Filter by name", key=f"{key_prefix}_filter", placeholder="e.g. DTLZ"
    )
    return [
        problem
        for problem in sorted(catalogue.values(), key=lambda p: (p.family, p.name))
        if encoding in (ALL, problem.encoding)
        and family in (ALL, problem.family)
        and text.strip().lower() in problem.name.lower()
    ]


def render_table(shown: list[Problem], total: int, key: str, selectable: bool = False) -> list[str]:
    """Show the problems as a table.

    Args:
        shown: The problems to show.
        total: How many problems there are before filtering, for the count under the filters.
        key: The table's widget key, when its rows can be selected.
        selectable: Whether rows can be selected (several at once).

    Returns:
        The names of the selected problems; empty when the rows cannot be selected.
    """
    st.caption(f"{len(shown)} of {total} problems.")
    frame = problem_frame(shown)
    if not selectable:
        st.dataframe(frame, hide_index=True)
        return []
    event = st.dataframe(
        frame, hide_index=True, on_select="rerun", selection_mode="multi-row", key=key
    )
    return [shown[index].name for index in event.selection.rows]
