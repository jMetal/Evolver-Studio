"""The listing of Evolver's problems and the selector built on it: shared by Explore, Training
and Validation.

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


def add_problems(current: list[str], added: list[str]) -> list[str]:
    """Add problems to the chosen ones, keeping their order and leaving out the repeated.

    Args:
        current: The problems already chosen.
        added: The problems to add.

    Returns:
        The chosen problems followed by the new ones.
    """
    return list(dict.fromkeys([*current, *added]))


def _add_chosen_problems(selector_key: str, added: list[str], resets_key: str) -> None:
    """Add problems to the chosen ones, and clear the listing's selection.

    A button's callback, which may set the selector's value. The listing's rows are unselected by
    giving the table a new key: a selection cannot be cleared otherwise.
    """
    st.session_state[selector_key] = add_problems(st.session_state.get(selector_key) or [], added)
    st.session_state[resets_key] = st.session_state.get(resets_key, 0) + 1


def render_problem_adder(
    catalogue: Mapping[str, Problem], encoding: str, selector_key: str, key_prefix: str
) -> None:
    """Offer the listing, to add problems to a selector with their details in view.

    A switch shows the listing of Explore › Problems for an encoding; selecting rows and pressing
    the button adds them to the selector that holds the chosen problems.

    Args:
        catalogue: The problem catalogue.
        encoding: The encoding whose problems are listed.
        selector_key: The widget key of the selector the problems are added to.
        key_prefix: Prefixes the widgets' keys, so that pages do not share them.
    """
    if not st.toggle(
        "Browse the problems",
        key=f"{key_prefix}_browse_{encoding}",
        help="The same listing as Explore › Problems, with each problem's objectives, variables, "
        "arguments and reference fronts: select rows and add them.",
    ):
        return
    shown = render_filters(catalogue, f"{key_prefix}_browser_{encoding}", fixed_encoding=encoding)
    resets_key = f"{key_prefix}_browser_resets_{encoding}"
    selected = render_table(
        shown,
        len(catalogue),
        f"{key_prefix}_browser_table_{encoding}_{st.session_state.get(resets_key, 0)}",
        selectable=True,
    )
    st.button(
        f"Add the {len(selected)} selected" if selected else "Add the selected problems",
        disabled=not selected,
        on_click=_add_chosen_problems,
        args=(selector_key, selected, resets_key),
        key=f"{key_prefix}_browser_add_{encoding}",
    )


def render_problem_selector(
    catalogue: Mapping[str, Problem] | None,
    options: list[str],
    encoding: str,
    label: str,
    selector_key: str,
    key_prefix: str,
    default: list[str] | None = None,
    help_text: str | None = None,
    accept_class_names: bool = False,
) -> list[str]:
    """Let the user choose problems: from a list, or with their details in a listing.

    It is the one way Training and Validation choose them: a selector with the problems of an
    encoding and, when the jar describes the problems, the listing of Explore › Problems to add
    several rows at once.

    Args:
        catalogue: The problem catalogue, or None when the jar does not describe the problems
            (there is then no listing).
        options: The problems the selector offers.
        encoding: The encoding of the problems.
        label: The selector's label.
        selector_key: The selector's widget key, also the one the listing adds to.
        key_prefix: Prefixes the listing's widget keys, so that pages do not share them.
        default: The problems chosen at first, those of them that are offered.
        help_text: The selector's help.
        accept_class_names: Whether a fully-qualified class name can be typed to add a problem
            that is not in the list.

    Returns:
        The names chosen.
    """
    names = st.multiselect(
        label,
        options,
        default=[name for name in default or [] if name in options],
        accept_new_options=accept_class_names,
        placeholder="Choose the problems, or type a class name"
        if accept_class_names
        else "Choose the problems",
        key=selector_key,
        help=help_text,
    )
    if catalogue is not None:
        render_problem_adder(catalogue, encoding, selector_key, key_prefix)
    return names
