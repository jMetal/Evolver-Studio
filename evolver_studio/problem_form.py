"""The form of a problem's arguments and reference front, shared by the pages that choose problems.

Run algorithm chooses one problem; Training and Validation choose several, and keep them as the
rows of a table (problem, arguments as a comma-separated text, reference front and, in Training,
evaluations), which their checks and the restoring of a study or a preset read and write. This
module edits those rows with widgets instead of a text table: a checkbox to set a problem's
arguments, one input per argument with its name and its default, and a selector of the problem's
reference fronts that follows its number of objectives (choosing 3 objectives for WFG1 offers
WFG1.3D.csv).

The widgets of the rows are keyed with a version of the table: when the rows are set from outside
the form (a study run again, a training's configuration taken to Validation, a tutorial's preset),
the version changes and the widgets start again from the rows.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import streamlit as st

from evolver_studio.problem_catalogue import (
    ArgumentValue,
    Problem,
    arguments_text,
    number_of_objectives,
    parse_arguments_text,
)
from evolver_studio.problems import (
    default_reference_front,
    reference_front_candidates,
    reference_front_dimension,
    reference_front_for_objectives,
)

OTHER_FRONT = "(another file)"
FRONT_HELP = (
    "The indicators compare each front with it. Fronts with a dimension suffix (3D) belong to the "
    "problem with that number of objectives: it must match the problem's."
)
FRONT_FILE_HELP = (
    "A reference front file, relative to the app's working directory (e.g. "
    "resources/referenceFronts/DTLZ1.3D.csv)."
)
UNTYPED_ARGUMENTS_HELP = (
    "Optional: the constructor's arguments, all of them or none, comma-separated; Evolver matches "
    "them against a constructor of the class."
)


def defaults_help(problem: Problem) -> str:
    """Say what a problem is built with when its arguments are not set.

    Args:
        problem: The problem.

    Returns:
        E.g. "Without them, WFG1 is built with its defaults (2 objectives, 24 variables)."
    """
    shape = ", ".join(
        part
        for part in (
            f"{problem.number_of_objectives} objectives" if problem.number_of_objectives else "",
            f"{problem.number_of_variables} variables" if problem.number_of_variables else "",
        )
        if part
    )
    return f"Without them, {problem.name} is built with its defaults" + (
        f" ({shape})." if shape else "."
    )


def initial_argument_values(problem: Problem, text: str) -> list[ArgumentValue | None]:
    """The values the inputs of a problem's arguments start from.

    Args:
        problem: The problem.
        text: Its arguments as a comma-separated text (as a row keeps them), or "" for none.

    Returns:
        The values of the text when it is valid and complete, else the defaults of the arguments.
    """
    parsed = parse_arguments_text(problem, text)
    if isinstance(parsed, list) and parsed:
        return list(parsed)
    return [argument.default for argument in problem.arguments]


def argument_values_text(values: Sequence[ArgumentValue | None]) -> str:
    """Write the values of the inputs as a row keeps them, a missing one as an empty piece.

    Args:
        values: The values, None for an input left empty.

    Returns:
        The comma-separated text (e.g. "4, 20, 3"); an empty piece makes the checks of the page
        say which argument has no value.
    """
    return ", ".join("" if value is None else arguments_text([value]) for value in values)


def objectives_of(problem: Problem, values: Sequence[ArgumentValue | None]) -> int | None:
    """The number of objectives of a problem built with the values of its inputs.

    Args:
        problem: The problem.
        values: The values of its arguments, all of them, or empty for none.

    Returns:
        The number of objectives, that of the problem with no arguments while a value is missing,
        or None when it is unknown.
    """
    if any(value is None for value in values):
        return problem.number_of_objectives
    complete = [value for value in values if value is not None]
    return number_of_objectives(problem, complete)


def initial_front(
    candidates: list[str], current: str, objectives: int | None, working_directory: Path
) -> str | None:
    """The reference front a problem's selector starts from.

    Args:
        candidates: The problem's reference front files.
        current: The front its row has, or "".
        objectives: The problem's number of objectives, or None when it is unknown.
        working_directory: The directory the fronts' paths are relative to.

    Returns:
        The current front when it is a candidate with the problem's number of objectives, else the
        only candidate, else the only one with that number of objectives; None when none fits.
    """
    if current in candidates and _has_objectives(working_directory / current, objectives):
        return current
    chosen = default_reference_front(candidates)
    if chosen is None and objectives is not None:
        chosen = reference_front_for_objectives(candidates, objectives, working_directory)
    return chosen


def rows_digest(table: pd.DataFrame) -> str:
    """A text that changes when the rows of a table change, to tell whether the form wrote them."""
    return table.to_json(orient="records")


def render_argument_inputs(
    problem: Problem,
    argument_key: Callable[[str], str],
    initial: Sequence[ArgumentValue | None],
) -> list[ArgumentValue | None]:
    """An input per argument of a problem, by its type, in up to four columns.

    Args:
        problem: The problem, which has arguments.
        argument_key: The widget key of an argument, given its name.
        initial: The values the inputs start from, one per argument (None for an empty input).

    Returns:
        The values, one per argument; None for an input left empty.
    """
    columns = st.columns(min(len(problem.arguments), 4))
    values: list[ArgumentValue | None] = []
    for index, (argument, start) in enumerate(zip(problem.arguments, initial, strict=True)):
        column = columns[index % len(columns)]
        key = argument_key(argument.name)
        if argument.type == "boolean":
            values.append(column.checkbox(argument.name, value=bool(start), key=key))
        elif argument.type == "integer":
            value = column.number_input(argument.name, value=start, step=1, key=key)
            values.append(None if value is None else int(value))
        else:
            value = column.number_input(argument.name, value=start, key=key)
            values.append(None if value is None else float(value))
    return values


def render_problem_rows(
    table: pd.DataFrame,
    catalogue: Mapping[str, Problem],
    state_key: str,
    key_prefix: str,
    working_directory: Path,
) -> pd.DataFrame:
    """Edit the rows of a table of problems, one bordered block per problem.

    Each block offers the problem's arguments (a checkbox and an input per argument when the
    catalogue describes it, a text when it does not), its reference front and, when the table has
    an "evaluations" column, its evaluation budget. The edited rows are kept under `state_key`.

    Args:
        table: The rows, with the columns "problem", "arguments" and "reference_front" (and
            "evaluations").
        catalogue: The problem catalogue (empty when the jar has none).
        state_key: The session state key the page keeps the rows under.
        key_prefix: Prefixes the widget keys, so that pages do not share them.
        working_directory: The directory the fronts' paths are relative to.

    Returns:
        The edited rows, with the columns of the table.
    """
    prefix = f"{key_prefix}_{_form_version(state_key)}"
    with_evaluations = "evaluations" in table.columns
    rows = [
        _render_row(row, catalogue.get(str(row["problem"])), prefix, working_directory)
        if not with_evaluations
        else _render_row_with_evaluations(
            row, catalogue.get(str(row["problem"])), prefix, working_directory
        )
        for _, row in table.iterrows()
    ]
    edited = pd.DataFrame(rows, columns=list(table.columns))
    st.session_state[state_key] = edited
    st.session_state[_digest_key(state_key)] = rows_digest(edited)
    return edited


def _render_row(
    row: pd.Series, problem: Problem | None, prefix: str, working_directory: Path
) -> dict[str, object]:
    name = str(row["problem"])
    with st.container(border=True):
        st.markdown(f"**{name}**")
        arguments, objectives = _render_arguments(
            name, problem, _text(row.get("arguments")), prefix
        )
        front = _render_front(
            name, _text(row.get("reference_front")), objectives, prefix, working_directory
        )
    return {"problem": name, "arguments": arguments, "reference_front": front}


def _render_row_with_evaluations(
    row: pd.Series, problem: Problem | None, prefix: str, working_directory: Path
) -> dict[str, object]:
    name = str(row["problem"])
    with st.container(border=True):
        st.markdown(f"**{name}**")
        arguments, objectives = _render_arguments(
            name, problem, _text(row.get("arguments")), prefix
        )
        front_column, evaluations_column = st.columns([3, 1])
        with front_column:
            front = _render_front(
                name, _text(row.get("reference_front")), objectives, prefix, working_directory
            )
        evaluations = evaluations_column.number_input(
            "Evaluations",
            min_value=1,
            step=100,
            value=_evaluations(row.get("evaluations")),
            key=f"{prefix}_evaluations_{name}",
            help="The base-level evaluations of each run on this problem.",
        )
    return {
        "problem": name,
        "arguments": arguments,
        "reference_front": front,
        "evaluations": evaluations,
    }


def _render_arguments(
    name: str, problem: Problem | None, text: str, prefix: str
) -> tuple[str, int | None]:
    """The arguments of a row, as a text, and the problem's number of objectives with them."""
    if problem is None:
        typed = st.text_input(
            "Arguments", value=text, key=f"{prefix}_arguments_{name}", help=UNTYPED_ARGUMENTS_HELP
        )
        return typed.strip(), None
    if not problem.arguments:
        return "", problem.number_of_objectives
    custom = st.checkbox(
        "Set its arguments",
        value=bool(text),
        key=f"{prefix}_set_arguments_{name}",
        help=defaults_help(problem),
    )
    if not custom:
        return "", problem.number_of_objectives
    values = render_argument_inputs(
        problem,
        lambda argument: f"{prefix}_argument_{name}_{argument}",
        initial_argument_values(problem, text),
    )
    return argument_values_text(values), objectives_of(problem, values)


def _render_front(
    name: str, current: str, objectives: int | None, prefix: str, working_directory: Path
) -> str:
    """The reference front of a row: a selector of the problem's fronts, or a file to type.

    The selector's key carries the number of objectives, so that changing them starts it again
    from the front that has them.
    """
    candidates = reference_front_candidates(name)
    if not candidates:
        return _render_front_file(name, current, prefix)
    options = [*candidates, OTHER_FRONT]
    start = (
        OTHER_FRONT
        if current and current not in candidates
        else initial_front(candidates, current, objectives, working_directory)
    )
    choice = st.selectbox(
        "Reference front",
        options,
        index=options.index(start) if start is not None else None,
        placeholder="Choose the reference front",
        key=f"{prefix}_front_{name}_{objectives}",
        help=FRONT_HELP,
    )
    if choice == OTHER_FRONT:
        return _render_front_file(name, "" if current in candidates else current, prefix)
    return choice or ""


def _render_front_file(name: str, current: str, prefix: str) -> str:
    typed = st.text_input(
        "Reference front file",
        value=current,
        key=f"{prefix}_front_file_{name}",
        help=FRONT_FILE_HELP,
    )
    return typed.strip()


def _form_version(state_key: str) -> int:
    """The version of the rows' widgets: a new one when the rows were not written by the form."""
    version_key = f"{state_key}_version"
    stored = st.session_state.get(state_key)
    if stored is None or rows_digest(stored) != st.session_state.get(_digest_key(state_key)):
        st.session_state[version_key] = st.session_state.get(version_key, 0) + 1
    return st.session_state[version_key]


def _digest_key(state_key: str) -> str:
    return f"{state_key}_digest"


def _has_objectives(front_file: Path, objectives: int | None) -> bool:
    if objectives is None:
        return True
    dimension = reference_front_dimension(front_file)
    return dimension is None or dimension == objectives


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _evaluations(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
