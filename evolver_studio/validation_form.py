"""The inputs of the Validation page that are worth testing apart from it.

The problems of a study are an editable table, one row per problem, with its constructor arguments
(optional, as a comma-separated text) and its reference front; the contenders are checked against
the study's encoding and parameter space.
"""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from evolver_studio.configuration import (
    complete_values,
    parse_configuration,
    values_outside_the_space,
)
from evolver_studio.parameter_space import parse_parameter_space
from evolver_studio.problem_catalogue import (
    Problem,
    number_of_objectives,
    parse_arguments_text,
)
from evolver_studio.problems import (
    default_reference_front,
    reference_front_candidates,
    reference_front_dimension,
    reference_front_for_objectives,
)
from evolver_studio.validation import StudyProblem

PROBLEM_COLUMNS = ("problem", "arguments", "reference_front")


def problem_table(
    names: list[str],
    catalogue: Mapping[str, Problem],
    working_directory: Path,
    previous: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The table of the study's problems, a row per chosen problem.

    A problem that was already in the table keeps its row (what the user edited); a new one gets
    the reference front that fits it, when there is no doubt about which, and no arguments.

    Args:
        names: The chosen problems, in order.
        catalogue: The problem catalogue.
        working_directory: The directory the reference fronts' paths are relative to.
        previous: The table as it was, with the user's edits.

    Returns:
        The table, with the columns `PROBLEM_COLUMNS`.
    """
    kept = (
        {row["problem"]: row for _, row in previous.iterrows()}
        if previous is not None and not previous.empty
        else {}
    )
    rows = []
    for name in names:
        if name in kept:
            rows.append({column: kept[name][column] for column in PROBLEM_COLUMNS})
            continue
        problem = catalogue.get(name)
        objectives = number_of_objectives(problem, ()) if problem is not None else None
        rows.append(
            {
                "problem": name,
                "arguments": "",
                "reference_front": _default_front(name, objectives, working_directory),
            }
        )
    return pd.DataFrame(rows, columns=list(PROBLEM_COLUMNS))


def parse_problem_table(
    table: pd.DataFrame,
    catalogue: Mapping[str, Problem],
    encoding: str,
    working_directory: Path,
) -> tuple[tuple[StudyProblem, ...], list[str]]:
    """Read the edited table of the study's problems, and check it.

    Args:
        table: The table, with the columns `PROBLEM_COLUMNS`.
        catalogue: The problem catalogue.
        encoding: The encoding of the study.
        working_directory: The directory the reference fronts' paths are relative to.

    Returns:
        The problems, and the messages of what is wrong (empty when nothing is): a problem the
        catalogue does not know or of another encoding, arguments that do not fit it, and a
        reference front that is missing or has another number of objectives.
    """
    problems: list[StudyProblem] = []
    errors: list[str] = []
    for _, row in table.iterrows():
        name = str(row["problem"])
        described = catalogue.get(name)
        if described is None:
            errors.append(f"{name} is not a problem Evolver knows.")
            continue
        if described.encoding != encoding:
            errors.append(f"{name} is a {described.encoding} problem, not a {encoding} one.")
            continue
        arguments = parse_arguments_text(described, _text(row.get("arguments")))
        if isinstance(arguments, str):
            errors.append(arguments)
            continue
        front = _text(row.get("reference_front"))
        error = _front_error(
            name, front, number_of_objectives(described, arguments), working_directory
        )
        if error is not None:
            errors.append(error)
            continue
        problems.append(StudyProblem(name, tuple(arguments), front))
    return tuple(problems), errors


def configuration_errors(configuration: str, parameter_space_text_: str, label: str) -> list[str]:
    """Check a configuration against a parameter space.

    Args:
        configuration: The configuration, "--parameter value ...".
        parameter_space_text_: The parameter space the contender is run with (a YAML text).
        label: What to call the contender in the messages.

    Returns:
        The messages of what is wrong; empty when the configuration fits the space. Parameters
        the space does not have are not an error (Evolver ignores or rejects them as it sees
        fit); values it does not allow are.
    """
    try:
        values = parse_configuration(configuration)
    except ValueError as error:
        return [f"{label}: {error}"]
    if not values:
        return [f"{label}: the configuration is empty."]
    parameters = parse_parameter_space(parameter_space_text_)
    outside = values_outside_the_space(parameters, complete_values(parameters, values))
    if outside:
        return [f"{label}: the space does not allow the value of {', '.join(outside)}."]
    return []


def _default_front(name: str, objectives: int | None, working_directory: Path) -> str:
    candidates = reference_front_candidates(name)
    chosen = default_reference_front(candidates)
    if chosen is None and objectives is not None:
        chosen = reference_front_for_objectives(candidates, objectives, working_directory)
    return chosen or ""


def _front_error(
    name: str, front: str, objectives: int | None, working_directory: Path
) -> str | None:
    if not front:
        return f"{name} needs a reference front: the indicators use it."
    if not (working_directory / front).is_file():
        return f"{name}: the reference front {front} does not exist."
    dimension = reference_front_dimension(working_directory / front)
    if objectives is not None and dimension is not None and dimension != objectives:
        return f"{name}: the reference front has {dimension} objectives, the problem {objectives}."
    return None


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()
