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
    arguments_text,
    number_of_objectives,
    parse_arguments_text,
)
from evolver_studio.problems import reference_front_dimension, suggested_reference_front
from evolver_studio.validation import StudyProblem, ValidationStudy, base_algorithm

PROBLEM_COLUMNS = ("problem", "arguments", "reference_front")

# The widget keys of the Validation page, defined here because restoring a study means setting
# them: the page and this module must agree on them.
ENCODING_KEY = "validation_encoding"
PROBLEM_ROWS_KEY = "validation_problem_rows"
TUNED_SOURCE_KEY = "validation_tuned_source"
PASTED_ALGORITHM_KEY = "validation_pasted_algorithm"
PASTED_CONFIGURATION_KEY = "validation_pasted_configuration"
POPULATION_KEY = "validation_population"
EVALUATIONS_KEY = "validation_evaluations"
RUNS_KEY = "validation_runs"
SEED_KEY = "validation_seed"
INDICATORS_KEY = "validation_indicators"
ADJUSTED_SUFFIX = " (adjusted)"
TUNED_FROM_TRAINING = "A training run"
TUNED_PASTED = "A configuration I paste"


def problems_key(encoding: str) -> str:
    """The key of the problems selector of an encoding."""
    return f"validation_problems_{encoding}"


def defaults_key(encoding: str) -> str:
    """The key of the selector of the algorithms compared with their default configuration."""
    return f"validation_defaults_{encoding}"


def adjusted_key(encoding: str, name: str) -> str:
    """The key of the values an algorithm compared with starts from, when they are not its
    default configuration's (a study run again with that algorithm adjusted)."""
    return f"validation_adjusted_{encoding}_{name}"


def adjusted_name(name: str) -> str:
    """The name, in a study, of an algorithm whose default configuration was adjusted."""
    return f"{name}{ADJUSTED_SUFFIX}"


def tuned_name_key(registry_name: str) -> str:
    """The key of the name of the tuned configuration, which depends on its algorithm."""
    return f"validation_tuned_name_{registry_name}"


def form_state_from_study(study: ValidationStudy) -> dict[str, object]:
    """Translate a study into the values of the Validation form's widgets.

    The tuned configuration (the pivot) is restored as a pasted one: the training run it came from
    may be gone, and its configuration is all the study needs. An algorithm compared with an
    adjusted configuration is chosen again by its name, and starts from that configuration.

    Args:
        study: The study.

    Returns:
        The session state to set, widget keys to values.

    Raises:
        ValueError: If the pivot is not one of the study's contenders or its algorithm is unknown.
    """
    pivot = next((c for c in study.contenders if c.name == study.pivot), None)
    algorithm = base_algorithm(pivot.algorithm) if pivot is not None else None
    if pivot is None or algorithm is None:
        raise ValueError("the study's pivot is not one of its contenders")
    others = []
    adjusted = {}
    for contender in study.contenders:
        if contender.name == study.pivot:
            continue
        name = contender.name.removesuffix(ADJUSTED_SUFFIX)
        others.append(name)
        # None leaves an algorithm adjusted in the form before at its default configuration.
        adjusted[adjusted_key(study.encoding, name)] = (
            parse_configuration(contender.configuration) if name != contender.name else None
        )
    rows = [
        {
            "problem": problem.name,
            "arguments": arguments_text(problem.arguments),
            "reference_front": problem.reference_front,
        }
        for problem in study.problems
    ]
    return {
        ENCODING_KEY: study.encoding,
        problems_key(study.encoding): [problem.name for problem in study.problems],
        PROBLEM_ROWS_KEY: pd.DataFrame(rows, columns=list(PROBLEM_COLUMNS)),
        TUNED_SOURCE_KEY: TUNED_PASTED,
        PASTED_ALGORITHM_KEY: algorithm.name,
        PASTED_CONFIGURATION_KEY: pivot.configuration,
        tuned_name_key(pivot.algorithm): pivot.name,
        defaults_key(study.encoding): others,
        POPULATION_KEY: study.population_size,
        EVALUATIONS_KEY: study.max_evaluations,
        RUNS_KEY: study.runs,
        SEED_KEY: study.seed,
        INDICATORS_KEY: list(study.indicators),
        **adjusted,
    }


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
                "reference_front": suggested_reference_front(name, objectives, working_directory),
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
