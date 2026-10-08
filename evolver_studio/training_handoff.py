"""Taking a configuration a training found to the pages that use it.

The Analysis page shows what a training found; from there a configuration goes to Validation, to be
compared with other algorithms on a set of problems, or to Run algorithm, to be run on one problem.
Both pages are forms whose widgets are filled through the session state: this module says what to
put in it, as plain data, so that it can be tested without a browser.
"""

from collections.abc import Mapping

import pandas as pd

from evolver_studio.catalogue import QUALITY_INDICATORS
from evolver_studio.problem_catalogue import (
    Problem,
)
from evolver_studio.solve_form import form_state_from_request
from evolver_studio.training_results import TrainedConfiguration
from evolver_studio.training_runs import FinishedTraining
from evolver_studio.validation import base_algorithm
from evolver_studio.validation_form import (
    ENCODING_KEY,
    INDICATORS_KEY,
    PASTED_ALGORITHM_KEY,
    PASTED_CONFIGURATION_KEY,
    POPULATION_KEY,
    PROBLEM_COLUMNS,
    PROBLEM_ROWS_KEY,
    TUNED_PASTED,
    TUNED_SOURCE_KEY,
    defaults_key,
    problems_key,
    tuned_name_key,
)

DEFAULT_MAX_EVALUATIONS = 25000
DEFAULT_POPULATION_SIZE = 100


def validation_form_state(
    training: FinishedTraining,
    configuration: TrainedConfiguration,
) -> dict[str, object] | None:
    """The values of the Validation form for validating a configuration of a training.

    The tuned configuration is the one given; the algorithm to compare it with is the base
    algorithm's own default configuration, when it has a single one; and the population size and
    the indicators are the training's. The problems are left empty: the user chooses them (those
    of the training, or any others).

    Args:
        training: The training.
        configuration: One of its configurations.

    Returns:
        The session state to set, widget keys to values; None when the base algorithm is not one
        Validation knows.
    """
    algorithm = base_algorithm(training.algorithm)
    if algorithm is None:
        return None
    state: dict[str, object] = {
        ENCODING_KEY: training.encoding,
        TUNED_SOURCE_KEY: TUNED_PASTED,
        PASTED_ALGORITHM_KEY: algorithm.name,
        PASTED_CONFIGURATION_KEY: configuration.configuration,
        tuned_name_key(training.algorithm): f"{algorithm.name} (tuned)",
    }
    if len(algorithm.default_configurations.get(training.encoding, ())) == 1:
        state[defaults_key(training.encoding)] = [algorithm.name]
    if training.population_size:
        state[POPULATION_KEY] = training.population_size
    known = {indicator.registry_name for indicator in QUALITY_INDICATORS}
    if training.indicators and set(training.indicators) <= known:
        state[INDICATORS_KEY] = list(training.indicators)
    state[problems_key(training.encoding)] = []
    state[PROBLEM_ROWS_KEY] = pd.DataFrame(columns=list(PROBLEM_COLUMNS))
    return state


def solve_form_state(
    training: FinishedTraining,
    configuration: TrainedConfiguration,
    problem_index: int,
    problem_names: list[str],
    configuration_version: int,
    catalogue: Mapping[str, Problem] | None,
) -> dict[str, object] | None:
    """The values of the Run algorithm form for running a configuration of a training.

    Args:
        training: The training.
        configuration: One of its configurations.
        problem_index: Which of the training's problems to run it on.
        problem_names: The problems Run algorithm offers.
        configuration_version: The current version of the form's configuration widgets' keys.
        catalogue: The problem catalogue, or None when the jar has none.

    Returns:
        The session state to set; None when the form cannot take it (an algorithm or a problem
        Run algorithm does not offer).
    """
    algorithm = base_algorithm(training.algorithm)
    if algorithm is None or not 0 <= problem_index < len(training.problem_specs):
        return None
    request: dict[str, object] = {
        "algorithmName": training.algorithm,
        "encoding": training.encoding,
        "populationSize": training.population_size or DEFAULT_POPULATION_SIZE,
        "configuration": configuration.configuration,
        "problem": training.problem_specs[problem_index],
        "maxEvaluations": _at(training.evaluations, problem_index, DEFAULT_MAX_EVALUATIONS),
        "numberOfIndependentRuns": 1,
        "indicatorNames": list(training.indicators),
        "extraConfig": training.extra_config or {},
    }
    front = _at(training.reference_fronts, problem_index, None)
    if front:
        request["referenceFrontFileName"] = front
    return form_state_from_request(
        request, [algorithm], problem_names, configuration_version, catalogue
    )


def _at(values: tuple, index: int, default):
    return values[index] if index < len(values) else default
