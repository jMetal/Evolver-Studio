"""The state of the Run algorithm form, and how to restore it from a past run's request.

The widget keys are defined here, once, because restoring a run means setting them: the page and
this module must agree on them. This module has no Streamlit dependency, so the translation from a
request to form state is testable on its own.
"""

from collections.abc import Collection, Iterable

from evolver_studio.catalogue import BaseAlgorithm
from evolver_studio.configuration import parse_configuration

NO_REFERENCE_FRONT = "(none: no indicators)"

PROBLEM_KEY = "solve_problem"
ALGORITHM_KEY = "solve_algorithm"
WEIGHT_VECTORS_KEY = "solve_weight_vectors"
POPULATION_KEY = "solve_population"
POPULATION_CHOICE_KEY = "solve_population_choice"
EVALUATIONS_KEY = "solve_evaluations"
RUNS_KEY = "solve_runs"
FIX_SEED_KEY = "solve_fix_seed"
SEED_KEY = "solve_seed"
# A new version gives the configuration widgets new keys, so they start from their values again.
CONFIGURATION_VERSION_KEY = "solve_configuration_version"
# The configuration a past run used, for the form to start from:
# {"algorithm", "encoding", "values"}.
LOADED_CONFIGURATION_KEY = "solve_loaded_configuration"


def reference_front_key(problem: str) -> str:
    """The key of the reference front selector of a problem."""
    return f"solve_reference_front_{problem}"


def encoding_key(algorithm_name: str) -> str:
    """The key of the encoding selector of an algorithm."""
    return f"solve_encoding_{algorithm_name}"


def indicators_key(has_reference_front: bool) -> str:
    """The key of the indicators selector, which differs with and without a reference front."""
    return f"solve_indicators_{has_reference_front}"


def form_state_from_request(
    request: dict,
    algorithms: Iterable[BaseAlgorithm],
    problem_names: Collection[str],
    configuration_version: int,
) -> dict[str, object] | None:
    """Translate a past run's request into the values of the form's widgets.

    Args:
        request: The request file's content of a past run, as Evolver reads it.
        algorithms: The algorithms the page offers.
        problem_names: The problems the page offers.
        configuration_version: The current version of the configuration widgets' keys.

    Returns:
        The session state to set, widget keys to values, plus `LOADED_CONFIGURATION_KEY` and a new
        `CONFIGURATION_VERSION_KEY`; or None if the request cannot be restored (an algorithm or a
        problem the page does not offer, or a request that lacks what the form needs).
    """
    algorithm = next(
        (a for a in algorithms if a.registry_name == request.get("algorithmName")), None
    )
    problem = request.get("problem")
    if algorithm is None or problem not in problem_names:
        return None
    try:
        encoding = request["encoding"]
        configuration = parse_configuration(request["configuration"])
        state: dict[str, object] = {
            PROBLEM_KEY: problem,
            ALGORITHM_KEY: algorithm.name,
            POPULATION_KEY: request["populationSize"],
            POPULATION_CHOICE_KEY: request["populationSize"],
            EVALUATIONS_KEY: request["maxEvaluations"],
            RUNS_KEY: request.get("numberOfIndependentRuns", 1),
        }
    except (KeyError, ValueError):
        return None
    reference_front = request.get("referenceFrontFileName")
    state[reference_front_key(problem)] = reference_front or NO_REFERENCE_FRONT
    state[indicators_key(bool(reference_front))] = list(request.get("indicatorNames") or [])
    if len(algorithm.runnable_encodings) > 1:
        state[encoding_key(algorithm.name)] = encoding
    weight_vectors = (request.get("extraConfig") or {}).get("weightVectorFilesDirectory")
    if weight_vectors:
        state[WEIGHT_VECTORS_KEY] = weight_vectors
    state[FIX_SEED_KEY] = request.get("seed") is not None
    if request.get("seed") is not None:
        state[SEED_KEY] = request["seed"]
    state[LOADED_CONFIGURATION_KEY] = {
        "algorithm": algorithm.name,
        "encoding": encoding,
        "values": configuration,
    }
    state[CONFIGURATION_VERSION_KEY] = configuration_version + 1
    return state
