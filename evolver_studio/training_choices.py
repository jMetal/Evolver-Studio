"""What a training chooses about its base level besides the algorithm and the training set.

The meta-objectives (the quality indicators the meta-optimizer minimizes), how many independent
runs measure each configuration, and the base algorithm's population size, with the checks Evolver
would make at launch: two meta-objectives, Spread only on problems with two objectives, and, for
an algorithm that reads weight vectors (MOEA/D, RVEA), a weight vector file for the population
size and the objectives of every training problem.

Nothing here depends on Streamlit.
"""

from dataclasses import dataclass
from pathlib import Path

from evolver_studio.catalogue import QualityIndicator, evolver_indicators
from evolver_studio.weight_vectors import available_population_sizes

DEFAULT_INDICATORS = ("Epsilon", "NormalizedHypervolume")
DEFAULT_POPULATION_SIZE = 100
DEFAULT_INDEPENDENT_RUNS = 1
META_OBJECTIVES = 2
BI_OBJECTIVE_ONLY = ("Spread",)


@dataclass(frozen=True)
class BaseLevelChoices:
    """What the base level of a training uses, besides its algorithm and training set.

    Attributes:
        population_size: The base algorithm's population size.
        independent_runs: How many runs measure each configuration (its indicator values are
            aggregated over them).
        indicators: The meta-objectives, by registry name.
    """

    population_size: int = DEFAULT_POPULATION_SIZE
    independent_runs: int = DEFAULT_INDEPENDENT_RUNS
    indicators: tuple[str, ...] = DEFAULT_INDICATORS


def training_indicators() -> tuple[QualityIndicator, ...]:
    """The indicators a training can minimize: those Evolver computes (all are minimized)."""
    return tuple(i for i in evolver_indicators() if not i.maximized)


def indicator_errors(indicators: tuple[str, ...], objectives: list[int | None]) -> list[str]:
    """What is wrong with the meta-objectives of a training.

    Args:
        indicators: The meta-objectives, by registry name.
        objectives: The number of objectives of each training problem (None when unknown).

    Returns:
        The errors; empty when they are right.
    """
    errors = []
    if len(indicators) != META_OBJECTIVES:
        errors.append(f"Choose {META_OBJECTIVES} meta-objectives (you chose {len(indicators)}).")
    for name in indicators:
        if name in BI_OBJECTIVE_ONLY and any(m is not None and m != 2 for m in objectives):
            errors.append(
                f"{name} is defined only for problems with two objectives: use "
                "GeneralizedSpread instead."
            )
    return errors


def population_errors(
    population_size: int,
    reads_weight_vectors: bool,
    objectives: list[int | None],
    weight_vectors_directory: Path,
) -> list[str]:
    """What is wrong with the base algorithm's population size for the training problems.

    Args:
        population_size: The population size.
        reads_weight_vectors: Whether the algorithm reads weight vectors (MOEA/D, RVEA).
        objectives: The number of objectives of each training problem (None when unknown).
        weight_vectors_directory: Where the weight vector files are.

    Returns:
        The errors; empty when every problem has a weight vector file for the size, or the
        algorithm reads none.
    """
    if not reads_weight_vectors:
        return []
    errors = []
    for count in sorted({m for m in objectives if m is not None}):
        sizes = available_population_sizes(weight_vectors_directory, count)
        if population_size not in sizes:
            offered = ", ".join(map(str, sizes)) or "none"
            errors.append(
                f"No weight vector file for a population of {population_size} and {count} "
                f"objectives (W{count}D_{population_size}.dat); the sizes there are: {offered}."
            )
    return errors
