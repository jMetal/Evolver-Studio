"""The feasibility of the configurations of a training on problems with constraints.

When a training problem has constraints, Evolver writes FEASIBILITY.csv next to INDICATORS.csv (and
POPULATION_FEASIBILITY.csv when it writes the population): in how many of its runs each
configuration found no feasible solution. The files ending in _BY_PROBLEM.csv tell it by training
problem. A training without constraints writes none of them.

Nothing here depends on Streamlit.
"""

import io
from pathlib import Path

import pandas as pd

FEASIBILITY_FILE = "FEASIBILITY.csv"
POPULATION_PREFIX = "POPULATION_"


def read_feasibility(
    output_directory: Path, population: bool = False, by_problem: bool = False
) -> pd.DataFrame | None:
    """Read how feasible the configurations of each checkpoint are.

    Args:
        output_directory: The training's output directory.
        population: Whether those of the whole population, instead of the front's.
        by_problem: Whether a row per configuration and problem, instead of per configuration.

    Returns:
        The table (`Evaluation`, `SolutionId`, `FailedRuns`, `Runs`...), without the line Evolver
        may be in the middle of writing; None when there is no such file, or it has no row yet.
    """
    name = FEASIBILITY_FILE
    if by_problem:
        name = name.replace(".csv", "_BY_PROBLEM.csv")
    if population:
        name = POPULATION_PREFIX + name
    try:
        text = (output_directory / name).read_text()
        table = pd.read_csv(io.StringIO(text[: text.rfind("\n") + 1]))
    except (OSError, pd.errors.EmptyDataError):
        return None
    return None if table.empty else table


def feasible_by_checkpoint(table: pd.DataFrame) -> pd.DataFrame:
    """Count, at each checkpoint, the configurations without failed runs.

    Args:
        table: What `read_feasibility` returns, a row per configuration.

    Returns:
        One row per checkpoint: `Evaluation`, `configurations` and `without_failures`.
    """
    groups = table.assign(without=table["FailedRuns"] == 0).groupby("Evaluation", as_index=False)
    return groups.agg(configurations=("without", "size"), without_failures=("without", "sum"))


def failures_by_problem(table: pd.DataFrame) -> pd.DataFrame:
    """Tell, at the last checkpoint, on which problems the configurations fail.

    Args:
        table: What `read_feasibility` returns by problem.

    Returns:
        One row per problem: `problem`, `configurations`, how many of them have `failing` runs on
        it, and the `failed_runs` and `runs` of all of them.
    """
    last = table[table["Evaluation"] == table["Evaluation"].max()]
    groups = last.assign(failing=last["FailedRuns"] > 0).groupby(["ProblemId", "Problem"])
    summary = groups.agg(
        configurations=("failing", "size"),
        failing=("failing", "sum"),
        failed_runs=("FailedRuns", "sum"),
        runs=("Runs", "sum"),
    )
    return summary.reset_index().drop(columns="ProblemId").rename(columns={"Problem": "problem"})


def failed_configurations(table: pd.DataFrame, evaluation: int) -> set[int]:
    """The `SolutionId` of the configurations of a checkpoint that have failed runs.

    Args:
        table: What `read_feasibility` returns, a row per configuration.
        evaluation: The checkpoint's meta-evaluations.
    """
    rows = table[(table["Evaluation"] == evaluation) & (table["FailedRuns"] > 0)]
    return set(rows["SolutionId"])
