"""Trainings the tutorials prepare in the Training page's form, for the reader to launch there.

A preset is the value of every widget of the form that describes a training, by widget key: all
of them are set, so that the page shows the preset whatever was chosen before. `open_training`
sets them and goes to the page.
"""

import streamlit as st

from evolver_studio.training_choices import (
    DEFAULT_INDEPENDENT_RUNS,
    DEFAULT_INDICATORS,
    DEFAULT_POPULATION_SIZE,
)

# The Training page keeps the rows of its training set table under this key; dropping them makes
# the table start from the problems chosen, with their default budgets and fronts.
TRAINING_SET_ROWS_KEY = "training_set_rows"
STOP_BY_EVALUATIONS = "Evaluations"
STOP_BY_TIME = "Computing time"


def training_state(
    algorithm: str = "NSGA-II",
    encoding: str = "Double",
    problems: tuple[str, ...] = ("ZDT4",),
    meta_algorithm: str = "NSGA-II",
    meta_encoding: str = "Flat",
    meta_evaluations: int = 2000,
    meta_minutes: float | None = None,
    meta_population: int = 50,
    update_every: int = 100,
    write_population: bool = True,
    indicators: tuple[str, ...] = DEFAULT_INDICATORS,
    independent_runs: int = DEFAULT_INDEPENDENT_RUNS,
    population_size: int = DEFAULT_POPULATION_SIZE,
) -> dict[str, object]:
    """The values of the Training form for a training, by widget key.

    The defaults are the training of Evolver's tutorial E3 (NSGA-II tuned for ZDT4 by NSGA-II).

    Args:
        algorithm: The base algorithm.
        encoding: Its encoding.
        problems: The training problems.
        meta_algorithm: The meta-optimizer.
        meta_encoding: "Flat" or "Tree".
        meta_evaluations: The meta-evaluations, when it stops by evaluations.
        meta_minutes: The computing time, when it stops by time; None to stop by evaluations.
        meta_population: The meta-optimizer's population.
        update_every: How often the training writes a checkpoint, a multiple of the population.
        write_population: Whether it also writes the meta-optimizer's population.
        indicators: The meta-objectives.
        independent_runs: The runs that measure each configuration.
        population_size: The base algorithm's population.

    Returns:
        The session state to set.
    """
    state: dict[str, object] = {
        "train_base_algorithm": algorithm,
        f"train_encoding_{algorithm}": encoding,
        f"train_problems_{encoding}": list(problems),
        "train_meta_algorithm": meta_algorithm,
        f"train_meta_encoding_{meta_algorithm}": meta_encoding,
        "train_meta_stop_by": STOP_BY_EVALUATIONS if meta_minutes is None else STOP_BY_TIME,
        "train_meta_evals": meta_evaluations,
        f"train_meta_population_{meta_algorithm}": meta_population,
        f"train_update_every_{meta_population}": update_every,
        "train_write_population": write_population,
        "train_indicators": list(indicators),
        "train_independent_runs": independent_runs,
        f"train_population_{algorithm}": population_size,
    }
    if meta_minutes is not None:
        state["train_meta_minutes"] = meta_minutes
    return state


def open_training(state: dict[str, object]) -> None:
    """Fill the Training form with a preset and go to the page.

    Args:
        state: What `training_state` returns.
    """
    st.session_state.pop(TRAINING_SET_ROWS_KEY, None)
    st.session_state.update(state)
    st.switch_page("pages/training.py")
