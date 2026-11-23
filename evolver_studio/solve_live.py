"""The front of a solve run while it runs, as Evolver writes it to CURRENT_FRONT.csv.

Evolver overwrites that file in the run's output directory every `frontFrequency` evaluations (see
`SolveRequest.front_frequency`) and moves a finished copy over it, so a reader never sees a
half-written one. Streamlit-agnostic, so it can be tested without a UI.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CURRENT_FRONT_FILE = "CURRENT_FRONT.csv"
# The file is Run,Evaluations,NonDominated,F1,...,Fm[,Violation]: all but the Fs are not objectives.
_NOT_OBJECTIVES = ("Run", "Evaluations", "NonDominated", "Violation")


@dataclass(slots=True, frozen=True)
class CurrentFront:
    """The solutions of the run in progress, at one moment.

    Attributes:
        run: The number of the independent run (from 1).
        evaluations: The evaluations of that run when the file was written.
        objectives: One row per solution, with columns f1, f2, ... as `read_front` returns them.
    """

    run: int
    evaluations: int
    objectives: pd.DataFrame


def read_current_front(output_directory: Path) -> CurrentFront | None:
    """Read the front of the run in progress.

    Args:
        output_directory: The run's output directory.

    Returns:
        The front, or None while there is none (the run has not written one yet, or was not asked
        to) or the file cannot be read.
    """
    try:
        table = pd.read_csv(output_directory / CURRENT_FRONT_FILE)
        objectives = table[[name for name in table.columns if name not in _NOT_OBJECTIVES]]
        if table.empty or objectives.empty:
            return None
        objectives.columns = [f"f{index}" for index in range(1, objectives.shape[1] + 1)]
        return CurrentFront(
            run=int(table["Run"].iloc[0]),
            evaluations=int(table["Evaluations"].iloc[0]),
            objectives=objectives.reset_index(drop=True),
        )
    except (OSError, ValueError, KeyError, pd.errors.EmptyDataError):
        return None
