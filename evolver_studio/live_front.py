"""Throttled live preview of the meta-level indicator front during a run.

Deliberately Streamlit-agnostic: builds Plotly figures from INDICATORS.csv's
growing checkpoint history, so it can be unit-tested without a UI.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from evolver_studio.results import (
    deduplicate_consecutive_checkpoints,
    latest_checkpoint_evaluation,
    load_indicators,
)


def should_render_checkpoint(
    latest_evaluation: int, last_rendered_evaluation: int | None, update_every_evaluations: int
) -> bool:
    """Decide whether a newly seen checkpoint is due for a redraw.

    Args:
        latest_evaluation: Evaluation count of the most recent checkpoint seen.
        last_rendered_evaluation: Evaluation count last rendered, or None if none yet.
        update_every_evaluations: Minimum evaluations that must pass between redraws.

    Returns:
        True if the checkpoint should be rendered now.
    """
    if last_rendered_evaluation is None:
        return True
    return latest_evaluation - last_rendered_evaluation >= update_every_evaluations


def build_front_figure(history: pd.DataFrame) -> go.Figure:
    """Plot every checkpoint's non-dominated archive as one scatter, colored by evaluation.

    A single checkpoint's front can be as small as one point — showing every
    checkpoint seen so far together, colored on a gradient from early
    (evaluation count) to late, makes the front's evolution visible at a
    glance instead of one sparse snapshot at a time.

    Args:
        history: Rows spanning one or more checkpoints, typically already
            thinned by `deduplicate_consecutive_checkpoints`. Columns are
            Evaluation, SolutionId, <indicator1>, <indicator2>, ...

    Returns:
        A Plotly scatter figure, axes taken from the history's own columns.
    """
    x_axis, y_axis = history.columns[2], history.columns[3]
    return px.scatter(
        history,
        x=x_axis,
        y=y_axis,
        color="Evaluation",
        color_continuous_scale="Viridis",
        title="Indicator front evolution",
    )


@dataclass(slots=True, frozen=True)
class LiveFrontUpdate:
    """Result of one poll of INDICATORS.csv.

    Attributes:
        changed: Whether the file grew since the previous poll — independent
            of the render throttle, useful to drive an adaptive poll interval.
        updated: Whether a new checkpoint was folded into `last_history` this
            poll (i.e. the render throttle allowed it through).
    """

    changed: bool
    updated: bool


class LiveFrontRenderer:
    """Tracks the deduplicated checkpoint history available for display, throttled.

    Deliberately doesn't build a Plotly figure itself: how much of the
    history to actually plot (e.g. only the last N checkpoints, to keep
    early, large indicator values from swamping later convergence detail) is
    a display choice left to the caller, via `build_front_figure` and
    `last_n_checkpoints`.
    """

    def __init__(self, update_every_evaluations: int) -> None:
        self._update_every_evaluations = update_every_evaluations
        self._last_rendered_evaluation: int | None = None
        self._last_file_size = -1
        self._last_history: pd.DataFrame | None = None

    @property
    def last_history(self) -> pd.DataFrame | None:
        """The deduplicated checkpoint history as of the last due poll.

        A fragment fully replaces its own contents each time it reruns, so a
        caller must redraw from this every tick (not only when `poll` reports
        an update) to avoid the preview flickering away between throttled
        redraws.
        """
        return self._last_history

    @property
    def last_rendered_evaluation(self) -> int | None:
        """The most recent checkpoint's evaluation count in `last_history`."""
        return self._last_rendered_evaluation

    def poll(self, indicators_csv: Path) -> LiveFrontUpdate:
        """Check INDICATORS.csv once and, if due, fold new data into `last_history`.

        Args:
            indicators_csv: Path to the run's (still-growing) INDICATORS.csv.

        Returns:
            Whether the file grew since the last poll, and whether a new
            checkpoint was folded into `last_history` this time.
        """
        if not self._file_grew(indicators_csv):
            return LiveFrontUpdate(changed=False, updated=False)
        history = load_indicators(indicators_csv)
        latest = latest_checkpoint_evaluation(history)
        return LiveFrontUpdate(changed=True, updated=self._update_if_due(history, latest))

    def _update_if_due(self, history: pd.DataFrame, latest: int | None) -> bool:
        """Fold `history` into `last_history` only if the render throttle allows it now."""
        is_due = latest is not None and should_render_checkpoint(
            latest, self._last_rendered_evaluation, self._update_every_evaluations
        )
        if not is_due:
            return False
        self._last_rendered_evaluation = latest
        self._last_history = deduplicate_consecutive_checkpoints(history)
        return True

    def _file_grew(self, indicators_csv: Path) -> bool:
        """Check INDICATORS.csv's size against the last poll, to skip a no-op reparse."""
        if not indicators_csv.exists():
            return False
        size = indicators_csv.stat().st_size
        grew = size != self._last_file_size
        self._last_file_size = size
        return grew
