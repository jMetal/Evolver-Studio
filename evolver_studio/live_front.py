"""Throttled live preview of the meta-level indicator front during a run.

Deliberately Streamlit-agnostic: builds Plotly figures from INDICATORS.csv's
growing checkpoint history, so it can be unit-tested without a UI.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from evolver_studio.results import checkpoint_front, latest_checkpoint_evaluation, load_indicators


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


def build_front_figure(front: pd.DataFrame, evaluation: int) -> go.Figure:
    """Plot one checkpoint's non-dominated archive as an indicator scatter.

    Args:
        front: Rows of a single checkpoint (same Evaluation value).
        evaluation: That checkpoint's evaluation count, shown in the title.

    Returns:
        A Plotly scatter figure, axes taken from the front's own columns.
    """
    x_axis, y_axis = front.columns[2], front.columns[3]
    return px.scatter(
        front, x=x_axis, y=y_axis, title=f"Indicator front @ {evaluation} evaluations"
    )


@dataclass(slots=True, frozen=True)
class LiveFrontUpdate:
    """Result of one poll of INDICATORS.csv.

    Attributes:
        changed: Whether the file grew since the previous poll — independent
            of the render throttle, useful to drive an adaptive poll interval.
        figure: A new figure for the latest checkpoint, only if one was due.
        evaluation: That checkpoint's evaluation count, set whenever `figure`
            is — lets callers build a stable, unique widget key per redraw.
    """

    changed: bool
    figure: go.Figure | None
    evaluation: int | None = None


class LiveFrontRenderer:
    """Tracks throttling state across repeated polls of a growing INDICATORS.csv."""

    def __init__(self, update_every_evaluations: int) -> None:
        self._update_every_evaluations = update_every_evaluations
        self._last_rendered_evaluation: int | None = None
        self._last_file_size = -1
        self._last_figure: go.Figure | None = None

    @property
    def last_figure(self) -> go.Figure | None:
        """The most recently built figure, if any — for redrawing every tick.

        A fragment fully replaces its own contents each time it reruns, so a
        caller must redraw this every tick (not only when `poll` reports a new
        figure) to avoid the preview flickering away between throttled redraws.
        """
        return self._last_figure

    @property
    def last_rendered_evaluation(self) -> int | None:
        """The evaluation count of `last_figure`, or None if nothing rendered yet."""
        return self._last_rendered_evaluation

    def poll(self, indicators_csv: Path) -> LiveFrontUpdate:
        """Check INDICATORS.csv once for new data and, if due, a figure to show.

        Args:
            indicators_csv: Path to the run's (still-growing) INDICATORS.csv.

        Returns:
            Whether the file grew since the last poll, and a new figure only
            if a checkpoint is due to be (re)rendered.
        """
        if not self._file_grew(indicators_csv):
            return LiveFrontUpdate(changed=False, figure=None)
        history = load_indicators(indicators_csv)
        latest = latest_checkpoint_evaluation(history)
        figure = self._figure_if_due(history, latest)
        return LiveFrontUpdate(changed=True, figure=figure, evaluation=latest if figure else None)

    def _figure_if_due(self, history: pd.DataFrame, latest: int | None) -> go.Figure | None:
        """Build a figure for `latest` only if the render throttle allows it now."""
        is_due = latest is not None and should_render_checkpoint(
            latest, self._last_rendered_evaluation, self._update_every_evaluations
        )
        if not is_due:
            return None
        self._last_rendered_evaluation = latest
        figure = build_front_figure(checkpoint_front(history, latest), latest)
        self._last_figure = figure
        return figure

    def _file_grew(self, indicators_csv: Path) -> bool:
        """Check INDICATORS.csv's size against the last poll, to skip a no-op reparse."""
        if not indicators_csv.exists():
            return False
        size = indicators_csv.stat().st_size
        grew = size != self._last_file_size
        self._last_file_size = size
        return grew
