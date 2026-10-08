"""Monitoring a training run from the files Evolver appends to while it runs.

A training can last many hours, and Evolver only ever appends to its result files, so the monitor
follows them from where it stopped reading instead of reading them again at every poll:

- `CheckpointReader` follows INDICATORS.csv (the non-dominated front at each checkpoint) or
  POPULATION_INDICATORS.csv (the whole population), keeping a summary of every checkpoint and the
  rows of only the first and the latest ones, so that its memory does not grow with the run.
- `VarConfFollower` follows VAR_CONF.txt: the computing time of each checkpoint and the
  configurations of the latest front.
- `summarize_run` turns the status file and the polls so far into what to tell the user: the time
  spent, the pace, the time left, and whether the run seems stuck.
- The figures show the evolution of the population and the convergence of the meta-objectives.

Nothing here depends on Streamlit.
"""

import io
import statistics
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evolver_studio.evolver_client import RunStatus
from evolver_studio.training_results import Checkpoint, parse_var_conf_checkpoints

EVALUATION_COLUMN = "Evaluation"
SOLUTION_COLUMN = "SolutionId"
KEPT_RECENT_CHECKPOINTS = 30
MAX_PLOTTED_CHECKPOINTS = 2000
# A run seems stuck when it has made no progress for this many typical gaps between its updates,
# and at least this many seconds.
STUCK_AFTER_GAPS = 10
STUCK_AFTER_MINIMUM_SECONDS = 180.0
POPULATION_COLOR = "#b0b7c3"
FRONT_COLOR = "#1f77b4"
FIRST_FRONT_COLOR = "#e8b04a"


@dataclass(slots=True, frozen=True)
class CheckpointSummary:
    """What one checkpoint of the front or the population looked like, by indicator.

    Attributes:
        evaluation: The meta-evaluations done when it was written.
        size: How many solutions it has.
        minimum: The lowest value of each indicator among them.
        median: The median of each indicator.
        maximum: The highest value of each indicator.
    """

    evaluation: int
    size: int
    minimum: dict[str, float]
    median: dict[str, float]
    maximum: dict[str, float]


class CheckpointReader:
    """Follows an append-only CSV of checkpoints (`Evaluation,SolutionId,<indicators>`).

    Evolver writes one block of rows per checkpoint, and a block starts where `SolutionId` goes
    back to 0. The file is read from where the last poll stopped; a line not yet ended by a
    newline, which Evolver may be in the middle of writing, waits for the next poll.
    """

    def __init__(self, path: Path, kept_recent: int = KEPT_RECENT_CHECKPOINTS) -> None:
        self._path = path
        self._kept_recent = kept_recent
        self._clear()

    @property
    def indicators(self) -> list[str]:
        """The indicators of the file, in column order; empty until its header is read."""
        return list(self._columns[2:]) if self._columns else []

    def poll(self) -> bool:
        """Read what was appended since the last poll.

        Returns:
            Whether there was anything new.
        """
        try:
            size = self._path.stat().st_size
        except OSError:
            return False
        if size < self._offset:
            self._clear()
        if size == self._offset:
            return False
        with self._path.open("rb") as file:
            file.seek(self._offset)
            chunk = file.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return False
        self._offset += end + 1
        self._ingest(chunk[: end + 1].decode())
        return True

    @property
    def summaries(self) -> list[CheckpointSummary]:
        """A summary of every checkpoint read so far, the one being written included."""
        summaries = list(self._summaries)
        if self._open_rows:
            summaries.append(self._summarize(self._frame(self._open_rows)))
        return summaries

    @property
    def first(self) -> pd.DataFrame | None:
        """The rows of the first checkpoint (the first to be read), or None before any."""
        if self._first is not None:
            return self._first
        return self._frame(self._open_rows) if self._open_rows else None

    @property
    def latest(self) -> pd.DataFrame | None:
        """The rows of the latest checkpoint, or None before any."""
        if self._open_rows:
            return self._frame(self._open_rows)
        return self._recent[-1] if self._recent else None

    def recent(self) -> list[pd.DataFrame]:
        """The rows of the latest checkpoints that are kept (the one being written included)."""
        blocks = list(self._recent)
        if self._open_rows:
            blocks.append(self._frame(self._open_rows))
        return blocks

    def checkpoint(self, evaluation: int) -> pd.DataFrame | None:
        """The rows of a checkpoint, if it is among the first and the latest kept.

        Args:
            evaluation: The checkpoint's meta-evaluations.

        Returns:
            Its rows, or None if it is not kept (or does not exist).
        """
        for block in (*reversed(self.recent()), self.first):
            if block is not None and int(block[EVALUATION_COLUMN].iloc[0]) == evaluation:
                return block
        return None

    def _clear(self) -> None:
        self._offset = 0
        self._columns: list[str] | None = None
        self._open_rows: list[list[float]] = []
        self._summaries: list[CheckpointSummary] = []
        self._first: pd.DataFrame | None = None
        self._recent: deque[pd.DataFrame] = deque(maxlen=self._kept_recent)

    def _ingest(self, text: str) -> None:
        lines = text.splitlines()
        if self._columns is None:
            self._columns = lines[0].strip().split(",")
            lines = lines[1:]
        rows = pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=self._columns)
        for row in rows.itertuples(index=False, name=None):
            if int(row[1]) == 0 and self._open_rows:
                self._close_open_block()
            self._open_rows.append(list(row))

    def _close_open_block(self) -> None:
        frame = self._frame(self._open_rows)
        self._summaries.append(self._summarize(frame))
        if self._first is None:
            self._first = frame
        self._recent.append(frame)
        self._open_rows = []

    def _frame(self, rows: list[list[float]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=self._columns)
        return frame.astype({EVALUATION_COLUMN: int, SOLUTION_COLUMN: int})

    def _summarize(self, block: pd.DataFrame) -> CheckpointSummary:
        values = block[self.indicators]
        return CheckpointSummary(
            evaluation=int(block[EVALUATION_COLUMN].iloc[0]),
            size=len(block),
            minimum=values.min().to_dict(),
            median=values.median().to_dict(),
            maximum=values.max().to_dict(),
        )


class VarConfFollower:
    """Follows VAR_CONF.txt: the computing time of each checkpoint and the latest front.

    A checkpoint ends with a blank line, so only complete ones are read, from where the last poll
    stopped.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = 0
        self._minutes: dict[int, float] = {}
        self._latest: Checkpoint | None = None

    @property
    def minutes(self) -> dict[int, float]:
        """The computing time, in minutes, of every checkpoint read, by its evaluations."""
        return dict(self._minutes)

    @property
    def latest(self) -> Checkpoint | None:
        """The latest complete checkpoint: the front of configurations found so far."""
        return self._latest

    def poll(self) -> bool:
        """Read the checkpoints appended since the last poll.

        Returns:
            Whether there was a new one.
        """
        try:
            size = self._path.stat().st_size
        except OSError:
            return False
        if size < self._offset:
            self._offset, self._minutes, self._latest = 0, {}, None
        if size == self._offset:
            return False
        with self._path.open("rb") as file:
            file.seek(self._offset)
            chunk = file.read().decode()
        end = chunk.rfind("\n\n")
        if end < 0:
            return False
        self._offset += len(chunk[: end + 2].encode())
        checkpoints = parse_var_conf_checkpoints(chunk[: end + 2])
        for checkpoint in checkpoints:
            if checkpoint.evaluation is not None and checkpoint.minutes is not None:
                self._minutes[checkpoint.evaluation] = checkpoint.minutes
        if checkpoints:
            self._latest = checkpoints[-1]
        return bool(checkpoints)


@dataclass(slots=True, frozen=True)
class RunSummary:
    """What to tell the user about a run in progress.

    Attributes:
        elapsed_seconds: The time since the run started, or None if it is not known.
        evaluations_done: The meta-evaluations done so far.
        evaluations_total: The meta-evaluations to do, or None when the run is limited by time.
        evaluations_per_minute: The pace, or None while there is nothing to measure it with.
        remaining_seconds: The time left, estimated, or None when it cannot be.
        stuck: Whether the run made no progress for much longer than it takes between two
            updates, with its process alive.
        seconds_without_progress: For how long its evaluations have not changed.
    """

    elapsed_seconds: float | None
    evaluations_done: int
    evaluations_total: int | None
    evaluations_per_minute: float | None
    remaining_seconds: float | None
    stuck: bool
    seconds_without_progress: float


def summarize_run(
    status: RunStatus,
    started_at: datetime | None,
    samples: Sequence[tuple[float, int]],
    now: datetime,
    now_seconds: float,
    process_alive: bool,
) -> RunSummary:
    """Summarize a run from its status and from what was seen at each poll.

    Args:
        status: The run's status.
        started_at: When the run started, or None if it is not known.
        samples: What each poll saw, oldest first: the monotonic time and the evaluations done.
        now: The time now, to measure the time since the start.
        now_seconds: The monotonic time now, comparable with the samples'.
        process_alive: Whether the run's process is alive.

    Returns:
        The summary.
    """
    elapsed = _elapsed_seconds(status, started_at, now)
    total = status.max_evaluations or None
    rate = _evaluations_per_minute(samples, status.evaluations_done, elapsed)
    without_progress = _seconds_without_progress(samples, now_seconds)
    return RunSummary(
        elapsed_seconds=elapsed,
        evaluations_done=status.evaluations_done,
        evaluations_total=total,
        evaluations_per_minute=rate,
        remaining_seconds=_remaining_seconds(status, elapsed, rate),
        stuck=process_alive and without_progress > _stuck_after(samples),
        seconds_without_progress=without_progress,
    )


def format_duration(seconds: float | None) -> str:
    """Write a duration as "2 h 5 min", "12 min 4 s" or "9 s"; "—" when it is not known."""
    if seconds is None:
        return "—"
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def tail_text(path: Path, lines: int = 40, max_bytes: int = 65536) -> str | None:
    """Read the last lines of a log, without reading the whole file.

    Args:
        path: The log.
        lines: How many lines to give.
        max_bytes: How far from the end to read at most.

    Returns:
        The last lines, or None if the file does not exist or cannot be read.
    """
    try:
        with path.open("rb") as file:
            file.seek(0, 2)
            size = file.tell()
            file.seek(max(size - max_bytes, 0))
            data = file.read()
    except OSError:
        return None
    text = data.decode(errors="replace")
    if size > max_bytes:
        text = text.split("\n", 1)[-1]  # the first line may have been cut in half
    return "\n".join(text.splitlines()[-lines:])


def _elapsed_seconds(status: RunStatus, started_at: datetime | None, now: datetime) -> float | None:
    if status.elapsed_minutes is not None:
        return status.elapsed_minutes * 60.0
    return max((now - started_at).total_seconds(), 0.0) if started_at is not None else None


def _evaluations_per_minute(
    samples: Sequence[tuple[float, int]], evaluations_done: int, elapsed: float | None
) -> float | None:
    if len(samples) >= 2 and samples[-1][0] > samples[0][0] and samples[-1][1] > samples[0][1]:
        (first_time, first), (last_time, last) = samples[0], samples[-1]
        return (last - first) / (last_time - first_time) * 60.0
    if elapsed and evaluations_done > 0:
        return evaluations_done / elapsed * 60.0
    return None


def _remaining_seconds(
    status: RunStatus, elapsed: float | None, rate: float | None
) -> float | None:
    if status.max_computing_time_minutes:
        if elapsed is None:
            return None
        return max(status.max_computing_time_minutes * 60.0 - elapsed, 0.0)
    if status.max_evaluations and rate:
        return max(status.max_evaluations - status.evaluations_done, 0) / rate * 60.0
    return None


def _changes(samples: Sequence[tuple[float, int]]) -> list[float]:
    """The monotonic times at which the evaluations changed."""
    return [
        time for (_, previous), (time, current) in zip(samples, samples[1:]) if current != previous
    ]


def _seconds_without_progress(samples: Sequence[tuple[float, int]], now_seconds: float) -> float:
    if not samples:
        return 0.0
    changes = _changes(samples)
    return now_seconds - (changes[-1] if changes else samples[0][0])


def _stuck_after(samples: Sequence[tuple[float, int]]) -> float:
    changes = _changes(samples)
    gaps = [later - earlier for earlier, later in zip(changes, changes[1:])]
    typical = statistics.median(gaps) if gaps else 0.0
    return max(STUCK_AFTER_GAPS * typical, STUCK_AFTER_MINIMUM_SECONDS)


def _thinned(summaries: list[CheckpointSummary]) -> list[CheckpointSummary]:
    if len(summaries) <= MAX_PLOTTED_CHECKPOINTS:
        return summaries
    step = -(-len(summaries) // MAX_PLOTTED_CHECKPOINTS)
    thinned = summaries[::step]
    return thinned if thinned[-1] is summaries[-1] else [*thinned, summaries[-1]]


def build_population_figure(
    population: pd.DataFrame | None,
    front: pd.DataFrame | None,
    first_front: pd.DataFrame | None,
    indicators: tuple[str, str],
    log_scale: bool = False,
) -> go.Figure:
    """Plot a checkpoint's population in the space of two meta-objectives.

    The population is drawn small and grey, its non-dominated front on top in color, and the front
    of the first checkpoint faded, to see how far the search has come.

    Args:
        population: The rows of the population at the checkpoint, or None when the run did not
            write it.
        front: The rows of the non-dominated front at the same checkpoint.
        first_front: The rows of the front at the first checkpoint, or None.
        indicators: The meta-objectives of the axes.
        log_scale: Whether to draw the axes in a logarithmic scale.

    Returns:
        The figure, titled with the checkpoint's evaluations.
    """
    x, y = indicators
    figure = go.Figure()
    layers = (
        (first_front, "First checkpoint", FIRST_FRONT_COLOR, 6, 0.35),
        (population, "Population", POPULATION_COLOR, 7, 0.9),
        (front, "Non-dominated front", FRONT_COLOR, 9, 1.0),
    )
    for rows, name, color, size, opacity in layers:
        if rows is None or rows.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=rows[x],
                y=rows[y],
                mode="markers",
                name=name,
                marker={"color": color, "size": size, "opacity": opacity},
            )
        )
    shown = population if population is not None and not population.empty else front
    evaluation = int(shown[EVALUATION_COLUMN].iloc[0]) if shown is not None and len(shown) else 0
    figure.update_layout(
        title=f"Population at evaluation {evaluation}",
        xaxis_title=x,
        yaxis_title=y,
        legend={"orientation": "h", "y": -0.2},
    )
    if log_scale:
        figure.update_xaxes(type="log")
        figure.update_yaxes(type="log")
    return figure


def build_convergence_figure(
    summaries: list[CheckpointSummary],
    minutes: dict[int, float] | None = None,
    log_scale: bool = False,
) -> go.Figure:
    """Plot how each meta-objective improves: the best, the median and the worst of the front.

    One panel per indicator. The best value is a solid line and the band reaches the worst one;
    the median is dashed. Lower is better.

    Args:
        summaries: The summaries of the front's checkpoints.
        minutes: The computing time of each checkpoint, by evaluations, to put time on the x axis
            instead of the evaluations; checkpoints with no known time are left out.
        log_scale: Whether to draw the values in a logarithmic scale.

    Returns:
        The figure.
    """
    shown = _thinned(summaries)
    if minutes:
        shown = [s for s in shown if s.evaluation in minutes]
    indicators = list(shown[0].minimum) if shown else []
    figure = make_subplots(rows=max(len(indicators), 1), cols=1, shared_xaxes=True)
    x_values = [minutes[s.evaluation] if minutes else s.evaluation for s in shown]
    for row, indicator in enumerate(indicators, start=1):
        lowest = [s.minimum[indicator] for s in shown]
        highest = [s.maximum[indicator] for s in shown]
        median = [s.median[indicator] for s in shown]
        legend = row == 1
        figure.add_trace(
            go.Scatter(x=x_values, y=highest, line={"width": 0}, showlegend=False), row=row, col=1
        )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=lowest,
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.15)",
                line={"color": FRONT_COLOR},
                name="Best (band up to the worst)",
                showlegend=legend,
            ),
            row=row,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=median,
                line={"color": FRONT_COLOR, "dash": "dash"},
                name="Median",
                showlegend=legend,
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text=indicator, row=row, col=1)
        if log_scale:
            figure.update_yaxes(type="log", row=row, col=1)
    figure.update_xaxes(
        title_text="Computing time (min)" if minutes else "Meta-evaluations",
        row=len(indicators),
        col=1,
    )
    figure.update_layout(height=260 * max(len(indicators), 1) + 60, legend={"orientation": "h"})
    return figure
