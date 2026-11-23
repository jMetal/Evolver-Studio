"""What the home page shows besides the links: the run in progress, the latest result, the counts.

Streamlit-agnostic, so it can be unit-tested without a UI. Both uses of the app leave runs under
the working directory: Solve's in `solve-runs/` and the training runs' in `cli-runner-runs/`.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from evolver_studio.evolver_client import RunState
from evolver_studio.problem_catalogue import format_problem_spec
from evolver_studio.runs import (
    RUNS_DIRECTORY_NAME,
    SOLVE_RUNS_DIRECTORY_NAME,
    RunPhase,
    find_run_in_progress,
    run_phase,
)
from evolver_studio.solve_results import (
    list_solve_runs,
    read_request,
    read_run_fronts,
    read_run_violations,
)
from evolver_studio.training_runs import FinishedTraining, read_training

SOLVE = "Solve"
TRAINING = "Training"
MARKER_COLOR = "#14A3B8"
MAX_OBJECTIVES = 3
RECENT_RUNS = 3
MAX_PROBLEMS_SHOWN = 2
RUN_ID_FORMAT = "%Y%m%d-%H%M%S"


@dataclass(slots=True, frozen=True)
class RunningNow:
    """A run that is starting or running.

    Attributes:
        kind: SOLVE or TRAINING.
        run_id: The run's identifier.
        label: What it is, e.g. "NSGA-II on ZDT1".
    """

    kind: str
    run_id: str
    label: str


@dataclass(slots=True, frozen=True)
class LatestResult:
    """The newest finished run that has a front to show.

    Attributes:
        kind: SOLVE or TRAINING.
        run_id: The run's identifier.
        label: What it is, e.g. "NSGA-II on ZDT1".
        front: Its objective values, one column per objective and one row per solution (for a
            training, the meta-objectives of the configurations of its final front).
    """

    kind: str
    run_id: str
    label: str
    front: pd.DataFrame


@dataclass(slots=True, frozen=True)
class RecentRun:
    """A run among the newest, of either use.

    Attributes:
        kind: SOLVE or TRAINING.
        run_id: The run's identifier.
        label: What it is, e.g. "NSGA-II on ZDT1".
        phase: Where it is: running, finished, failed...
        launched: When it was launched, or None if its identifier is not a timestamp.
    """

    kind: str
    run_id: str
    label: str
    phase: RunPhase
    launched: datetime | None


@dataclass(slots=True, frozen=True)
class Activity:
    """What the app has been used for.

    Attributes:
        running: The run in progress (the most recent one, of either use), or None.
        latest: The newest finished run with a front, of either use, or None.
        solve_runs: How many solve runs are stored.
        trainings: How many training runs are stored.
        recent: The newest runs of either use, the newest first.
    """

    running: RunningNow | None
    latest: LatestResult | None
    solve_runs: int
    trainings: int
    recent: tuple[RecentRun, ...] = ()


def activity(working_directory: Path) -> Activity:
    """Summarize the runs stored under a working directory.

    Args:
        working_directory: The JVM's working directory, which holds both runs directories.

    Returns:
        The summary; empty when there are no runs.
    """
    solve_directory = working_directory / SOLVE_RUNS_DIRECTORY_NAME
    training_directory = working_directory / RUNS_DIRECTORY_NAME
    return Activity(
        running=_running_now(solve_directory, training_directory),
        latest=_latest_result(solve_directory, training_directory, working_directory),
        solve_runs=_count_runs(solve_directory),
        trainings=_count_runs(training_directory),
        recent=_recent_runs(solve_directory, training_directory, working_directory),
    )


def _newest(runs_directory: Path) -> list[Path]:
    if not runs_directory.is_dir():
        return []
    directories = [path for path in runs_directory.iterdir() if path.is_dir()]
    return sorted(directories, reverse=True)[:RECENT_RUNS]


def _launched(run_id: str) -> datetime | None:
    try:
        return datetime.strptime(run_id, RUN_ID_FORMAT)
    except ValueError:
        return None


def _recent_runs(
    solve_directory: Path, training_directory: Path, working_directory: Path
) -> tuple[RecentRun, ...]:
    recent = []
    for run_dir in _newest(solve_directory):
        request = read_request(run_dir) or {}
        problem = format_problem_spec(request.get("problem", "a problem"))
        label = f"{request.get('algorithmName', 'An algorithm')} on {problem}"
        recent.append(_recent(SOLVE, run_dir, label))
    for run_dir in _newest(training_directory):
        training = read_training(run_dir, working_directory)
        label = _training_label(training) if training is not None else "A training"
        recent.append(_recent(TRAINING, run_dir, label))
    recent.sort(key=lambda run: run.run_id, reverse=True)
    return tuple(recent[:RECENT_RUNS])


def _training_label(training: FinishedTraining) -> str:
    """The algorithm and its first problems, e.g. "PAES on RE31, RE32 and 5 more"."""
    shown = training.problems[:MAX_PROBLEMS_SHOWN]
    more = len(training.problems) - len(shown)
    problems = ", ".join(shown) + (f" and {more} more" if more else "")
    return f"{training.algorithm} on {problems}"


def _recent(kind: str, run_dir: Path, label: str) -> RecentRun:
    return RecentRun(kind, run_dir.name, label, run_phase(run_dir), _launched(run_dir.name))


def _count_runs(runs_directory: Path) -> int:
    if not runs_directory.is_dir():
        return 0
    return sum(1 for path in runs_directory.iterdir() if path.is_dir())


def _running_now(solve_directory: Path, training_directory: Path) -> RunningNow | None:
    candidates = []
    solve = find_run_in_progress(solve_directory)
    if solve is not None:
        request = read_request(solve) or {}
        problem = format_problem_spec(request.get("problem", "a problem"))
        label = f"{request.get('algorithmName', 'An algorithm')} on {problem}"
        candidates.append(RunningNow(SOLVE, solve.name, label))
    training = find_run_in_progress(training_directory)
    if training is not None:
        candidates.append(RunningNow(TRAINING, training.name, f"Training {training.name}"))
    return max(candidates, key=lambda running: running.run_id, default=None)


def _latest_result(
    solve_directory: Path, training_directory: Path, working_directory: Path
) -> LatestResult | None:
    candidates = [
        result
        for result in (
            _latest_solve(solve_directory, working_directory),
            _latest_training(training_directory, working_directory),
        )
        if result is not None
    ]
    return max(candidates, key=lambda result: result.run_id, default=None)


def _latest_solve(solve_directory: Path, working_directory: Path) -> LatestResult | None:
    for run in list_solve_runs(solve_directory, working_directory):
        if run.state != RunState.FINISHED or not run.output_directory.is_dir():
            continue
        fronts = read_run_fronts(run.output_directory)
        violations = read_run_violations(run.output_directory, fronts)
        # Only the feasible solutions are solutions of a problem with constraints.
        feasible = [
            front[violations[number] <= 0] if number in violations else front
            for number, front in fronts.items()
        ]
        front = pd.concat(feasible, ignore_index=True) if feasible else pd.DataFrame()
        if not front.empty:
            return LatestResult(SOLVE, run.run_id, f"{run.algorithm} on {run.problem}", front)
    return None


def _latest_training(training_directory: Path, working_directory: Path) -> LatestResult | None:
    if not training_directory.is_dir():
        return None
    for run_dir in sorted(training_directory.iterdir(), reverse=True):
        training = read_training(run_dir, working_directory)
        if training is None or not training.configurations:
            continue
        front = pd.DataFrame([config.objectives for config in training.configurations])
        label = f"Training of {training.algorithm} on {', '.join(training.problems)}"
        return LatestResult(TRAINING, training.run_id, label, front)
    return None


def front_figure(front: pd.DataFrame, height: int) -> go.Figure | None:
    """Plot a front, small, in the logo's teal.

    Args:
        front: The objective values, one column per objective.
        height: The figure's height in pixels.

    Returns:
        A scatter with two objectives, a 3D scatter with three, or None with any other number
        (a plot of a front of many objectives is not a picture for the home page).
    """
    columns = list(front.columns)
    if len(columns) == 2:
        figure = px.scatter(front, x=columns[0], y=columns[1])
    elif len(columns) == MAX_OBJECTIVES:
        figure = px.scatter_3d(front, x=columns[0], y=columns[1], z=columns[2])
    else:
        return None
    figure.update_traces(marker={"color": MARKER_COLOR, "size": 4})
    figure.update_layout(height=height, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    return figure
