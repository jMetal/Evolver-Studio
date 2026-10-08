"""Detecting an in-progress training run across app restarts/reconnects.

Evolver's own status.yaml contract has no CANCELLED state, and overwriting
it from Python would race with Evolver's own writes — so a cancelled run is
tracked with a separate marker file in its run directory instead.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

from evolver_studio.evolver_client import RunState, is_alive, read_pid, read_status

CANCELLED_MARKER_NAME = "CANCELLED"
RUNS_DIRECTORY_NAME = "cli-runner-runs"
# Solve runs are kept apart: find_active_run would take one for a training run.
SOLVE_RUNS_DIRECTORY_NAME = "solve-runs"


@dataclass(slots=True, frozen=True)
class ActiveRun:
    """Paths needed to reconnect to and control an in-progress run.

    Attributes:
        run_id: The run's timestamp-based identifier.
        run_dir: Directory holding this run's request/status/pid files.
        status_yaml: Path to the run's status file.
        pid_file: Path to the launched subprocess's PID file.
        indicators_csv: Path to the run's (still-growing) INDICATORS.csv.
        log_file: Path to the runner's log, `runner.log` in the run's directory.
    """

    run_id: str
    run_dir: Path
    status_yaml: Path
    pid_file: Path
    indicators_csv: Path
    log_file: Path


def mark_cancelled(run_dir: Path) -> None:
    """Record that a run was cancelled, so it's no longer reported as active.

    Args:
        run_dir: The cancelled run's directory.
    """
    (run_dir / CANCELLED_MARKER_NAME).write_text("")


def find_active_run(working_directory: Path) -> ActiveRun | None:
    """Find the most recent still-running, non-cancelled training run, if any.

    Args:
        working_directory: The JVM's working directory, whose cli-runner-runs
            subdirectory holds each run's files.

    Returns:
        The most recent matching run, or None if none is currently running.
    """
    runs_dir = working_directory / RUNS_DIRECTORY_NAME
    if not runs_dir.is_dir():
        return None
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        active_run = _active_run_in(run_dir)
        if active_run is not None:
            return active_run
    return None


def _active_run_in(run_dir: Path) -> ActiveRun | None:
    """Build an ActiveRun for `run_dir` if it holds a genuinely running job.

    status.yaml can lag behind reality: a process killed abruptly (crash,
    out-of-memory, `kill -9`) never gets to write a terminal state, leaving
    a stale "RUNNING" forever. Where a PID was recorded, its liveness is
    checked directly rather than trusting status.yaml alone.
    """
    if (run_dir / CANCELLED_MARKER_NAME).exists():
        return None
    status_yaml = run_dir / "status.yaml"
    status = read_status(status_yaml)
    if status is None or status.state != RunState.RUNNING:
        return None
    pid_file = run_dir / "pid.txt"
    pid = read_pid(pid_file)
    if pid is not None and not is_alive(pid):
        return None
    request_yaml = run_dir / "request.yaml"
    output_directory = yaml.safe_load(request_yaml.read_text())["outputDirectory"]
    return ActiveRun(
        run_id=run_dir.name,
        run_dir=run_dir,
        status_yaml=status_yaml,
        pid_file=pid_file,
        indicators_csv=run_dir.parent.parent / output_directory / "INDICATORS.csv",
        log_file=run_dir / "runner.log",
    )


class RunPhase(Enum):
    """Where a run is, as far as its files and its process say."""

    STARTING = "starting"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # The process is gone without the run having written a final status (it crashed, or never
    # started): its log, if it has one, says why.
    LOST = "lost"


def run_phase(run_dir: Path) -> RunPhase:
    """Tell where a run is.

    A final status (FINISHED, FAILED) is trusted. Otherwise the run is in progress only if its
    process is alive: no status yet means it is starting, and a dead process without a final
    status means it was lost, rather than leaving a stale RUNNING forever.

    Args:
        run_dir: The run's directory, with its status file, PID file and cancellation marker.

    Returns:
        The run's phase.
    """
    if (run_dir / CANCELLED_MARKER_NAME).exists():
        return RunPhase.CANCELLED
    status = read_status(run_dir / "status.yaml")
    if status is not None and status.state == RunState.FINISHED:
        return RunPhase.FINISHED
    if status is not None and status.state == RunState.FAILED:
        return RunPhase.FAILED
    pid = read_pid(run_dir / "pid.txt")
    if pid is None or not is_alive(pid):
        return RunPhase.LOST
    return RunPhase.STARTING if status is None else RunPhase.RUNNING


def find_run_in_progress(runs_directory: Path) -> Path | None:
    """Find the most recent run that is starting or running.

    Args:
        runs_directory: The directory whose subdirectories are runs.

    Returns:
        The run's directory, or None if no run is in progress.
    """
    if not runs_directory.is_dir():
        return None
    for run_dir in sorted(runs_directory.iterdir(), reverse=True):
        if run_phase(run_dir) in (RunPhase.STARTING, RunPhase.RUNNING):
            return run_dir
    return None
