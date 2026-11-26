"""Waiting in the tests for a run (a training, a validation study, a solve run) to end.

The time a run takes depends on the computer: a runner of the continuous integration has fewer
cores than a laptop, and is shared. The limits are given for a laptop and multiplied by the
environment variable EVOLVER_STUDIO_TEST_TIME_FACTOR (1 by default; the CI sets it).

A run that does not end in time is stopped before the test fails: left running, it would slow down
the tests that come after it, and make them fail too.
"""

import os
import signal
import time
from pathlib import Path

import pytest

from evolver_studio.evolver_client import read_pid
from evolver_studio.runs import RunPhase, run_phase

TIME_FACTOR_VARIABLE = "EVOLVER_STUDIO_TEST_TIME_FACTOR"


def time_factor() -> float:
    """How much more time the runs get on this computer than on a laptop."""
    try:
        return max(float(os.environ.get(TIME_FACTOR_VARIABLE, "1")), 1.0)
    except ValueError:
        return 1.0


def wait_for_run(run_dir: Path, seconds: float, what: str = "the run", poll: float = 0.3) -> None:
    """Wait until a run is no longer starting or running; stop it and fail if it takes too long.

    Args:
        run_dir: The run's directory, with its status and PID files.
        seconds: How long it may take on a laptop.
        what: The run, to say in the failure (e.g. "the training").
        poll: How often to look, in seconds.
    """
    limit = seconds * time_factor()
    deadline = time.monotonic() + limit
    while run_phase(run_dir) in (RunPhase.STARTING, RunPhase.RUNNING):
        if time.monotonic() >= deadline:
            _stop(read_pid(run_dir / "pid.txt"))
            pytest.fail(f"{what} did not finish in {limit:.0f} s")
        time.sleep(poll)


def _stop(pid: int | None) -> None:
    """Stop a run's process, and its children when it leads a process group of its own."""
    if pid is None:
        return
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
