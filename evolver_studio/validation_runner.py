"""Launching and cancelling a validation study from the app."""

import os
import signal
import subprocess
import sys
from pathlib import Path

from evolver_studio.evolver_client import WORKING_DIRECTORY, write_pid_file

VALIDATION_RUNS_DIRECTORY_NAME = "validation-runs"
WORKER_LOG_NAME = "worker.log"


def start_study(study_directory: Path, jar: Path, processes: int) -> subprocess.Popen:
    """Run a written study in a detached process, in a session of its own.

    The session lets `cancel_study` stop the worker and the JVMs it started at once.

    Args:
        study_directory: The study's directory, with its manifest and the request of each job.
        jar: Evolver's jar.
        processes: How many jobs run at the same time.

    Returns:
        The worker's process; its PID is also written to the study's `pid.txt`.
    """
    with (study_directory / WORKER_LOG_NAME).open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "evolver_studio.validation_worker",
                str(study_directory),
                str(jar),
                str(processes),
            ],
            cwd=WORKING_DIRECTORY,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write_pid_file(study_directory / "pid.txt", process.pid)
    return process


def cancel_study(pid: int) -> None:
    """Stop a study: its worker and every JVM it started.

    Args:
        pid: The worker's PID, which is also the id of its process group.
    """
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return
