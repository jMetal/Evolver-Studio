"""Subprocess integration with Evolver's cli.runner (build + launch + poll)."""

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

from evolver_studio.result import Err, Ok

CANCEL_GRACE_PERIOD_SECONDS = 2.0

TRAINING_RUNNER_MAIN_CLASS = "org.uma.evolver.cli.training.TrainingRunnerMain"
DESCRIBE_MAIN_CLASS = "org.uma.evolver.cli.training.DescribeMain"
# Evolver release this app is built against (the `v2.1` tag of jMetal/Evolver).
EVOLVER_VERSION = "2.1"
JAR_RELATIVE_PATH = Path(f"target/Evolver-{EVOLVER_VERSION}-jar-with-dependencies.jar")


class RunState(Enum):
    """Possible values of status.yaml's `status` field."""

    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


@dataclass(slots=True, frozen=True)
class RunStatus:
    """Parsed contents of status.yaml.

    Attributes:
        state: Current run state.
        evaluations_done: Meta-level evaluations completed so far.
        max_evaluations: Meta-level evaluation budget.
        updated_at: Timestamp of the last status write, as written by Evolver.
        error_message: Failure reason, only set when state is FAILED.
    """

    state: RunState
    evaluations_done: int
    max_evaluations: int
    updated_at: str
    error_message: str | None = None


def jar_path(evolver_home: Path) -> Path:
    """Return the expected path of Evolver's fat jar.

    Args:
        evolver_home: Path to the Evolver checkout.

    Returns:
        The absolute path where `mvn package` writes the fat jar.
    """
    return evolver_home / JAR_RELATIVE_PATH


def build_jar(evolver_home: Path) -> Ok[None] | Err:
    """Build Evolver's fat jar with Maven, skipping tests.

    Args:
        evolver_home: Path to the Evolver checkout.

    Returns:
        Ok(None) on success, Err(message) with captured Maven output on failure.
    """
    pom = evolver_home / "pom.xml"
    result = subprocess.run(
        ["mvn", "-q", "-f", str(pom), "package", "-DskipTests"],
        capture_output=True,
        text=True,
    )
    return Err(result.stdout + result.stderr) if result.returncode != 0 else Ok(None)


def start_training(
    evolver_home: Path, jar: Path, request_yaml: Path, status_yaml: Path
) -> subprocess.Popen:
    """Launch TrainingRunnerMain as a background subprocess.

    Args:
        evolver_home: Working directory the JVM resolves relative paths against.
        jar: Path to Evolver's fat jar.
        request_yaml: Path to the training request YAML.
        status_yaml: Path where the run's status is written.

    Returns:
        The launched subprocess, not yet awaited.
    """
    return subprocess.Popen(
        ["java", "-cp", str(jar), TRAINING_RUNNER_MAIN_CLASS, str(request_yaml), str(status_yaml)],
        cwd=evolver_home,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def write_pid_file(pid_file: Path, pid: int) -> None:
    """Persist a launched subprocess's PID so it can be found after this process exits.

    Args:
        pid_file: Path to write the PID to.
        pid: The subprocess's process ID.
    """
    pid_file.write_text(str(pid))


def read_pid(pid_file: Path) -> int | None:
    """Read a previously written PID file.

    Args:
        pid_file: Path written by `write_pid_file`.

    Returns:
        The PID, or None if the file is missing or not a valid integer.
    """
    try:
        return int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def is_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still running.

    Args:
        pid: The process ID to check.

    Returns:
        True if a process with that PID currently exists.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def cancel_training(pid: int) -> Ok[None] | Err:
    """Stop a running TrainingRunnerMain subprocess by PID.

    Sends SIGTERM and, if the process hasn't exited after a short grace
    period, follows up with SIGKILL. Idempotent: a PID that's already gone
    counts as success.

    Args:
        pid: The subprocess's process ID, as written by `write_pid_file`.

    Returns:
        Ok(None) once the process is confirmed gone, Err(message) if it
        couldn't be signaled (e.g. permission denied).
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return Ok(None)
    except PermissionError as error:
        return Err(f"Could not signal process {pid}: {error}")
    time.sleep(CANCEL_GRACE_PERIOD_SECONDS)
    if is_alive(pid):
        os.kill(pid, signal.SIGKILL)
    return Ok(None)


def describe(evolver_home: Path, jar: Path) -> Ok[dict] | Err:
    """Run DescribeMain and parse its manifest of what cli.training can resolve today.

    Args:
        evolver_home: Working directory the JVM resolves relative paths against.
        jar: Path to Evolver's fat jar.

    Returns:
        Ok(manifest) with the parsed manifest (base/meta algorithms, problems,
        indicators, resource directories, request/baseLevel/metaSearch
        schemas), or Err(message) if the subprocess failed or its output
        wasn't valid YAML.
    """
    result = subprocess.run(
        ["java", "-cp", str(jar), DESCRIBE_MAIN_CLASS],
        cwd=evolver_home,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return Err(result.stdout + result.stderr)
    try:
        manifest = yaml.safe_load(result.stdout)
    except yaml.YAMLError as error:
        return Err(f"DescribeMain produced invalid YAML: {error}")
    if not isinstance(manifest, dict):
        return Err(f"DescribeMain produced unexpected output: {result.stdout!r}")
    return Ok(manifest)


def read_status(status_yaml: Path) -> RunStatus | None:
    """Read and parse status.yaml, tolerating a missing or half-written file.

    Args:
        status_yaml: Path to the status file written by RunStatusWriter.

    Returns:
        The parsed status, or None if the file is absent or not yet valid YAML.
    """
    try:
        raw = yaml.safe_load(status_yaml.read_text())
    except (FileNotFoundError, yaml.YAMLError):
        return None
    required_keys = {"status", "evaluationsDone", "maxEvaluations", "updatedAt"}
    if not isinstance(raw, dict) or not required_keys.issubset(raw):
        return None
    try:
        state = RunState(raw["status"])
    except ValueError:
        return None
    return RunStatus(
        state=state,
        evaluations_done=raw["evaluationsDone"],
        max_evaluations=raw["maxEvaluations"],
        updated_at=raw["updatedAt"],
        error_message=raw.get("errorMessage"),
    )
