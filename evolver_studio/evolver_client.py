"""Subprocess integration with Evolver's cli.training (download + launch + poll)."""

import hashlib
import os
import signal
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

from evolver_studio.result import Err, Ok

CANCEL_GRACE_PERIOD_SECONDS = 2.0

TRAINING_RUNNER_MAIN_CLASS = "org.uma.evolver.cli.training.TrainingRunnerMain"
SOLVE_RUNNER_MAIN_CLASS = "org.uma.evolver.cli.solving.SolveRunnerMain"
DESCRIBE_MAIN_CLASS = "org.uma.evolver.cli.training.DescribeMain"

STUDIO_HOME = Path(__file__).resolve().parent.parent
# The JVM's working directory. Evolver resolves the relative paths of a training request
# (resources/referenceFronts/..., resources/weightVectors, the output directory) against it,
# so it must hold the resources/ directory copied from Evolver.
WORKING_DIRECTORY = STUDIO_HOME

# Evolver release this app is built against (the `v2.4` tag of jMetal/Evolver). Only stable
# releases: evolver_studio/catalogue.py mirrors this one.
EVOLVER_VERSION = "2.4"
JAR_FILE_NAME = f"Evolver-{EVOLVER_VERSION}-jar-with-dependencies.jar"
JAR_DIRECTORY = STUDIO_HOME / "lib"
MAVEN_CENTRAL_JAR_URL = (
    f"https://repo1.maven.org/maven2/org/uma/jmetal/Evolver/{EVOLVER_VERSION}/{JAR_FILE_NAME}"
)
# Points at another Evolver jar (e.g. one built from a local checkout) instead of the release's.
JAR_OVERRIDE_VARIABLE = "EVOLVER_JAR"
DOWNLOAD_CHUNK_BYTES = 1 << 20


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
        max_evaluations: Meta-level evaluation budget; 0 when the run is limited by computing
            time instead.
        updated_at: Timestamp of the last status write, as written by Evolver.
        error_message: Failure reason, only set when state is FAILED.
        max_computing_time_minutes: The computing time limit, when the run has one instead of a
            limit on the evaluations.
        elapsed_minutes: The computing time spent so far, for a run with a time limit.
    """

    state: RunState
    evaluations_done: int
    max_evaluations: int
    updated_at: str
    error_message: str | None = None
    max_computing_time_minutes: float | None = None
    elapsed_minutes: float | None = None


def is_jar_overridden() -> bool:
    """Tell whether the EVOLVER_JAR environment variable selects the jar to use."""
    return bool(os.environ.get(JAR_OVERRIDE_VARIABLE))


def jar_path() -> Path:
    """Return the path of the Evolver fat jar to run.

    Returns:
        The jar named by the EVOLVER_JAR environment variable if set, else where
        `download_jar` stores the release's jar (which may not exist yet).
    """
    if is_jar_overridden():
        return Path(os.environ[JAR_OVERRIDE_VARIABLE])
    return JAR_DIRECTORY / JAR_FILE_NAME


def download_jar(
    destination: Path,
    on_progress: Callable[[float], None] | None = None,
    url: str = MAVEN_CENTRAL_JAR_URL,
) -> Ok[None] | Err:
    """Download Evolver's fat jar from Maven Central, checking its SHA-1 checksum.

    The jar is written to a temporary file next to `destination` and renamed
    only once complete and verified, so an interrupted download never leaves
    a truncated jar behind.

    Args:
        destination: Where to store the jar.
        on_progress: Called with the fraction downloaded so far, if given.
        url: The jar's URL; Maven Central publishes its checksum at `url + ".sha1"`.

    Returns:
        Ok(None) once the jar is in place, Err(message) if the download failed
        or the checksum does not match.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        with urllib.request.urlopen(url + ".sha1") as response:
            expected_sha1 = response.read().decode().split()[0]
        actual_sha1 = _download_to(url, partial, on_progress)
    except OSError as error:
        partial.unlink(missing_ok=True)
        return Err(f"Could not download {url}: {error}")
    if actual_sha1 != expected_sha1:
        partial.unlink()
        return Err(f"Checksum mismatch for {url}: expected {expected_sha1}, got {actual_sha1}")
    partial.replace(destination)
    return Ok(None)


def _download_to(url: str, target: Path, on_progress: Callable[[float], None] | None) -> str:
    """Stream `url` into `target`, returning the SHA-1 hex digest of what was written."""
    sha1 = hashlib.sha1()
    with urllib.request.urlopen(url) as response, target.open("wb") as file:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
            file.write(chunk)
            sha1.update(chunk)
            done += len(chunk)
            if on_progress is not None and total:
                on_progress(min(done / total, 1.0))
    return sha1.hexdigest()


def start_training(
    working_directory: Path, jar: Path, request_yaml: Path, status_yaml: Path, log_file: Path
) -> subprocess.Popen:
    """Launch TrainingRunnerMain as a background subprocess.

    The runner's output goes to a log file. A pipe that nobody reads fills up (Evolver logs a line
    every `statusFrequency` evaluations) and, in a training of hours, would block the JVM; a file
    also keeps the log for the page to show, and tells why a run that never wrote its status
    (Java missing, a bad jar) did not start. The process is reaped when it ends, so that a
    finished run is not mistaken for a live one.

    Args:
        working_directory: Working directory the JVM resolves relative paths against.
        jar: Path to Evolver's fat jar.
        request_yaml: Path to the training request YAML.
        status_yaml: Path where the run's status is written.
        log_file: Path where the runner's standard output and error are written.

    Returns:
        The launched subprocess, not yet awaited.
    """
    with log_file.open("w") as log:
        process = subprocess.Popen(
            [
                "java",
                "-cp",
                str(jar),
                TRAINING_RUNNER_MAIN_CLASS,
                str(request_yaml),
                str(status_yaml),
            ],
            cwd=working_directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    threading.Thread(target=process.wait, daemon=True).start()
    return process


def start_solve(
    working_directory: Path, jar: Path, request_yaml: Path, status_yaml: Path, log_file: Path
) -> subprocess.Popen:
    """Launch SolveRunnerMain as a background subprocess.

    The runner's output goes to a log file, which is what tells the user why a run that never
    wrote its status (Java missing, a bad jar) did not start, and the process is reaped when it
    ends, so that a finished run is not mistaken for a live one.

    Args:
        working_directory: Working directory the JVM resolves relative paths against.
        jar: Path to Evolver's fat jar.
        request_yaml: Path to the solve request YAML.
        status_yaml: Path where the run's status is written.
        log_file: Path where the runner's standard output and error are written.

    Returns:
        The launched subprocess.
    """
    with log_file.open("w") as log:
        process = subprocess.Popen(
            ["java", "-cp", str(jar), SOLVE_RUNNER_MAIN_CLASS, str(request_yaml), str(status_yaml)],
            cwd=working_directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    threading.Thread(target=process.wait, daemon=True).start()
    return process


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


def describe(working_directory: Path, jar: Path) -> Ok[dict] | Err:
    """Run DescribeMain and parse its manifest of what cli.training can resolve today.

    Args:
        working_directory: Working directory the JVM resolves relative paths against.
        jar: Path to Evolver's fat jar.

    Returns:
        Ok(manifest) with the parsed manifest (base/meta algorithms, problems,
        indicators, resource directories, request/baseLevel/metaSearch
        schemas), or Err(message) if the subprocess failed or its output
        wasn't valid YAML.
    """
    try:
        result = subprocess.run(
            ["java", "-cp", str(jar), DESCRIBE_MAIN_CLASS],
            cwd=working_directory,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return Err("Java was not found on the PATH.")
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
        max_computing_time_minutes=raw.get("maxComputingTimeMinutes"),
        elapsed_minutes=raw.get("elapsedMinutes"),
    )
