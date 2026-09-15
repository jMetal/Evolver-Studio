"""Subprocess integration with Evolver's cli.runner (build + launch + poll)."""

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

from evolver_studio.result import Err, Ok

TRAINING_RUNNER_MAIN_CLASS = "org.uma.evolver.cli.runner.TrainingRunnerMain"
JAR_RELATIVE_PATH = Path("target/Evolver-2.1-SNAPSHOT-jar-with-dependencies.jar")


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
