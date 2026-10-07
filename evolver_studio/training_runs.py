"""The training runs that have finished, with the configurations they found.

A training run is kept under cli-runner-runs/<id>/: its `request.yaml` (which says where the
results are written), its `base_level.yaml` (what was tuned) and its `status.yaml`.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from evolver_studio.evolver_client import RunState, read_status
from evolver_studio.runs import CANCELLED_MARKER_NAME, RUNS_DIRECTORY_NAME
from evolver_studio.training_results import TrainedConfiguration, read_var_conf


@dataclass(slots=True, frozen=True)
class FinishedTraining:
    """A finished training run.

    Attributes:
        run_id: The run's timestamp-based identifier.
        algorithm: The registry name of the base algorithm it tuned (e.g. "NSGA-II").
        encoding: The encoding it tuned it for.
        problems: The training problems' names.
        configurations: The configurations of its final front.
    """

    run_id: str
    algorithm: str
    encoding: str
    problems: tuple[str, ...]
    configurations: tuple[TrainedConfiguration, ...]

    @property
    def label(self) -> str:
        """The run in one line, e.g. "20261007-120000 · NSGA-II (Double) · ZDT4 · 14 configs"."""
        problems = ", ".join(self.problems)
        count = len(self.configurations)
        return f"{self.run_id} · {self.algorithm} ({self.encoding}) · {problems} · {count} configs"


def list_finished_trainings(working_directory: Path) -> list[FinishedTraining]:
    """List the finished training runs that found configurations, most recent first.

    Args:
        working_directory: The JVM's working directory, whose cli-runner-runs/ holds the runs.

    Returns:
        The runs; a cancelled or failed run, one still going, and one without a readable
        VAR_CONF.txt or base-level file are left out.
    """
    runs_directory = working_directory / RUNS_DIRECTORY_NAME
    if not runs_directory.is_dir():
        return []
    trainings = []
    for run_dir in sorted(runs_directory.iterdir(), reverse=True):
        training = _read_training(run_dir, working_directory)
        if training is not None:
            trainings.append(training)
    return trainings


def _read_training(run_dir: Path, working_directory: Path) -> FinishedTraining | None:
    if (run_dir / CANCELLED_MARKER_NAME).exists():
        return None
    status = read_status(run_dir / "status.yaml")
    if status is None or status.state != RunState.FINISHED:
        return None
    try:
        request = yaml.safe_load((run_dir / "request.yaml").read_text())
        base_level = yaml.safe_load((run_dir / "base_level.yaml").read_text())
        output_directory = working_directory / request["outputDirectory"]
        configurations = read_var_conf(output_directory / "VAR_CONF.txt")
        algorithm, encoding = base_level["algorithmName"], base_level.get("encoding", "Double")
        problems = tuple(_problem_name(problem) for problem in base_level["trainingProblemNames"])
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return None
    if not configurations:
        return None
    return FinishedTraining(run_dir.name, algorithm, encoding, problems, tuple(configurations))


def _problem_name(problem: object) -> str:
    if isinstance(problem, dict):
        return str(problem.get("class", problem))
    return str(problem)
