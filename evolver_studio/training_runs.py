"""The training runs that have finished, with the configurations they found.

A training run is kept under cli-runner-runs/<id>/: its `request.yaml` (which says where the
results are written), its `base_level.yaml` (what was tuned), its `meta_search.yaml` (how the
meta-optimizer searched) and its `status.yaml`; the results are in the output directory, with a
METADATA.txt that closes with how long the run took.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from evolver_studio.evolver_client import RunState, read_status
from evolver_studio.runs import CANCELLED_MARKER_NAME, RUNS_DIRECTORY_NAME
from evolver_studio.training_results import TrainedConfiguration, read_var_conf

POPULATION_FILE_NAME = "POPULATION_INDICATORS.csv"
_WALL_CLOCK = re.compile(r"^Wall-clock time: (\d+)h (\d+)m (\d+)s", re.MULTILINE)
_EVALUATIONS = re.compile(r"^Meta-evaluations performed: (\d+)", re.MULTILINE)


@dataclass(slots=True, frozen=True)
class TrainingOutcome:
    """What a training run took, as its METADATA.txt records it.

    Attributes:
        wall_clock_seconds: How long it took, to the second, or None if the file does not say.
        meta_evaluations: The meta-evaluations it performed, or None.
    """

    wall_clock_seconds: int | None
    meta_evaluations: int | None


@dataclass(slots=True, frozen=True)
class FinishedTraining:
    """A finished training run.

    Attributes:
        run_id: The run's timestamp-based identifier.
        algorithm: The registry name of the base algorithm it tuned (e.g. "NSGA-II").
        encoding: The encoding it tuned it for.
        problems: The training problems' names.
        configurations: The configurations of its final front.
        run_dir: The run's directory, or None if not known.
        output_directory: Where its results are, or None.
        population_size: The population size of the base algorithm, or None.
        problem_specs: The training problems as the request spells them (a name, or a
            `{class, args}` map).
        reference_fronts: The reference front file of each problem.
        evaluations: The evaluation budget of the base algorithm on each problem.
        indicators: The meta-objectives.
        extra_config: The base algorithm's extra settings (e.g. its weight vector directory).
        parameter_space_file: The parameter space the run searched, as the run kept it.
        meta_algorithm: The meta-optimizer, e.g. "NSGA-II".
        meta_encoding: Its encoding, "flat" or "tree".
        meta_population_size: Its population size, or None if left to Evolver (or it has none).
        meta_limit: What stopped it, e.g. "2000 evaluations" or "10 min".
        outcome: How long it took and how much it did, or None without METADATA.txt.
        has_population: Whether the run wrote the meta-optimizer's population.
    """

    run_id: str
    algorithm: str
    encoding: str
    problems: tuple[str, ...]
    configurations: tuple[TrainedConfiguration, ...]
    run_dir: Path | None = None
    output_directory: Path | None = None
    population_size: int | None = None
    problem_specs: tuple[object, ...] = ()
    reference_fronts: tuple[str, ...] = ()
    evaluations: tuple[int, ...] = ()
    indicators: tuple[str, ...] = ()
    extra_config: dict | None = None
    parameter_space_file: Path | None = None
    meta_algorithm: str | None = None
    meta_encoding: str | None = None
    meta_population_size: int | None = None
    meta_limit: str | None = None
    outcome: TrainingOutcome | None = None
    has_population: bool = False

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
        specs = tuple(base_level["trainingProblemNames"])
        problems = tuple(_problem_name(problem) for problem in specs)
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return None
    if not configurations:
        return None
    meta = _read_meta_search(run_dir)
    return FinishedTraining(
        run_id=run_dir.name,
        algorithm=algorithm,
        encoding=encoding,
        problems=problems,
        configurations=tuple(configurations),
        run_dir=run_dir,
        output_directory=output_directory,
        population_size=base_level.get("populationSize"),
        problem_specs=specs,
        reference_fronts=tuple(base_level.get("trainingReferenceFrontFileNames") or ()),
        evaluations=tuple(base_level.get("trainingEvaluations") or ()),
        indicators=tuple(base_level.get("indicatorNames") or ()),
        extra_config=base_level.get("extraConfig"),
        parameter_space_file=_path(base_level.get("yamlParameterSpaceFile")),
        meta_algorithm=meta.get("algorithm"),
        meta_encoding=meta.get("encoding"),
        meta_population_size=meta.get("metaPopulationSize"),
        meta_limit=_meta_limit(meta),
        outcome=read_outcome(output_directory / "METADATA.txt"),
        has_population=(output_directory / POPULATION_FILE_NAME).is_file(),
    )


def read_outcome(metadata_file: Path) -> TrainingOutcome | None:
    """Read how long a training took and how much it did from its METADATA.txt.

    Args:
        metadata_file: The run's METADATA.txt.

    Returns:
        The outcome, or None if the file cannot be read or says neither.
    """
    try:
        text = metadata_file.read_text()
    except OSError:
        return None
    clock, evaluations = _WALL_CLOCK.search(text), _EVALUATIONS.search(text)
    seconds = (
        int(clock.group(1)) * 3600 + int(clock.group(2)) * 60 + int(clock.group(3))
        if clock
        else None
    )
    done = int(evaluations.group(1)) if evaluations else None
    return TrainingOutcome(seconds, done) if seconds is not None or done is not None else None


def _read_meta_search(run_dir: Path) -> dict:
    try:
        meta = yaml.safe_load((run_dir / "meta_search.yaml").read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _meta_limit(meta: dict) -> str | None:
    if meta.get("metaMaxComputingTimeMinutes") is not None:
        return f"{meta['metaMaxComputingTimeMinutes']:g} min"
    if meta.get("metaMaxEvaluations") is not None:
        return f"{meta['metaMaxEvaluations']:,} evaluations"
    return None


def _path(value: object) -> Path | None:
    return Path(value) if isinstance(value, str) and value else None


def _problem_name(problem: object) -> str:
    if isinstance(problem, dict):
        return str(problem.get("class", problem))
    return str(problem)
