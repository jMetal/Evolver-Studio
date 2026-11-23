"""A validation study: contenders run many times on the same problems, to be compared.

A study has a **pivot**, the tuned configuration, and other contenders: the default configurations
of Evolver's algorithms (those the jar ships one for) and any other configuration. Every contender
runs on every problem with the same population size and evaluation budget, as independent runs
with the same seeds, through `cli.solving`: one solve request per contender and problem, which is
a *job*. A study is kept under validation-runs/<id>/ as a manifest (`study.yaml`) and the files of
each job (`jobs/<n>/request.yaml`, its status and its `output/`).

The encoding is that of the problems, which must all share it: an algorithm can only solve problems
of an encoding it supports, so the contenders are the algorithms with a default configuration for
that encoding.

Most indicators are computed by Evolver as each job runs (`output/INDICATORS.csv`). Those Evolver
does not register (the hypervolume, maximized) are computed here from each run's front
(`output/run-<k>/FUN.csv`) when the runs are gathered, and kept next to Evolver's
(`output/STUDIO_INDICATORS.csv`), so that they are computed once.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from evolver_studio.catalogue import (
    BASE_ALGORITHMS,
    BaseAlgorithm,
    configuration_handles_constraints,
    quality_indicator,
)
from evolver_studio.hypervolume import hypervolume
from evolver_studio.problem_catalogue import ArgumentValue, format_problem_spec, problem_spec
from evolver_studio.problems import reference_front_dimension
from evolver_studio.solve_request import SolveRequest, solve_request_to_yaml
from evolver_studio.solve_results import read_violations
from evolver_studio.weight_vectors import available_population_sizes

MANIFEST_NAME = "study.yaml"
JOBS_DIRECTORY_NAME = "jobs"
OUTPUT_DIRECTORY_NAME = "output"
INDICATORS_FILE = "INDICATORS.csv"
STUDIO_INDICATORS_FILE = "STUDIO_INDICATORS.csv"
# The indicators Evolver-Studio computes from a run's front and its problem's reference front.
STUDIO_INDICATOR_FUNCTIONS = {"Hypervolume": hypervolume}
DEFAULT_RUNS = 30
DEFAULT_MAX_EVALUATIONS = 25000
DEFAULT_POPULATION_SIZE = 100
DEFAULT_WEIGHT_VECTORS_DIRECTORY = "resources/weightVectors"


@dataclass(slots=True, frozen=True)
class StudyProblem:
    """A problem of the study.

    Attributes:
        name: The problem's name.
        arguments: Its constructor's arguments, all of them in order, or empty for none.
        reference_front: Its reference front file, relative to the JVM's working directory.
    """

    name: str
    arguments: tuple[ArgumentValue, ...]
    reference_front: str

    @property
    def label(self) -> str:
        """The problem and its arguments, e.g. "DTLZ2(12, 2)"."""
        return format_problem_spec(problem_spec(self.name, self.arguments))


@dataclass(slots=True, frozen=True)
class Contender:
    """An algorithm with a configuration.

    Attributes:
        name: Its name in the study's tables, unique in the study.
        algorithm: The algorithm's registry name (`BaseAlgorithm.registry_name`).
        configuration: Its configuration, "--parameter value ...".
    """

    name: str
    algorithm: str
    configuration: str


@dataclass(slots=True, frozen=True)
class Job:
    """What one contender does on one problem.

    Attributes:
        number: Its place in the study, from 1.
        contender: The contender.
        problem: The problem.
        directory: Its directory under the study's, holding its request and its status.
        request: The solve request.
    """

    number: int
    contender: Contender
    problem: StudyProblem
    directory: Path
    request: SolveRequest


@dataclass(slots=True, frozen=True)
class SkippedJob:
    """A contender that cannot run on a problem, and why."""

    contender: str
    problem: str
    reason: str


@dataclass(slots=True, frozen=True)
class ValidationStudy:
    """A validation study.

    Attributes:
        encoding: The encoding of the problems.
        problems: The problems, all of that encoding.
        contenders: The contenders; the pivot is one of them.
        pivot: The name of the pivot, the tuned configuration the others are compared with.
        population_size: The population size of every contender.
        max_evaluations: The evaluations of each run.
        runs: The independent runs of each contender on each problem.
        seed: The seed of the first run; run i uses seed + i - 1, the same for every contender.
        indicators: The quality indicators computed on each run.
        weight_vectors_directory: Where the decomposition-based algorithms read their weights.
    """

    encoding: str
    problems: tuple[StudyProblem, ...]
    contenders: tuple[Contender, ...]
    pivot: str
    population_size: int = DEFAULT_POPULATION_SIZE
    max_evaluations: int = DEFAULT_MAX_EVALUATIONS
    runs: int = DEFAULT_RUNS
    seed: int = 1
    indicators: tuple[str, ...] = ("Epsilon", "NormalizedHypervolume")
    weight_vectors_directory: str = DEFAULT_WEIGHT_VECTORS_DIRECTORY

    def validation_errors(self) -> list[str]:
        """Check what the study needs, to tell the user before launching.

        Returns:
            One message per problem found; empty when the study can run.
        """
        names = [contender.name for contender in self.contenders]
        checks = [
            (not self.problems, "Choose at least one problem."),
            (len(self.contenders) < 2, "A study compares at least two contenders."),
            (self.pivot not in names, "The pivot must be one of the contenders."),
            (len(names) != len(set(names)), "Every contender needs its own name."),
            (not self.indicators, "Choose at least one quality indicator."),
            (self.population_size < 2, "The population size must be at least 2."),
            (self.max_evaluations < 1, "The evaluations of each run must be positive."),
            (self.runs < 2, "At least two runs are needed to compare contenders."),
            (
                any(not problem.reference_front for problem in self.problems),
                "Every problem needs a reference front: the indicators use it.",
            ),
        ]
        return [message for failed, message in checks if failed]


def default_contenders(
    encoding: str, default_configuration: Callable[[str], str], constrained: bool = False
) -> list[Contender]:
    """The algorithms that have a default configuration for an encoding, as contenders.

    Args:
        encoding: The encoding of the study's problems.
        default_configuration: Reads a default configuration file of the jar by its name.
        constrained: Whether the study has problems with constraints: the configurations are then
            those that handle them.

    Returns:
        A contender per default configuration (RVEA has three variants), named as the algorithm
        (or its variant), in catalogue order.
    """
    contenders = []
    for algorithm in BASE_ALGORITHMS:
        if not algorithm.runnable_today or encoding not in algorithm.runnable_encodings:
            continue
        defaults = algorithm.defaults(encoding, constrained)
        for label, file_name in defaults:
            variants = len(defaults)
            name = algorithm.name if variants == 1 or label == algorithm.name else label
            configuration = default_configuration(file_name).strip().splitlines()[0]
            contenders.append(Contender(name, algorithm.registry_name, configuration))
    return contenders


def base_algorithm(registry_name: str) -> BaseAlgorithm | None:
    """The catalogue entry of an algorithm, by its registry name."""
    return next((a for a in BASE_ALGORITHMS if a.registry_name == registry_name), None)


def contender_space_file(algorithm: BaseAlgorithm, encoding: str, configuration: str) -> str:
    """The parameter space a configuration is run with: the one for problems with constraints if
    the configuration says how it handles them.

    Args:
        algorithm: Its algorithm.
        encoding: The encoding of the study.
        configuration: The configuration, "--parameter value ...".
    """
    constrained = algorithm.handles_constraints(encoding) and configuration_handles_constraints(
        configuration
    )
    return algorithm.parameter_space_file(encoding, constrained)


def plan_jobs(
    study: ValidationStudy, study_directory: Path, working_directory: Path
) -> tuple[list[Job], list[SkippedJob]]:
    """Plan what the study runs: a job per contender and problem, minus what cannot run.

    A contender that reads weight vectors (MOEA/D, RVEA, NSGA-III) cannot run on a problem when no
    file of the directory matches the population size and the problem's number of objectives.

    Args:
        study: The study.
        study_directory: Where the study is kept (validation-runs/<id>/).
        working_directory: The JVM's working directory, which the request's paths are relative to.

    Returns:
        The jobs, ordered by problem and then contender, and the pairs left out.
    """
    jobs: list[Job] = []
    skipped: list[SkippedJob] = []
    for problem in study.problems:
        for contender in study.contenders:
            reason = _cannot_run(study, contender, problem, working_directory)
            if reason is not None:
                skipped.append(SkippedJob(contender.name, problem.label, reason))
                continue
            number = len(jobs) + 1
            directory = study_directory / JOBS_DIRECTORY_NAME / f"{number:03d}"
            jobs.append(
                Job(
                    number,
                    contender,
                    problem,
                    directory,
                    _request(study, contender, problem, working_directory, directory),
                )
            )
    return jobs, skipped


def build_manifest(study: ValidationStudy, jobs: list[Job], skipped: list[SkippedJob]) -> dict:
    """The manifest of a planned study: what it is, and what each job is.

    Args:
        study: The study.
        jobs: Its jobs, as `plan_jobs` returns them.
        skipped: The pairs left out.

    Returns:
        The manifest, as `write_study` writes it and `read_manifest` reads it back.
    """
    return {
        "encoding": study.encoding,
        "pivot": study.pivot,
        "populationSize": study.population_size,
        "maxEvaluations": study.max_evaluations,
        "runs": study.runs,
        "seed": study.seed,
        "indicators": list(study.indicators),
        "problems": [
            {
                "name": problem.name,
                "arguments": list(problem.arguments),
                "referenceFront": problem.reference_front,
            }
            for problem in study.problems
        ],
        "contenders": [
            {"name": c.name, "algorithm": c.algorithm, "configuration": c.configuration}
            for c in study.contenders
        ],
        "jobs": [
            {
                "number": job.number,
                "contender": job.contender.name,
                "problem": job.problem.label,
                "directory": job.directory.name,
            }
            for job in jobs
        ],
        "skipped": [
            {"contender": s.contender, "problem": s.problem, "reason": s.reason} for s in skipped
        ],
    }


def write_study(
    study: ValidationStudy, study_directory: Path, working_directory: Path
) -> list[Job]:
    """Write a study to disk: its manifest and the request of each job.

    Args:
        study: The study.
        study_directory: Where to keep it, created if missing.
        working_directory: The JVM's working directory.

    Returns:
        The jobs written.
    """
    jobs, skipped = plan_jobs(study, study_directory, working_directory)
    for job in jobs:
        job.directory.mkdir(parents=True, exist_ok=True)
        (job.directory / "request.yaml").write_text(solve_request_to_yaml(job.request))
    manifest = build_manifest(study, jobs, skipped)
    study_directory.mkdir(parents=True, exist_ok=True)
    (study_directory / MANIFEST_NAME).write_text(yaml.safe_dump(manifest, sort_keys=False))
    return jobs


def study_is_written(
    study: ValidationStudy, study_directory: Path, working_directory: Path
) -> bool:
    """Tell whether a directory holds exactly this study, planned the same way.

    Args:
        study: The study.
        study_directory: Where it would be kept.
        working_directory: The JVM's working directory.

    Returns:
        Whether the manifest there is the one `write_study` would write now.
    """
    existing = read_manifest(study_directory)
    if existing is None:
        return False
    jobs, skipped = plan_jobs(study, study_directory, working_directory)
    return existing == build_manifest(study, jobs, skipped)


def read_manifest(study_directory: Path) -> dict | None:
    """Read a study's manifest.

    Args:
        study_directory: The study's directory.

    Returns:
        The manifest, or None if it is missing or not valid.
    """
    try:
        manifest = yaml.safe_load((study_directory / MANIFEST_NAME).read_text())
    except (OSError, yaml.YAMLError):
        return None
    return manifest if isinstance(manifest, dict) and "jobs" in manifest else None


def collect_runs(study_directory: Path, working_directory: Path) -> pd.DataFrame:
    """Gather the runs of every finished job into one table.

    Args:
        study_directory: The study's directory.
        working_directory: The JVM's working directory.

    Returns:
        One row per run of a contender on a problem: `contender`, `problem`, `Run`, `Seed`,
        `TimeMs` and one column per indicator; empty when no job has results yet.
    """
    manifest = read_manifest(study_directory)
    if manifest is None:
        return pd.DataFrame()
    studio_indicators = [n for n in manifest.get("indicators", []) if _computed_by_studio(n)]
    reference_fronts = _reference_fronts_by_problem(manifest)
    tables = []
    for job in manifest["jobs"]:
        output = _job_output(study_directory, job["directory"])
        try:
            table = pd.read_csv(output / INDICATORS_FILE)
        except (OSError, pd.errors.EmptyDataError):
            continue
        if studio_indicators:
            reference_front = working_directory / reference_fronts.get(job["problem"], "")
            computed = studio_indicator_values(
                output, table["Run"].tolist(), reference_front, studio_indicators
            )
            table = table.merge(computed, on="Run", how="left")
        table.insert(0, "problem", job["problem"])
        table.insert(0, "contender", job["contender"])
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def studio_indicator_values(
    output: Path, runs: list[int], reference_front_file: Path, indicators: list[str]
) -> pd.DataFrame:
    """The indicators Evolver-Studio computes, for the runs of a job, from their fronts.

    They are read from `STUDIO_INDICATORS.csv` when it holds them all, and computed and written
    there otherwise; a run whose front or reference front cannot be read gets no value.

    Args:
        output: The job's output directory.
        runs: The numbers of its runs.
        reference_front_file: The reference front of its problem.
        indicators: The registry names of the indicators (e.g. "Hypervolume").

    Returns:
        One row per run: `Run` and a column per indicator, named by its abbreviation (e.g. "HV").
    """
    columns = [_column(name) for name in indicators]
    cache = output / STUDIO_INDICATORS_FILE
    try:
        cached = pd.read_csv(cache)
        if set(columns) <= set(cached.columns) and set(runs) <= set(cached["Run"]):
            return cached[["Run", *columns]]
    except (OSError, pd.errors.EmptyDataError, KeyError):
        pass
    try:
        reference_front = pd.read_csv(reference_front_file, header=None).to_numpy()
    except (OSError, pd.errors.EmptyDataError):
        reference_front = None
    rows = []
    for run in runs:
        row: dict[str, float | int | None] = {"Run": run}
        front = _read_feasible_front(output, run)
        for name, column in zip(indicators, columns, strict=True):
            usable = front is not None and reference_front is not None
            row[column] = (
                STUDIO_INDICATOR_FUNCTIONS[name](front, reference_front) if usable else None
            )
        rows.append(row)
    values = pd.DataFrame(rows, columns=["Run", *columns])
    try:
        values.to_csv(cache, index=False)
    except OSError:
        pass
    return values


def read_study_front(
    study_directory: Path, manifest: dict, contender: str, problem: str, run: int
) -> pd.DataFrame | None:
    """The front a run of a contender found on a problem.

    Args:
        study_directory: The study's directory.
        manifest: Its manifest.
        contender: The contender's name.
        problem: The problem's label, as in the jobs (e.g. "DTLZ2(12, 3)").
        run: The run's number.

    Returns:
        The objective values of its solutions (only the feasible ones, on a problem with
        constraints), with columns f1, f2, ...; None when the job or the run has no front, or no
        feasible solution.
    """
    job = next(
        (j for j in manifest["jobs"] if j["contender"] == contender and j["problem"] == problem),
        None,
    )
    if job is None:
        return None
    front = _read_feasible_front(_job_output(study_directory, job["directory"]), run)
    if front is None or len(front) == 0:
        return None
    return _as_objectives(front)


def feasibility(study_directory: Path) -> pd.DataFrame:
    """Whether each run on a problem with constraints found feasible solutions.

    Args:
        study_directory: The study's directory.

    Returns:
        One row per run of a contender on a problem with constraints: `contender`, `problem`,
        `Run`, `feasible` and `violation`, the least of its solutions (0 if it is feasible). Empty
        when no problem of the study has constraints.
    """
    columns = ["contender", "problem", "Run", "feasible", "violation"]
    manifest = read_manifest(study_directory)
    rows = []
    for job in manifest["jobs"] if manifest is not None else []:
        output = _job_output(study_directory, job["directory"])
        for run in range(1, manifest.get("runs", 0) + 1):
            violations = read_violations(output, run)
            if violations is not None:
                least = float(violations.min())
                rows.append((job["contender"], job["problem"], run, least <= 0, max(least, 0.0)))
    return pd.DataFrame(rows, columns=columns)


def read_study_reference_front(
    manifest: dict, problem: str, working_directory: Path
) -> pd.DataFrame | None:
    """The reference front of a problem of a study, with columns f1, f2, ...; None if missing."""
    file = _reference_fronts_by_problem(manifest).get(problem)
    return _read_objectives(working_directory / file) if file else None


def _read_objectives(front_file: Path) -> pd.DataFrame | None:
    front = _read_front(front_file)
    return None if front is None else _as_objectives(front)


def _as_objectives(front) -> pd.DataFrame:
    return pd.DataFrame(front, columns=[f"f{i}" for i in range(1, front.shape[1] + 1)])


def _read_feasible_front(output: Path, run: int):
    """The front of a run without its infeasible solutions, if the problem has constraints."""
    front = _read_front(output / f"run-{run}" / "FUN.csv")
    violations = read_violations(output, run)
    if front is None or violations is None or len(violations) != len(front):
        return front
    return front[(violations <= 0).to_numpy()]


def _computed_by_studio(name: str) -> bool:
    indicator = quality_indicator(name)
    return indicator is not None and indicator.computed_by_studio


def _column(name: str) -> str:
    indicator = quality_indicator(name)
    return indicator.short_name if indicator is not None else name


def _reference_fronts_by_problem(manifest: dict) -> dict[str, str]:
    """The reference front of each problem, by its label in the jobs (e.g. "DTLZ2(12, 2)")."""
    return {
        StudyProblem(p["name"], tuple(p.get("arguments") or ()), p["referenceFront"]).label: p[
            "referenceFront"
        ]
        for p in manifest.get("problems", [])
    }


def _read_front(front_file: Path):
    try:
        return pd.read_csv(front_file, header=None).to_numpy()
    except (OSError, pd.errors.EmptyDataError):
        return None


def _job_output(study_directory: Path, job_directory: str) -> Path:
    return study_directory / JOBS_DIRECTORY_NAME / job_directory / OUTPUT_DIRECTORY_NAME


def _cannot_run(
    study: ValidationStudy, contender: Contender, problem: StudyProblem, working_directory: Path
) -> str | None:
    algorithm = base_algorithm(contender.algorithm)
    if (
        algorithm is None
        or "weightVectorFilesDirectory" not in algorithm.required_extra_config_keys
    ):
        return None
    objectives = reference_front_dimension(working_directory / problem.reference_front)
    if objectives is None:
        return None
    sizes = available_population_sizes(
        working_directory / study.weight_vectors_directory, objectives
    )
    if study.population_size in sizes:
        return None
    return (
        f"no weight vector file for population size {study.population_size} "
        f"and {objectives} objectives"
    )


def _request(
    study: ValidationStudy,
    contender: Contender,
    problem: StudyProblem,
    working_directory: Path,
    job_directory: Path,
) -> SolveRequest:
    algorithm = base_algorithm(contender.algorithm)
    assert algorithm is not None, f"unknown algorithm {contender.algorithm}"
    extra_config = (
        {"weightVectorFilesDirectory": study.weight_vectors_directory}
        if "weightVectorFilesDirectory" in algorithm.required_extra_config_keys
        else None
    )
    output = job_directory / OUTPUT_DIRECTORY_NAME
    try:
        relative_output = output.relative_to(working_directory)
    except ValueError:
        relative_output = output
    return SolveRequest(
        algorithm_name=algorithm.registry_name,
        encoding=study.encoding,
        population_size=study.population_size,
        yaml_parameter_space_file=contender_space_file(
            algorithm, study.encoding, contender.configuration
        ),
        extra_config=extra_config,
        configuration=contender.configuration,
        problem=problem.name,
        problem_arguments=problem.arguments,
        reference_front_file_name=problem.reference_front,
        max_evaluations=study.max_evaluations,
        number_of_independent_runs=study.runs,
        seed=study.seed,
        # Evolver computes only those it registers; the others are computed from the fronts.
        indicator_names=[name for name in study.indicators if not _computed_by_studio(name)],
        status_frequency=None,
        front_frequency=None,
        write_population=False,
        output_directory=str(relative_output),
    )


@dataclass(slots=True, frozen=True)
class StudyInfo:
    """A study that was run, as listed for reopening.

    Attributes:
        study_id: Its timestamp-based identifier (its directory's name).
        directory: Its directory.
        manifest: Its manifest.
    """

    study_id: str
    directory: Path
    manifest: dict

    @property
    def label(self) -> str:
        """The study in one line, e.g. "20261007-120000 · NSGA-II (tuned) · 9 problems"."""
        problems = len(self.manifest.get("problems", ()))
        return f"{self.study_id} · {self.manifest.get('pivot', '?')} · {problems} problems"


def list_studies(runs_directory: Path) -> list[StudyInfo]:
    """List the studies kept in a directory, most recent first.

    Args:
        runs_directory: The directory whose subdirectories are studies (validation-runs/).

    Returns:
        The studies whose manifest can be read.
    """
    if not runs_directory.is_dir():
        return []
    studies = []
    for directory in sorted(runs_directory.iterdir(), reverse=True):
        manifest = read_manifest(directory)
        if manifest is not None:
            studies.append(StudyInfo(directory.name, directory, manifest))
    return studies
