"""Running a training on a cluster managed by Slurm, and bringing its results back.

A training on expensive problems can take days, longer than a laptop can be left running. Instead
of launching it, the Training page can write it as a **package**: a zip with everything a cluster
needs to run it, without Evolver-Studio or a checkout of Evolver there:

    <experiment>.<run id>/
        Evolver-<version>-jar-with-dependencies.jar
        experiment/           request.yaml, base_level.yaml, meta_search.yaml and the
                              parameter space, with paths relative to the package
        resources/            the reference fronts, weight vectors and instances it reads
        train.sbatch          one replication, as a task of a Slurm job array
        submit.sh             submits the replications that are not done yet
        README.txt            the steps, for whoever runs it
        studio-package.yaml   what Evolver-Studio needs to import the results

Each replication is an ordinary training (`TrainingRunnerMain --output-dir results/runNN`), as in
Evolver's own scripts/slurm/ (tutorial E20), from which the two scripts are adapted. Once the
package folder is back, with its results/ and logs/, `import_cluster_results` turns each finished
replication into a training run of this app, under cli-runner-runs/ as if it had been launched
here, so Training analysis, Validation and Run algorithm take it like any other.
"""

import datetime as dt
import importlib.metadata
import re
import shutil
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from evolver_studio.evolver_client import EVOLVER_VERSION, RunState, read_status
from evolver_studio.request import (
    BaseLevelConfig,
    MetaSearchConfig,
    base_level_to_yaml,
    meta_search_to_yaml,
    request_to_yaml,
)
from evolver_studio.result import Err, Ok
from evolver_studio.runs import RUNS_DIRECTORY_NAME
from evolver_studio.training_choices import experiment_directory

EXPORTS_DIRECTORY_NAME = "cluster-exports"
MANIFEST_FILE_NAME = "studio-package.yaml"
MANIFEST_FORMAT = 1
EXPERIMENT_DIRECTORY = "experiment"
PARAMETER_SPACE_FILE_NAME = "base_parameter_space.yaml"
EXTERNAL_RESOURCES_DIRECTORY = "resources/external"
DEFAULT_JAVA_SETUP = "module load java/jdk-25"
DEFAULT_TIME_LIMIT = "100:00:00"
DEFAULT_MEMORY = "64gb"

_REPLICATION = re.compile(r"^run(\d{2,})$")
# Slurm's --time: minutes, MM:SS, HH:MM:SS, D-HH, D-HH:MM or D-HH:MM:SS.
_TIME_LIMIT = re.compile(r"^(\d+-)?\d+(:\d{1,2}){0,2}$")
_MEMORY = re.compile(r"^\d+[KMGTkmgt]?[Bb]?$")


@dataclass(slots=True, frozen=True)
class SlurmSettings:
    """What the batch script asks Slurm for, beyond the cores, which the training fixes.

    Attributes:
        replications: Independent replications of the training, one task of the job array each.
        time_limit: The time limit of each task (`--time`), e.g. "100:00:00" or "4-00:00:00".
        memory: The memory of each task (`--mem`), e.g. "64gb".
        java_setup: The command that puts Java 21 or later on the path of a node (e.g. "module load
            java/jdk-25"); empty if it already is.
        extra_options: More sbatch options, one per line (e.g. "--partition=long").
    """

    replications: int = 1
    time_limit: str = DEFAULT_TIME_LIMIT
    memory: str = DEFAULT_MEMORY
    java_setup: str = DEFAULT_JAVA_SETUP
    extra_options: str = ""


@dataclass(slots=True, frozen=True)
class TrainingPackage:
    """A training, as the Training page defines it, to be run elsewhere.

    Attributes:
        experiment: The name of the experiment.
        run_id: The timestamp-based identifier of the export.
        results_folder: Where the app keeps its results, e.g. "results/NSGA-II"; the imported
            replications go to `<results_folder>/<experiment>.<run_id>-runNN`.
        base_level: The base-level configuration, with the paths of this app's working directory.
        meta_search: The meta-search configuration.
        parameter_space_text: The base algorithm's parameter space, as edited.
        write_frequency: How often, in meta-evaluations, Evolver writes its files and its status.
        write_population: Whether Evolver also writes the meta-optimizer's whole population.
    """

    experiment: str
    run_id: str
    results_folder: str
    base_level: BaseLevelConfig
    meta_search: MetaSearchConfig
    parameter_space_text: str
    write_frequency: int
    write_population: bool

    @property
    def name(self) -> str:
        """The name of the package, its zip and its folder, e.g. "ZDT4.20261126-120000"."""
        return f"{self.experiment}.{self.run_id}"


@dataclass(slots=True, frozen=True)
class ImportReport:
    """What importing a package did.

    Attributes:
        imported: The run identifiers of the replications imported now.
        skipped: A line for each replication left out, saying why.
    """

    imported: tuple[str, ...]
    skipped: tuple[str, ...]


def slurm_settings_errors(settings: SlurmSettings) -> list[str]:
    """What is wrong with the settings of the batch script; empty if nothing is."""
    errors = []
    if settings.replications < 1:
        errors.append("The number of replications must be at least 1.")
    if not _TIME_LIMIT.match(settings.time_limit.strip()):
        errors.append(
            "The time limit must be in a format of Slurm's --time, e.g. 100:00:00 (hours, minutes "
            "and seconds) or 4-00:00:00 (days first)."
        )
    if not _MEMORY.match(settings.memory.strip()):
        errors.append("The memory must be a number with an optional unit, e.g. 64gb or 32000M.")
    for option in _extra_options(settings):
        if not option.startswith("-"):
            errors.append(f"Each extra sbatch option must start with a hyphen: {option!r}.")
    return errors


def write_cluster_package(
    package: TrainingPackage,
    settings: SlurmSettings,
    jar: Path,
    working_directory: Path,
    destination: Path,
) -> Path:
    """Write the zip of a package.

    Args:
        package: The training.
        settings: What the batch script asks Slurm for.
        jar: Evolver's jar, which goes in the package.
        working_directory: This app's working directory, against which the relative paths of the
            training (reference fronts, weight vectors, instances) are resolved.
        destination: The directory of the zip.

    Returns:
        The zip's path.

    Raises:
        FileNotFoundError: A file the training reads does not exist.
    """
    base_level, files, path_map = _portable_base_level(package.base_level, working_directory)
    root = package.name
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{root}.zip"
    partial = target.with_name(target.name + ".part")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(jar, f"{root}/{jar.name}", compress_type=zipfile.ZIP_STORED)
        for source, name in files:
            archive.write(source, f"{root}/{name}")
        texts = {
            f"{EXPERIMENT_DIRECTORY}/{PARAMETER_SPACE_FILE_NAME}": package.parameter_space_text,
            f"{EXPERIMENT_DIRECTORY}/base_level.yaml": base_level_to_yaml(base_level),
            f"{EXPERIMENT_DIRECTORY}/meta_search.yaml": meta_search_to_yaml(package.meta_search),
            f"{EXPERIMENT_DIRECTORY}/request.yaml": request_to_yaml(
                f"{EXPERIMENT_DIRECTORY}/base_level.yaml",
                f"{EXPERIMENT_DIRECTORY}/meta_search.yaml",
                "results/run01",
                write_frequency=package.write_frequency,
                status_frequency=package.write_frequency,
                write_population=package.write_population,
            ),
            "README.txt": _readme(package, settings, jar.name),
            MANIFEST_FILE_NAME: yaml.safe_dump(
                _manifest(package, settings, path_map), sort_keys=False
            ),
        }
        executables = {
            "train.sbatch": _batch_script(package, settings, jar.name),
            "submit.sh": _submit_script(package, settings),
        }
        for name, text in [*texts.items(), *executables.items()]:
            info = zipfile.ZipInfo(f"{root}/{name}", date_time=dt.datetime.now().timetuple()[:6])
            info.external_attr = (0o100755 if name in executables else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, text)
    partial.replace(target)
    return target


def import_cluster_results(package_directory: Path, working_directory: Path) -> Ok | Err:
    """Make each finished replication of a package a training run of this app.

    A replication is imported once: one already imported, unfinished or failed is left out, and
    said why, so importing the folder again after more replications finish takes only the new ones.

    Args:
        package_directory: The package's folder, back from the cluster with its results/ (and,
            if there is one, logs/).
        working_directory: This app's working directory.

    Returns:
        Ok(ImportReport), or Err(message) if the folder is not a package of this app.
    """
    manifest_file = package_directory / MANIFEST_FILE_NAME
    try:
        manifest = yaml.safe_load(manifest_file.read_text())
        experiment_dir = package_directory / EXPERIMENT_DIRECTORY
        request = yaml.safe_load((experiment_dir / "request.yaml").read_text())
        base_level = yaml.safe_load((experiment_dir / "base_level.yaml").read_text())
        meta_search_text = (experiment_dir / "meta_search.yaml").read_text()
        parameter_space_text = (experiment_dir / PARAMETER_SPACE_FILE_NAME).read_text()
        experiment, export_id = manifest["experiment"], manifest["runId"]
        results_folder = manifest["resultsFolder"]
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return Err(
            f"{package_directory} is not the folder of a package exported by Evolver-Studio: it "
            f"needs {MANIFEST_FILE_NAME} and the {EXPERIMENT_DIRECTORY}/ folder."
        )
    results_dir = package_directory / "results"
    replications = (
        sorted(
            (d for d in results_dir.iterdir() if d.is_dir() and _REPLICATION.match(d.name)),
            key=lambda d: int(_REPLICATION.match(d.name).group(1)),
        )
        if results_dir.is_dir()
        else []
    )
    if not replications:
        return Err(f"{results_dir} has no replication (run01, run02...): has the training run?")
    path_map = manifest.get("pathMap") or {}
    imported, skipped = [], []
    for replication in replications:
        run_id = f"{export_id}-{replication.name}"
        status = read_status(replication / "status.yaml")
        if status is None or status.state != RunState.FINISHED:
            why = "has no status yet" if status is None else f"is {status.state.value}"
            skipped.append(
                f"{replication.name}: it has not finished (it {why}), so it was not imported."
            )
            continue
        run_dir = working_directory / RUNS_DIRECTORY_NAME / run_id
        output = experiment_directory(results_folder, experiment, run_id)
        if run_dir.exists() or (working_directory / output).exists():
            skipped.append(f"{replication.name}: already imported, as {run_id}.")
            continue
        shutil.copytree(replication, working_directory / output)
        run_dir.mkdir(parents=True)
        (run_dir / PARAMETER_SPACE_FILE_NAME).write_text(parameter_space_text)
        local_base_level = _local_base_level(base_level, path_map, run_dir)
        (run_dir / "base_level.yaml").write_text(yaml.safe_dump(local_base_level, sort_keys=False))
        (run_dir / "meta_search.yaml").write_text(meta_search_text)
        (run_dir / "request.yaml").write_text(
            request_to_yaml(
                str(run_dir / "base_level.yaml"),
                str(run_dir / "meta_search.yaml"),
                output,
                write_frequency=request.get("writeFrequency"),
                status_frequency=request.get("statusFrequency"),
                write_population=bool(request.get("writePopulation")),
            )
        )
        shutil.copy2(replication / "status.yaml", run_dir / "status.yaml")
        _write_results_pointer(replication, f"results/{replication.name}", output, run_dir)
        _write_log(package_directory / "logs", replication.name, run_dir / "runner.log")
        imported.append(run_id)
    return Ok(ImportReport(tuple(imported), tuple(skipped)))


def studio_version() -> str:
    """This app's version, or "unknown" when it is not installed as a package."""
    try:
        return importlib.metadata.version("evolver-studio")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _portable_base_level(
    base_level: BaseLevelConfig, working_directory: Path
) -> tuple[BaseLevelConfig, list[tuple[Path, str]], dict[str, str]]:
    """The base level with paths inside the package, the files to copy into it, and the paths it
    changed (package path -> original), to restore them when the results are imported.

    A relative path keeps its place, under the package as under the working directory; an absolute
    one is copied to resources/external/.
    """
    files: list[tuple[Path, str]] = []
    path_map: dict[str, str] = {}

    def portable(path: str, directory: bool = False) -> str:
        source = Path(path) if Path(path).is_absolute() else working_directory / path
        if not (source.is_dir() if directory else source.is_file()):
            raise FileNotFoundError(f"The training reads {path}, which does not exist.")
        name = path
        if Path(path).is_absolute():
            name = f"{EXTERNAL_RESOURCES_DIRECTORY}/{len(path_map) + 1}-{source.name}"
            path_map[name] = path
        if directory:
            files.extend(
                (file, f"{name}/{file.relative_to(source).as_posix()}")
                for file in sorted(source.rglob("*"))
                if file.is_file()
            )
        else:
            files.append((source, name))
        return name

    fronts = [portable(front) for front in base_level.training_reference_front_file_names]
    extra = dict(base_level.extra_config) if base_level.extra_config else None
    if extra and extra.get("weightVectorFilesDirectory"):
        extra["weightVectorFilesDirectory"] = portable(
            extra["weightVectorFilesDirectory"], directory=True
        )
    problems = [
        _portable_problem(problem, working_directory, portable)
        for problem in base_level.training_problem_names
    ]
    portable_base_level = replace(
        base_level,
        yaml_parameter_space_file=f"{EXPERIMENT_DIRECTORY}/{PARAMETER_SPACE_FILE_NAME}",
        training_reference_front_file_names=fronts,
        extra_config=extra,
        training_problem_names=problems,
    )
    return portable_base_level, _unique(files), path_map


def _portable_problem(problem: str | dict, working_directory: Path, portable) -> str | dict:
    """A problem of the training set, with the files its arguments name (e.g. a TSP instance)
    made part of the package."""
    if not isinstance(problem, dict) or not isinstance(problem.get("args"), list):
        return problem
    args = [
        portable(arg)
        if isinstance(arg, str)
        and (Path(arg).is_absolute() or "/" in arg)
        and (working_directory / arg).is_file()
        else arg
        for arg in problem["args"]
    ]
    return {**problem, "args": args}


def _unique(files: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    seen: set[str] = set()
    return [(s, n) for s, n in files if not (n in seen or seen.add(n))]


def _local_base_level(base_level: dict, path_map: dict[str, str], run_dir: Path) -> dict:
    """The base level of an imported replication: the parameter space in its run folder and the
    paths the package changed back to the original ones."""

    def local(value):
        if isinstance(value, str):
            return path_map.get(value, value)
        if isinstance(value, list):
            return [local(item) for item in value]
        if isinstance(value, dict):
            return {key: local(item) for key, item in value.items()}
        return value

    restored = local(base_level)
    restored["yamlParameterSpaceFile"] = str(run_dir / PARAMETER_SPACE_FILE_NAME)
    return restored


def _write_results_pointer(replication: Path, old: str, new: str, run_dir: Path) -> None:
    """The results.yaml of an imported replication, pointing to where its results are now."""
    try:
        pointer = yaml.safe_load((replication / "results.yaml").read_text())
    except (OSError, yaml.YAMLError):
        return
    if isinstance(pointer, dict):
        moved = {
            key: value.replace(old, new, 1) if isinstance(value, str) else value
            for key, value in pointer.items()
        }
        (run_dir / "results.yaml").write_text(yaml.safe_dump(moved, sort_keys=False))


def _write_log(logs_directory: Path, replication: str, log_file: Path) -> None:
    """The runner.log of an imported replication: the output and errors of its last Slurm task."""
    task = int(_REPLICATION.match(replication).group(1))
    if not logs_directory.is_dir():
        return
    outputs = sorted(logs_directory.glob(f"*_{task}.out"), key=lambda path: path.stat().st_mtime)
    if not outputs:
        return
    output = outputs[-1]
    errors = output.with_suffix(".err")
    text = output.read_text(errors="replace")
    if errors.is_file():
        text += errors.read_text(errors="replace")
    log_file.write_text(text)


def _extra_options(settings: SlurmSettings) -> list[str]:
    return [line.strip() for line in settings.extra_options.splitlines() if line.strip()]


def _manifest(package: TrainingPackage, settings: SlurmSettings, path_map: dict) -> dict:
    return {
        "format": MANIFEST_FORMAT,
        "experiment": package.experiment,
        "runId": package.run_id,
        "resultsFolder": package.results_folder,
        "replications": settings.replications,
        "evolverVersion": EVOLVER_VERSION,
        "studioVersion": studio_version(),
        "createdAt": dt.datetime.now().isoformat(timespec="seconds"),
        "pathMap": path_map,
    }


def _fill(template: str, values: dict[str, object]) -> str:
    for key, value in values.items():
        template = template.replace(f"@@{key}@@", str(value))
    return template


def _batch_script(package: TrainingPackage, settings: SlurmSettings, jar_name: str) -> str:
    extra = "".join(f"#SBATCH {option}\n" for option in _extra_options(settings))
    return _fill(
        _BATCH_TEMPLATE,
        {
            "NAME": package.name,
            "STUDIO_VERSION": studio_version(),
            "JOB": f"evolver_{package.experiment}",
            "CORES": package.meta_search.number_of_cores,
            "MEMORY": settings.memory.strip(),
            "TIME": settings.time_limit.strip(),
            "EXTRA": extra,
            "JAVA_SETUP": settings.java_setup.strip().replace('"', '\\"'),
            "JAR": jar_name,
        },
    )


def _submit_script(package: TrainingPackage, settings: SlurmSettings) -> str:
    return _fill(_SUBMIT_TEMPLATE, {"NAME": package.name, "REPLICATIONS": settings.replications})


def _readme(package: TrainingPackage, settings: SlurmSettings, jar_name: str) -> str:
    cores = package.meta_search.number_of_cores
    return _fill(
        _README_TEMPLATE,
        {
            "NAME": package.name,
            "ALGORITHM": f"{package.base_level.algorithm_name} ({package.base_level.encoding})",
            "PROBLEMS": ", ".join(
                p if isinstance(p, str) else str(p.get("class", p))
                for p in package.base_level.training_problem_names
            ),
            "META": package.meta_search.algorithm,
            "REPLICATIONS": settings.replications,
            "CORES": cores,
            "MEMORY": settings.memory.strip(),
            "TIME": settings.time_limit.strip(),
            "JAVA_SETUP": settings.java_setup.strip() or "(nothing: Java is on the path)",
            "JAR": jar_name,
            "EVOLVER_VERSION": EVOLVER_VERSION,
        },
    )


_BATCH_TEMPLATE = """#!/usr/bin/env bash
#
# One replication of the training @@NAME@@, as a task of a Slurm job array: task i writes its
# results to results/runNN (NN = i with two digits). Submit the replications with ./submit.sh, and
# see README.txt. Written by Evolver-Studio @@STUDIO_VERSION@@, adapted from Evolver's
# scripts/slurm/train_replicas.sbatch (tutorial E20).
#
# Without Slurm, it runs one replication by hand: TASK=1 bash train.sbatch
#
# ---- Adapt to your cluster ---------------------------------------------------------------------
# The cores must stay as they are: they are the numberOfCores of experiment/meta_search.yaml, which
# the meta-optimizer uses whatever Slurm gives it. Options given to sbatch override these lines.
#SBATCH --job-name=@@JOB@@
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=@@CORES@@
#SBATCH --mem=@@MEMORY@@
#SBATCH --time=@@TIME@@
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
@@EXTRA@@
# How to put Java (21 or later) on the path of a node; empty if it already is.
JAVA_SETUP=${JAVA_SETUP-"@@JAVA_SETUP@@"}
# A fast local disk of the node to run on (the results are copied back); empty to run in place.
SCRATCH_DIR=${SCRATCH_DIR-${LOCALSCRATCH:+$LOCALSCRATCH/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}}}
# Java options, e.g. "-Xmx48g".
JAVA_OPTS=${JAVA_OPTS:-}
# ------------------------------------------------------------------------------------------------

set -euo pipefail

# Slurm runs a copy of this script; the package is where it was submitted from.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
JAR=@@JAR@@
TASK=${SLURM_ARRAY_TASK_ID:-${TASK:?"set TASK (the replication number) when running without Slurm"}}
RUN=$(printf "run%02d" "$TASK")
OUT=results/$RUN

if [ -f "$OUT/results.yaml" ]; then
    echo "Replication $RUN is already done: $OUT"
    exit 0
fi

if [ -n "$JAVA_SETUP" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    eval "$JAVA_SETUP"
fi
if ! command -v java > /dev/null; then
    echo "No java on the path: set JAVA_SETUP in train.sbatch" >&2
    exit 1
fi
JAVA_VERSION=$(java -XshowSettings:properties -version 2>&1 \\
    | awk -F'= ' '/java.specification.version/ {print $2}')
if [ "${JAVA_VERSION%%.*}" -lt 21 ] 2> /dev/null; then
    echo "Evolver needs Java 21 or later, and the java on the path is $JAVA_VERSION" >&2
    exit 1
fi

HERE=$PWD
# A replication that did not finish leaves files behind, and Evolver does not write over them.
rm -rf "$OUT"
if [ -n "$SCRATCH_DIR" ]; then
    mkdir -p "$SCRATCH_DIR"
    cp "$JAR" "$SCRATCH_DIR/"
    cp -R resources experiment "$SCRATCH_DIR/"
    cd "$SCRATCH_DIR"
fi

CORES=$(nproc 2> /dev/null || sysctl -n hw.ncpu)
echo "Replication $RUN of @@NAME@@ on $(hostname), $CORES cores, Java $JAVA_VERSION, $(date)"
STATUS=0
# shellcheck disable=SC2086
java $JAVA_OPTS -cp "$JAR" org.uma.evolver.cli.training.TrainingRunnerMain \\
    experiment/request.yaml --output-dir "$OUT" || STATUS=$?

if [ -n "$SCRATCH_DIR" ]; then
    mkdir -p "$HERE/results"
    if [ -d "$OUT" ]; then
        cp -R "$OUT" "$HERE/results/"
    fi
    cd "$HERE"
    rm -rf "${SCRATCH_DIR:?}"
fi
echo "Replication $RUN ended with status $STATUS, $(date): $OUT"
exit "$STATUS"
"""

_SUBMIT_TEMPLATE = """#!/usr/bin/env bash
#
# Submits the replications of the training @@NAME@@ that are not done yet, as a Slurm job array of
# train.sbatch: one task per replication. Run it again to submit only the missing ones (those that
# failed or were cancelled). More sbatch options can be given, e.g. ./submit.sh --partition=long
# Written by Evolver-Studio, adapted from Evolver's scripts/slurm/submit_replicas.sh.

set -euo pipefail
cd "$(dirname "$0")"

REPLICATIONS=@@REPLICATIONS@@
MISSING=()
for i in $(seq 1 "$REPLICATIONS"); do
    [ -f "results/$(printf "run%02d" "$i")/results.yaml" ] || MISSING+=("$i")
done
if [ ${#MISSING[@]} -eq 0 ]; then
    echo "All $REPLICATIONS replications of @@NAME@@ are done"
    exit 0
fi
ARRAY=$(IFS=,; echo "${MISSING[*]}")

mkdir -p logs
echo "Submitting ${#MISSING[@]} of $REPLICATIONS replications of @@NAME@@: tasks $ARRAY"
sbatch --array="$ARRAY" "$@" train.sbatch
"""

_README_TEMPLATE = """@@NAME@@
================================================================================

A training of Evolver @@EVOLVER_VERSION@@, written by Evolver-Studio to be run on a cluster managed
by Slurm:

    Base algorithm   @@ALGORITHM@@
    Problems         @@PROBLEMS@@
    Meta-optimizer   @@META@@
    Replications     @@REPLICATIONS@@, each with @@CORES@@ cores, @@MEMORY@@ and a limit of @@TIME@@

The package has everything it needs: Evolver's jar (@@JAR@@), the training
(experiment/), the files it reads (resources/) and the scripts. The cluster only needs Java 21 or
later; train.sbatch puts it on the path with: @@JAVA_SETUP@@


1. Copy the package to the cluster and unzip it

    scp @@NAME@@.zip user@cluster:
    ssh user@cluster
    unzip @@NAME@@.zip
    cd @@NAME@@


2. Check the top of train.sbatch

The lines between "Adapt to your cluster" and the dashes depend on the cluster: memory, time
limit, partition, how Java is loaded (JAVA_SETUP) and the local disk of the nodes (SCRATCH_DIR).
Keep --cpus-per-task: it is the number of cores the training uses.

To try it before submitting, run one replication by hand with a short training (not on the login
node of a cluster with a long one):

    TASK=1 bash train.sbatch


3. Submit the replications

    ./submit.sh

Each replication is a task of a job array and writes its results to results/run01,
results/run02... Its progress is in results/runNN/status.yaml, and its output in logs/.

    squeue -u $USER
    cat results/run01/status.yaml

Running ./submit.sh again submits only the replications that are not done (failed, cancelled or
stopped by the time limit).


4. Bring the results back

Copy the whole folder back to the computer where Evolver-Studio runs (results/ and logs/ are
what it needs):

    scp -r user@cluster:@@NAME@@ .

and import it in Evolver-Studio, in Training analysis > Import a training run on a cluster, giving
the folder. Each finished replication becomes a training run of the app, ready to analyze and
validate. Importing the folder again later adds only the replications that were not imported.
"""
