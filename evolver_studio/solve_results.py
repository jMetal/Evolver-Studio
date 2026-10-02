"""Reading the results of a solve run (cli.solving): fronts, indicators and past runs.

A run's output directory holds `run-<i>/VAR.csv` and `FUN.csv` for each independent run,
`INDICATORS.csv` (`Run,Seed,TimeMs,<indicators>`, one row per run) and `METADATA.txt`; the run's own
directory, next to its request, holds `status.yaml`.
"""

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from evolver_studio.evolver_client import RunState, read_status

RUN_DIRECTORY_PATTERN = re.compile(r"run-(\d+)")
INDICATORS_FILE = "INDICATORS.csv"
METADATA_FILE = "METADATA.txt"
# INDICATORS.csv's columns that are not indicators.
RUN_COLUMNS = ("Run", "Seed", "TimeMs")


@dataclass(slots=True, frozen=True)
class SolveRunInfo:
    """A past solve run, as listed for reopening.

    Attributes:
        run_id: The run's timestamp-based identifier (its directory's name).
        run_dir: The directory holding its request and status files.
        output_directory: The directory holding its results.
        algorithm: The algorithm's name.
        problem: The problem's name.
        state: Its state, or None if its status file is missing or unreadable.
    """

    run_id: str
    run_dir: Path
    output_directory: Path
    algorithm: str
    problem: str
    state: RunState | None


def read_run_fronts(output_directory: Path) -> dict[int, pd.DataFrame]:
    """Read the front each independent run found.

    Args:
        output_directory: A solve run's output directory.

    Returns:
        The objective values of each run, by run number (from 1), with columns f1, f2, ...; only
        the runs whose FUN.csv exists.
    """
    fronts: dict[int, pd.DataFrame] = {}
    for directory in output_directory.iterdir():
        match = RUN_DIRECTORY_PATTERN.fullmatch(directory.name)
        fun_file = directory / "FUN.csv"
        if match and fun_file.is_file():
            fronts[int(match.group(1))] = read_front(fun_file)
    return dict(sorted(fronts.items()))


def read_front(front_file: Path) -> pd.DataFrame:
    """Read a front file: one solution per row, one objective per column, no header.

    Args:
        front_file: A FUN.csv or a reference front file.

    Returns:
        The objective values, with columns f1, f2, ...
    """
    front = pd.read_csv(front_file, header=None)
    front.columns = [f"f{index}" for index in range(1, front.shape[1] + 1)]
    return front


def read_indicators(output_directory: Path) -> pd.DataFrame:
    """Read INDICATORS.csv: one row per run, with its seed, time and indicator values.

    Args:
        output_directory: A solve run's output directory.

    Returns:
        The table, with the columns Run, Seed, TimeMs and one per indicator.
    """
    return pd.read_csv(output_directory / INDICATORS_FILE)


def indicator_summary(indicators: pd.DataFrame) -> pd.DataFrame:
    """Summarize each indicator, and the time, over the runs.

    Args:
        indicators: The table `read_indicators` returns.

    Returns:
        One row per indicator (and TimeMs), with the mean, standard deviation, minimum and maximum
        over the runs; the standard deviation is empty for a single run.
    """
    columns = [column for column in indicators.columns if column not in ("Run", "Seed")]
    summary = indicators[columns].agg(["mean", "std", "min", "max"]).T
    summary.index.name = "Indicator"
    return summary


def zip_fronts(output_directory: Path) -> bytes:
    """Pack the results of a run (every VAR and FUN file, the indicators and the metadata).

    Args:
        output_directory: A solve run's output directory.

    Returns:
        The content of a zip file, with paths relative to the output directory.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_directory))
    return buffer.getvalue()


def list_solve_runs(runs_directory: Path, working_directory: Path) -> list[SolveRunInfo]:
    """List the solve runs stored under a runs directory, the most recent first.

    Args:
        runs_directory: The directory whose subdirectories are runs (solve-runs/).
        working_directory: The JVM's working directory, which a request's relative
            `outputDirectory` is resolved against.

    Returns:
        The runs whose request file can be read.
    """
    if not runs_directory.is_dir():
        return []
    runs = []
    for run_dir in sorted(runs_directory.iterdir(), reverse=True):
        try:
            request = yaml.safe_load((run_dir / "request.yaml").read_text())
            status = read_status(run_dir / "status.yaml")
            runs.append(
                SolveRunInfo(
                    run_id=run_dir.name,
                    run_dir=run_dir,
                    output_directory=working_directory / request["outputDirectory"],
                    algorithm=request["algorithmName"],
                    problem=request["problem"],
                    state=status.state if status is not None else None,
                )
            )
        except (OSError, yaml.YAMLError, KeyError, TypeError):
            continue
    return runs
