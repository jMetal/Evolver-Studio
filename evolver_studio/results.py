"""Parsing of Evolver's training results (results.yaml, METADATA.txt, CSVs)."""

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass(slots=True, frozen=True)
class ResultsPointer:
    """Parsed contents of results.yaml, plus the VAR_CONF.txt path it omits.

    Attributes:
        output_directory: Directory holding all result files.
        metadata_file: Path to METADATA.txt.
        indicators_file: Path to INDICATORS.csv.
        configurations_file: Path to CONFIGURATIONS.csv.
        var_conf_file: Path to VAR_CONF.txt (not listed in results.yaml itself).
    """

    output_directory: Path
    metadata_file: Path
    indicators_file: Path
    configurations_file: Path
    var_conf_file: Path


def read_results_pointer(results_yaml: Path, evolver_home: Path) -> ResultsPointer:
    """Read results.yaml and resolve its paths against Evolver's working directory.

    The paths inside results.yaml (outputDirectory, metadataFile, ...) are the
    same relative strings the request gave for `outputDirectory`, resolved by
    the JVM against its own working directory (`evolver_home`) — not against
    where results.yaml itself lives.

    Args:
        results_yaml: Path to the results.yaml written by TrainingRunnerMain.
        evolver_home: Path to the Evolver checkout used as the JVM's cwd.

    Returns:
        The parsed pointer to the run's output directory and result files.
    """
    raw = yaml.safe_load(results_yaml.read_text())
    output_directory = evolver_home / raw["outputDirectory"]
    return ResultsPointer(
        output_directory=output_directory,
        metadata_file=evolver_home / raw["metadataFile"],
        indicators_file=evolver_home / raw["indicatorsFile"],
        configurations_file=evolver_home / raw["configurationsFile"],
        var_conf_file=output_directory / "VAR_CONF.txt",
    )


def load_indicators(indicators_csv: Path) -> pd.DataFrame:
    """Load the accumulated per-checkpoint non-dominated archives.

    INDICATORS.csv is appended to on every meta-optimizer generation, one
    block of rows per checkpoint (tagged by the `Evaluation` column) — it is
    never overwritten. Since this may be read while Evolver is mid-write, a
    trailing line not yet terminated by a newline is dropped before parsing.

    Args:
        indicators_csv: Path to INDICATORS.csv.

    Returns:
        One row per non-dominated configuration, per checkpoint written so far.
    """
    text = indicators_csv.read_text()
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines = lines[:-1]
    return pd.read_csv(io.StringIO("".join(lines)))


def latest_checkpoint_evaluation(history: pd.DataFrame) -> int | None:
    """Return the most recent checkpoint's evaluation count.

    Args:
        history: Rows loaded by `load_indicators`, possibly spanning several
            checkpoints.

    Returns:
        The maximum `Evaluation` value, or None if `history` is empty.
    """
    return int(history["Evaluation"].max()) if not history.empty else None


def checkpoint_front(history: pd.DataFrame, evaluation: int) -> pd.DataFrame:
    """Select one checkpoint's non-dominated archive.

    Args:
        history: Rows loaded by `load_indicators`, possibly spanning several
            checkpoints.
        evaluation: The checkpoint's evaluation count.

    Returns:
        Only the rows belonging to that checkpoint.
    """
    return history[history["Evaluation"] == evaluation]


def _front_values(
    history: pd.DataFrame, evaluation: int, indicator_columns: list[str]
) -> frozenset:
    """The indicator-value set for one checkpoint, order-independent for comparison."""
    front = history.loc[history["Evaluation"] == evaluation, indicator_columns]
    return frozenset(map(tuple, front.to_numpy()))


def deduplicate_consecutive_checkpoints(history: pd.DataFrame) -> pd.DataFrame:
    """Drop a checkpoint when its indicator values match the previous checkpoint's.

    Meant for a multi-checkpoint scatter: plotting the same front repeatedly
    at different evaluation counts adds visual clutter without new
    information, so a run that has converged (or stalled) collapses to a
    single entry instead of one per unchanged checkpoint.

    Args:
        history: Rows loaded by `load_indicators`, possibly spanning several
            checkpoints.

    Returns:
        The same rows, minus any checkpoint whose indicator-value set equals
        the immediately preceding (lower-Evaluation) checkpoint kept so far.
    """
    if history.empty:
        return history
    indicator_columns = [c for c in history.columns if c not in ("Evaluation", "SolutionId")]
    kept_evaluations = []
    previous_front = None
    for evaluation in sorted(history["Evaluation"].unique()):
        front = _front_values(history, evaluation, indicator_columns)
        if front != previous_front:
            kept_evaluations.append(evaluation)
            previous_front = front
    return history[history["Evaluation"].isin(kept_evaluations)]


def read_metadata(metadata_txt: Path) -> str:
    """Read METADATA.txt's free-text run summary.

    Args:
        metadata_txt: Path to METADATA.txt.

    Returns:
        The file's full text content.
    """
    return metadata_txt.read_text()


def list_output_dir(output_directory: Path) -> list[tuple[str, int]]:
    """List result files and their sizes for display.

    Args:
        output_directory: The run's output directory.

    Returns:
        (file name, size in bytes) pairs, sorted by name.
    """
    entries = [
        (path.name, path.stat().st_size) for path in output_directory.iterdir() if path.is_file()
    ]
    return sorted(entries)
