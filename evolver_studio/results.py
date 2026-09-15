"""Parsing of Evolver's training results (results.yaml, METADATA.txt, CSVs)."""

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
    """Load the meta-level non-dominated archive's indicator values.

    Args:
        indicators_csv: Path to INDICATORS.csv.

    Returns:
        One row per non-dominated configuration, one column per indicator.
    """
    return pd.read_csv(indicators_csv)


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
