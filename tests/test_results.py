"""Tests for results parsing (results.yaml, METADATA.txt, CSVs, folder listing)."""

from pathlib import Path

from evolver_studio.results import (
    list_output_dir,
    load_indicators,
    read_metadata,
    read_results_pointer,
)


class TestReadResultsPointer:
    def test_should_resolve_paths_against_evolver_home(self, tmp_path: Path):
        """outputDirectory/*.File paths are relative to evolver_home, not results.yaml's folder."""
        # Arrange
        evolver_home = tmp_path / "Evolver"
        run_dir = evolver_home / "cli-runner-runs" / "20260915-000000"
        run_dir.mkdir(parents=True)
        results_yaml = run_dir / "results.yaml"
        results_yaml.write_text(
            "outputDirectory: results/nsgaii/ZDT4\n"
            "metadataFile: results/nsgaii/ZDT4/METADATA.txt\n"
            "indicatorsFile: results/nsgaii/ZDT4/INDICATORS.csv\n"
            "configurationsFile: results/nsgaii/ZDT4/CONFIGURATIONS.csv\n"
        )

        # Act
        pointer = read_results_pointer(results_yaml, evolver_home)

        # Assert
        assert pointer.output_directory == evolver_home / "results/nsgaii/ZDT4"
        assert pointer.var_conf_file == evolver_home / "results/nsgaii/ZDT4" / "VAR_CONF.txt"


class TestLoadIndicators:
    def test_should_load_indicator_columns(self, tmp_path: Path):
        """INDICATORS.csv rows/columns must load into a DataFrame with the same header."""
        # Arrange
        indicators_csv = tmp_path / "INDICATORS.csv"
        indicators_csv.write_text(
            "Evaluation,SolutionId,Epsilon,NormalizedHypervolume\n2000,0,0.1,0.9\n"
        )

        # Act
        indicators = load_indicators(indicators_csv)

        # Assert
        assert list(indicators.columns) == [
            "Evaluation",
            "SolutionId",
            "Epsilon",
            "NormalizedHypervolume",
        ]
        assert indicators.iloc[0]["Epsilon"] == 0.1


class TestReadMetadata:
    def test_should_return_full_file_content(self, tmp_path: Path):
        """METADATA.txt must be returned verbatim."""
        # Arrange
        metadata_txt = tmp_path / "METADATA.txt"
        metadata_txt.write_text("=== Meta-Optimization Experiment ===\n")

        # Act
        content = read_metadata(metadata_txt)

        # Assert
        assert content == "=== Meta-Optimization Experiment ===\n"


class TestListOutputDir:
    def test_should_list_files_with_sizes_sorted_by_name(self, tmp_path: Path):
        """Only files (not subdirectories) are listed, sorted alphabetically."""
        # Arrange
        (tmp_path / "METADATA.txt").write_text("abc")
        (tmp_path / "INDICATORS.csv").write_text("de")
        (tmp_path / "subdir").mkdir()

        # Act
        entries = list_output_dir(tmp_path)

        # Assert
        assert entries == [("INDICATORS.csv", 2), ("METADATA.txt", 3)]
