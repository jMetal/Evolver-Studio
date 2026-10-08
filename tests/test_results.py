"""Tests for results parsing (results.yaml, METADATA.txt, CSVs, folder listing)."""

from pathlib import Path

import pandas as pd

from evolver_studio.results import (
    checkpoint_front,
    deduplicate_consecutive_checkpoints,
    last_n_checkpoints,
    latest_checkpoint_evaluation,
    list_output_dir,
    load_indicators,
    read_metadata,
    read_results_pointer,
)


class TestReadResultsPointer:
    def test_should_resolve_paths_against_the_working_directory(self, tmp_path: Path):
        """outputDirectory/*.File paths are relative to the JVM's working directory."""
        # Arrange
        working_directory = tmp_path / "Evolver-Studio"
        run_dir = working_directory / "cli-runner-runs" / "20260915-000000"
        run_dir.mkdir(parents=True)
        results_yaml = run_dir / "results.yaml"
        results_yaml.write_text(
            "outputDirectory: results/nsgaii/ZDT4\n"
            "metadataFile: results/nsgaii/ZDT4/METADATA.txt\n"
            "indicatorsFile: results/nsgaii/ZDT4/INDICATORS.csv\n"
            "configurationsFile: results/nsgaii/ZDT4/CONFIGURATIONS.csv\n"
        )

        # Act
        pointer = read_results_pointer(results_yaml, working_directory)

        # Assert
        assert pointer.output_directory == working_directory / "results/nsgaii/ZDT4"
        assert pointer.var_conf_file == working_directory / "results/nsgaii/ZDT4" / "VAR_CONF.txt"


class TestReadResultsPointerPopulation:
    RESULTS = (
        "outputDirectory: results/x\nmetadataFile: results/x/METADATA.txt\n"
        "indicatorsFile: results/x/INDICATORS.csv\n"
        "configurationsFile: results/x/CONFIGURATIONS.csv\n"
    )

    def test_should_point_at_the_population_files_when_the_run_wrote_them(self, tmp_path: Path):
        # Arrange
        results_yaml = tmp_path / "results.yaml"
        results_yaml.write_text(
            self.RESULTS
            + "populationIndicatorsFile: results/x/POPULATION_INDICATORS.csv\n"
            + "populationConfigurationsFile: results/x/POPULATION_CONFIGURATIONS.csv\n"
        )

        # Act
        pointer = read_results_pointer(results_yaml, tmp_path)

        # Assert
        assert (
            pointer.population_indicators_file == tmp_path / "results/x/POPULATION_INDICATORS.csv"
        )
        assert (
            pointer.population_configurations_file
            == tmp_path / "results/x/POPULATION_CONFIGURATIONS.csv"
        )

    def test_should_have_no_population_files_otherwise(self, tmp_path: Path):
        # Arrange
        results_yaml = tmp_path / "results.yaml"
        results_yaml.write_text(self.RESULTS)

        # Act
        pointer = read_results_pointer(results_yaml, tmp_path)

        # Assert
        assert pointer.population_indicators_file is None
        assert pointer.population_configurations_file is None


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

    def test_should_ignore_incomplete_trailing_line(self, tmp_path: Path):
        """A checkpoint write caught mid-flush must not break parsing of prior rows."""
        # Arrange
        indicators_csv = tmp_path / "INDICATORS.csv"
        indicators_csv.write_text(
            "Evaluation,SolutionId,Epsilon,NormalizedHypervolume\n"
            "100,0,0.5,0.8\n"
            "200,0,0.3,0.9\n"
            "300,0,0.1"  # cut off mid-write, no trailing newline
        )

        # Act
        indicators = load_indicators(indicators_csv)

        # Assert
        assert list(indicators["Evaluation"]) == [100, 200]

    def test_should_load_multiple_checkpoints(self, tmp_path: Path):
        """Rows from every checkpoint written so far must all be present."""
        # Arrange
        indicators_csv = tmp_path / "INDICATORS.csv"
        indicators_csv.write_text(
            "Evaluation,SolutionId,Epsilon,NormalizedHypervolume\n100,0,0.5,0.8\n200,0,0.3,0.9\n"
        )

        # Act
        indicators = load_indicators(indicators_csv)

        # Assert
        assert list(indicators["Evaluation"]) == [100, 200]


class TestLatestCheckpointEvaluation:
    def test_should_return_max_evaluation(self):
        """The most recent checkpoint is the one with the highest Evaluation."""
        # Arrange
        history = pd.DataFrame({"Evaluation": [100, 300, 200]})

        # Act
        latest = latest_checkpoint_evaluation(history)

        # Assert
        assert latest == 300

    def test_should_return_none_for_empty_history(self):
        """No checkpoints written yet must not raise."""
        # Arrange
        history = pd.DataFrame({"Evaluation": []})

        # Act
        latest = latest_checkpoint_evaluation(history)

        # Assert
        assert latest is None


class TestCheckpointFront:
    def test_should_filter_rows_by_evaluation(self):
        """Only the requested checkpoint's rows are returned, others are excluded."""
        # Arrange
        history = pd.DataFrame(
            {"Evaluation": [100, 100, 200], "SolutionId": [0, 1, 0], "Epsilon": [0.5, 0.4, 0.1]}
        )

        # Act
        front = checkpoint_front(history, 100)

        # Assert
        assert list(front["SolutionId"]) == [0, 1]


class TestDeduplicateConsecutiveCheckpoints:
    def test_should_keep_all_checkpoints_when_fronts_differ(self):
        """Distinct fronts must all survive, in every case an improvement."""
        # Arrange
        history = pd.DataFrame(
            {
                "Evaluation": [100, 200],
                "SolutionId": [0, 0],
                "Epsilon": [0.5, 0.3],
                "NHV": [0.8, 0.9],
            }
        )

        # Act
        deduplicated = deduplicate_consecutive_checkpoints(history)

        # Assert
        assert sorted(deduplicated["Evaluation"].unique()) == [100, 200]

    def test_should_drop_a_checkpoint_identical_to_the_previous_one(self):
        """A stalled run's repeated identical front must collapse to one entry."""
        # Arrange
        history = pd.DataFrame(
            {
                "Evaluation": [100, 200],
                "SolutionId": [0, 0],
                "Epsilon": [0.5, 0.5],
                "NHV": [0.8, 0.8],
            }
        )

        # Act
        deduplicated = deduplicate_consecutive_checkpoints(history)

        # Assert
        assert list(deduplicated["Evaluation"].unique()) == [100]

    def test_should_keep_a_checkpoint_that_changes_after_a_repeat(self):
        """A->A->B must drop the second A but keep the later, different B."""
        # Arrange
        history = pd.DataFrame(
            {
                "Evaluation": [100, 200, 300],
                "SolutionId": [0, 0, 0],
                "Epsilon": [0.5, 0.5, 0.2],
                "NHV": [0.8, 0.8, 0.95],
            }
        )

        # Act
        deduplicated = deduplicate_consecutive_checkpoints(history)

        # Assert
        assert sorted(deduplicated["Evaluation"].unique()) == [100, 300]

    def test_should_return_empty_history_unchanged(self):
        """An empty history (no checkpoints written yet) must not raise."""
        # Arrange
        history = pd.DataFrame({"Evaluation": [], "SolutionId": [], "Epsilon": []})

        # Act
        deduplicated = deduplicate_consecutive_checkpoints(history)

        # Assert
        assert deduplicated.empty


class TestLastNCheckpoints:
    def test_should_keep_only_the_n_most_recent_checkpoints(self):
        """Early, large-scale checkpoints must be dropped when N is smaller than the total."""
        # Arrange
        history = pd.DataFrame(
            {
                "Evaluation": [100, 100, 200, 300],
                "SolutionId": [0, 1, 0, 0],
                "Epsilon": [9, 8, 2, 1],
            }
        )

        # Act
        limited = last_n_checkpoints(history, n=2)

        # Assert
        assert sorted(limited["Evaluation"].unique()) == [200, 300]

    def test_should_keep_everything_when_n_covers_all_checkpoints(self):
        """N at least as large as the number of checkpoints must be a no-op."""
        # Arrange
        history = pd.DataFrame({"Evaluation": [100, 200], "SolutionId": [0, 0], "Epsilon": [9, 1]})

        # Act
        limited = last_n_checkpoints(history, n=10)

        # Assert
        assert sorted(limited["Evaluation"].unique()) == [100, 200]

    def test_should_return_empty_history_unchanged(self):
        """An empty history must not raise."""
        # Arrange
        history = pd.DataFrame({"Evaluation": [], "SolutionId": [], "Epsilon": []})

        # Act
        limited = last_n_checkpoints(history, n=5)

        # Assert
        assert limited.empty


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
