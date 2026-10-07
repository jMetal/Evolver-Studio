"""Tests for reading the results of a solve run."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from evolver_studio.evolver_client import RunState
from evolver_studio.solve_results import (
    filter_by_objectives,
    indicator_summary,
    is_permutation,
    list_solve_runs,
    objective_columns,
    read_front,
    read_indicators,
    read_request,
    read_run_fronts,
    read_solutions,
    variable_columns,
    zip_fronts,
)

INDICATORS = "Run,Seed,TimeMs,EP,NHV\n1,1,53,0.8,0.9\n2,2,24,0.4,0.7\n"


@pytest.fixture
def output(tmp_path: Path) -> Path:
    """The output directory of a solve run with two runs of two-objective fronts."""
    directory = tmp_path / "output"
    for run, front in ((1, "0.0,1.0\n1.0,0.0\n"), (2, "0.5,0.5\n")):
        (directory / f"run-{run}").mkdir(parents=True)
        (directory / f"run-{run}" / "FUN.csv").write_text(front)
        (directory / f"run-{run}" / "VAR.csv").write_text("0.1,0.2\n")
    (directory / "INDICATORS.csv").write_text(INDICATORS)
    (directory / "METADATA.txt").write_text("# Solve run\n")
    return directory


class TestReadRunFronts:
    def test_should_read_each_runs_front_by_run_number(self, output: Path):
        # Act
        fronts = read_run_fronts(output)

        # Assert
        assert list(fronts) == [1, 2]
        assert list(fronts[1].columns) == ["f1", "f2"]
        assert len(fronts[1]) == 2
        assert len(fronts[2]) == 1

    def test_should_order_the_runs_numerically(self, output: Path):
        # Arrange
        (output / "run-10").mkdir()
        (output / "run-10" / "FUN.csv").write_text("0.1,0.9\n")

        # Act / Assert
        assert list(read_run_fronts(output)) == [1, 2, 10]

    def test_should_skip_a_run_that_has_not_written_its_front_yet(self, output: Path):
        # Arrange
        (output / "run-3").mkdir()

        # Act / Assert
        assert list(read_run_fronts(output)) == [1, 2]

    def test_should_read_a_front_with_any_number_of_objectives(self, tmp_path: Path):
        # Arrange
        file = tmp_path / "front.csv"
        file.write_text("1,2,3\n4,5,6\n")

        # Act / Assert
        assert list(read_front(file).columns) == ["f1", "f2", "f3"]


class TestReadSolutions:
    def test_should_join_the_objectives_and_the_variables_of_each_solution(self, output: Path):
        # Arrange: the fixture's VAR files have one solution row each, FUN files two and one
        (output / "run-1" / "VAR.csv").write_text("0.1,0.2,0.3\n0.4,0.5,0.6\n")

        # Act
        solutions = read_solutions(output, 1)

        # Assert
        assert list(solutions.columns) == ["f1", "f2", "x1", "x2", "x3"]
        assert objective_columns(solutions) == ["f1", "f2"]
        assert variable_columns(solutions) == ["x1", "x2", "x3"]
        assert solutions.loc[1, "x1"] == 0.4

    def test_should_give_only_the_objectives_when_the_run_has_no_variables_file(self, output: Path):
        # Arrange
        (output / "run-2" / "VAR.csv").unlink()

        # Act
        solutions = read_solutions(output, 2)

        # Assert
        assert list(solutions.columns) == ["f1", "f2"]


class TestFilterByObjectives:
    @staticmethod
    def _solutions() -> pd.DataFrame:
        return pd.DataFrame({"f1": [0.0, 0.5, 1.0], "f2": [1.0, 0.5, 0.0], "x1": [9, 8, 7]})

    def test_should_keep_the_solutions_inside_every_range_with_their_index(self):
        # Act
        kept = filter_by_objectives(self._solutions(), {"f1": (0.4, 1.0), "f2": (0.0, 0.5)})

        # Assert
        assert list(kept.index) == [1, 2]

    def test_should_keep_everything_without_ranges(self):
        # Act / Assert
        assert len(filter_by_objectives(self._solutions(), {})) == 3

    def test_should_include_the_bounds(self):
        # Act
        kept = filter_by_objectives(self._solutions(), {"f1": (0.5, 0.5)})

        # Assert
        assert list(kept.index) == [1]


class TestIsPermutation:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([2, 0, 1], True),
            ([2.0, 0.0, 1.0], True),
            ([1, 2, 3], False),
            ([0, 0, 1], False),
            ([0.5, 1.5, 2.5], False),
        ],
    )
    def test_should_tell_a_permutation_of_zero_to_n_minus_one(self, values: list, expected: bool):
        # Act / Assert
        assert is_permutation(pd.Series(values)) is expected


class TestReadRequest:
    def test_should_read_the_request_of_a_run(self, tmp_path: Path):
        # Arrange
        (tmp_path / "request.yaml").write_text("algorithmName: NSGA-II\nstatusFrequency: 500\n")

        # Act / Assert
        assert read_request(tmp_path) == {"algorithmName": "NSGA-II", "statusFrequency": 500}

    @pytest.mark.parametrize("content", [None, "[not, a, mapping]", "a: [unclosed"])
    def test_should_give_none_when_there_is_no_readable_request(
        self, tmp_path: Path, content: str | None
    ):
        # Arrange
        if content is not None:
            (tmp_path / "request.yaml").write_text(content)

        # Act / Assert
        assert read_request(tmp_path) is None


class TestIndicators:
    def test_should_read_one_row_per_run(self, output: Path):
        # Act
        indicators = read_indicators(output)

        # Assert
        assert list(indicators.columns) == ["Run", "Seed", "TimeMs", "EP", "NHV"]
        assert len(indicators) == 2

    def test_should_summarize_each_indicator_over_the_runs(self, output: Path):
        # Act
        summary = indicator_summary(read_indicators(output))

        # Assert
        assert list(summary.index) == ["TimeMs", "EP", "NHV"]
        assert summary.loc["EP", "mean"] == pytest.approx(0.6)
        assert summary.loc["EP", "min"] == pytest.approx(0.4)
        assert summary.loc["EP", "max"] == pytest.approx(0.8)

    def test_should_leave_the_deviation_empty_for_a_single_run(self):
        # Arrange
        indicators = pd.read_csv(io.StringIO("Run,Seed,TimeMs,EP\n1,1,5,0.3\n"))

        # Act
        summary = indicator_summary(indicators)

        # Assert
        assert pd.isna(summary.loc["EP", "std"])


class TestZipFronts:
    def test_should_pack_every_result_file_with_its_relative_path(self, output: Path):
        # Act
        content = zip_fronts(output)

        # Assert
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            assert set(archive.namelist()) == {
                "INDICATORS.csv",
                "METADATA.txt",
                "run-1/FUN.csv",
                "run-1/VAR.csv",
                "run-2/FUN.csv",
                "run-2/VAR.csv",
            }


class TestListSolveRuns:
    @staticmethod
    def _run(runs: Path, run_id: str, status: str | None) -> None:
        directory = runs / run_id
        directory.mkdir(parents=True)
        (directory / "request.yaml").write_text(
            f"algorithmName: NSGA-II\nproblem: ZDT1\noutputDirectory: solve-runs/{run_id}/output\n"
        )
        if status is not None:
            (directory / "status.yaml").write_text(
                f"status: {status}\nevaluationsDone: 10\nmaxEvaluations: 10\n"
                "updatedAt: '2026-01-01'\n"
            )

    def test_should_list_the_runs_most_recent_first_with_their_state(self, tmp_path: Path):
        # Arrange
        runs = tmp_path / "solve-runs"
        self._run(runs, "20260101-100000", "FINISHED")
        self._run(runs, "20260102-100000", "FAILED")
        self._run(runs, "20260103-100000", None)

        # Act
        listed = list_solve_runs(runs, tmp_path)

        # Assert
        assert [run.run_id for run in listed] == [
            "20260103-100000",
            "20260102-100000",
            "20260101-100000",
        ]
        assert [run.state for run in listed] == [None, RunState.FAILED, RunState.FINISHED]
        assert listed[0].output_directory == tmp_path / "solve-runs/20260103-100000/output"
        assert (listed[0].algorithm, listed[0].problem) == ("NSGA-II", "ZDT1")
        assert listed[0].request["algorithmName"] == "NSGA-II"

    def test_should_skip_a_directory_without_a_readable_request(self, tmp_path: Path):
        # Arrange
        runs = tmp_path / "solve-runs"
        self._run(runs, "20260101-100000", "FINISHED")
        (runs / "stray").mkdir()

        # Act
        listed = list_solve_runs(runs, tmp_path)

        # Assert
        assert [run.run_id for run in listed] == ["20260101-100000"]

    def test_should_list_nothing_when_there_is_no_runs_directory(self, tmp_path: Path):
        # Act / Assert
        assert list_solve_runs(tmp_path / "missing", tmp_path) == []
