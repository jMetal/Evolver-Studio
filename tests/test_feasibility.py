"""Tests for the feasibility of the configurations of a training."""

from pathlib import Path

from evolver_studio.feasibility import (
    failed_configurations,
    failures_by_problem,
    feasible_by_checkpoint,
    read_feasibility,
)

HEADER = "Evaluation,SolutionId,FailedRuns,Runs,FeasibleRunRate,MeanScaledViolation\n"
ROWS = "50,0,1,1,0.0,0.4\n50,1,0,1,1.0,0.0\n100,0,0,1,1.0,0.0\n100,1,0,1,1.0,0.0\n"


class TestReading:
    def test_should_be_none_for_a_training_without_constraints(self, tmp_path: Path):
        # Act / Assert
        assert read_feasibility(tmp_path) is None

    def test_should_leave_out_the_line_being_written(self, tmp_path: Path):
        # Arrange
        (tmp_path / "FEASIBILITY.csv").write_text(HEADER + ROWS + "150,0,1")

        # Act
        table = read_feasibility(tmp_path)

        # Assert
        assert table is not None and len(table) == 4

    def test_should_be_none_before_the_first_checkpoint(self, tmp_path: Path):
        # Arrange
        (tmp_path / "FEASIBILITY.csv").write_text(HEADER)

        # Act / Assert
        assert read_feasibility(tmp_path) is None


class TestSummaries:
    def test_should_count_the_configurations_without_failed_runs(self, tmp_path: Path):
        # Arrange
        (tmp_path / "POPULATION_FEASIBILITY.csv").write_text(HEADER + ROWS)
        table = read_feasibility(tmp_path, population=True)

        # Act
        counts = feasible_by_checkpoint(table)

        # Assert
        assert counts["without_failures"].tolist() == [1, 2]
        assert counts["configurations"].tolist() == [2, 2]
        assert failed_configurations(table, 50) == {0}
        assert failed_configurations(table, 100) == set()

    def test_should_tell_the_failures_by_problem_at_the_last_checkpoint(self, tmp_path: Path):
        # Arrange
        (tmp_path / "FEASIBILITY_BY_PROBLEM.csv").write_text(
            "Evaluation,SolutionId,ProblemId,Problem,FailedRuns,Runs,MeanScaledViolation\n"
            "50,0,0,MW1,3,3,0.5\n100,0,0,MW1,0,3,0.0\n100,0,1,DASCMOP5,2,3,0.1\n"
        )

        # Act
        table = failures_by_problem(read_feasibility(tmp_path, by_problem=True))

        # Assert
        assert table["problem"].tolist() == ["MW1", "DASCMOP5"]
        assert table["failing"].tolist() == [0, 1]
        assert table["failed_runs"].tolist() == [0, 2]
