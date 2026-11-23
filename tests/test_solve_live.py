"""Tests for the front of a solve run while it runs."""

from pathlib import Path

from evolver_studio.solve_live import CURRENT_FRONT_FILE, read_current_front


def write(directory: Path, text: str) -> None:
    (directory / CURRENT_FRONT_FILE).write_text(text)


class TestReadCurrentFront:
    def test_should_read_the_objectives_and_say_which_run_and_evaluation(self, tmp_path: Path):
        # Arrange
        write(tmp_path, "Run,Evaluations,NonDominated,F1,F2\n2,3000,1,0.1,0.9\n2,3000,1,0.5,0.4\n")

        # Act
        front = read_current_front(tmp_path)

        # Assert
        assert front is not None
        assert (front.run, front.evaluations) == (2, 3000)
        assert list(front.objectives.columns) == ["f1", "f2"]
        assert front.objectives["f2"].tolist() == [0.9, 0.4]

    def test_should_leave_out_the_violation_column_of_a_problem_with_constraints(
        self, tmp_path: Path
    ):
        # Arrange
        write(tmp_path, "Run,Evaluations,NonDominated,F1,F2,F3,Violation\n1,100,1,1,2,3,0\n")

        # Act
        front = read_current_front(tmp_path)

        # Assert
        assert front is not None
        assert list(front.objectives.columns) == ["f1", "f2", "f3"]

    def test_should_be_none_without_a_file(self, tmp_path: Path):
        # Act / Assert
        assert read_current_front(tmp_path) is None

    def test_should_be_none_for_an_empty_file_or_one_with_only_the_header(self, tmp_path: Path):
        # Arrange
        write(tmp_path, "")
        empty = read_current_front(tmp_path)
        write(tmp_path, "Run,Evaluations,NonDominated,F1,F2\n")

        # Act / Assert
        assert empty is None
        assert read_current_front(tmp_path) is None
