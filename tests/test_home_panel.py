"""Tests for what the home page shows besides the links."""

from pathlib import Path

import pandas as pd
import pytest

from evolver_studio import home_panel
from evolver_studio.home_panel import SOLVE, activity, front_figure
from evolver_studio.training_runs import FinishedTraining


def write_solve_run(root: Path, run_id: str, state: str, objectives: list[list[float]]) -> Path:
    run_dir = root / "solve-runs" / run_id
    output = root / "out" / run_id
    (output / "run-1").mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (run_dir / "request.yaml").write_text(
        f"algorithmName: NSGA-II\nproblem: ZDT1\noutputDirectory: out/{run_id}\n"
    )
    (run_dir / "status.yaml").write_text(
        f"{{status: {state}, evaluationsDone: 10, maxEvaluations: 10, updatedAt: '2026-01-01'}}"
    )
    pd.DataFrame(objectives).to_csv(output / "run-1" / "FUN.csv", header=False, index=False)
    return run_dir


class TestActivity:
    def test_should_be_empty_without_runs(self, tmp_path: Path):
        # Act
        summary = activity(tmp_path)

        # Assert
        assert summary == home_panel.Activity(None, None, 0, 0)

    def test_should_count_the_stored_runs_of_each_use(self, tmp_path: Path):
        # Arrange
        write_solve_run(tmp_path, "20261001-100000", "FINISHED", [[0.0, 1.0]])
        write_solve_run(tmp_path, "20261001-110000", "FINISHED", [[0.0, 1.0]])
        (tmp_path / "cli-runner-runs" / "20261001-120000").mkdir(parents=True)

        # Act
        summary = activity(tmp_path)

        # Assert
        assert (summary.solve_runs, summary.trainings) == (2, 1)

    def test_should_offer_the_newest_finished_solve_run_with_its_front(self, tmp_path: Path):
        # Arrange
        write_solve_run(tmp_path, "20261001-100000", "FINISHED", [[0.0, 1.0]])
        write_solve_run(tmp_path, "20261002-100000", "FINISHED", [[0.2, 0.8], [0.5, 0.5]])

        # Act
        latest = activity(tmp_path).latest

        # Assert
        assert latest is not None
        assert (latest.kind, latest.run_id) == (SOLVE, "20261002-100000")
        assert latest.label == "NSGA-II on ZDT1"
        assert len(latest.front) == 2

    def test_should_skip_a_solve_run_that_failed(self, tmp_path: Path):
        # Arrange
        write_solve_run(tmp_path, "20261001-100000", "FINISHED", [[0.0, 1.0]])
        write_solve_run(tmp_path, "20261002-100000", "FAILED", [[0.2, 0.8]])

        # Act
        latest = activity(tmp_path).latest

        # Assert
        assert latest is not None
        assert latest.run_id == "20261001-100000"


class TestProblemsWithConstraints:
    def test_should_show_only_the_feasible_solutions(self, tmp_path: Path):
        # Arrange
        write_solve_run(tmp_path, "20261001-100000", "FINISHED", [[0.0, 1.0], [0.1, 0.2]])
        (tmp_path / "out/20261001-100000/run-1/CV.csv").write_text("0.0\n3.0\n")

        # Act
        latest = activity(tmp_path).latest

        # Assert
        assert latest is not None and len(latest.front) == 1

    def test_should_skip_a_run_without_feasible_solutions(self, tmp_path: Path):
        # Arrange
        write_solve_run(tmp_path, "20261001-100000", "FINISHED", [[0.0, 1.0]])
        write_solve_run(tmp_path, "20261002-100000", "FINISHED", [[0.0, 1.0]])
        (tmp_path / "out/20261002-100000/run-1/CV.csv").write_text("3.0\n")

        # Act
        latest = activity(tmp_path).latest

        # Assert
        assert latest is not None and latest.run_id == "20261001-100000"


class TestFrontFigure:
    @pytest.mark.parametrize("objectives", [2, 3])
    def test_should_plot_two_or_three_objectives(self, objectives: int):
        # Arrange
        front = pd.DataFrame({f"f{i}": [0.1, 0.9] for i in range(1, objectives + 1)})

        # Act
        figure = front_figure(front, height=300)

        # Assert
        assert figure is not None
        assert figure.layout.height == 300

    @pytest.mark.parametrize("objectives", [1, 4])
    def test_should_not_plot_any_other_number_of_objectives(self, objectives: int):
        # Arrange
        front = pd.DataFrame({f"f{i}": [0.1, 0.9] for i in range(1, objectives + 1)})

        # Act / Assert
        assert front_figure(front, height=300) is None


class TestRecentRuns:
    def test_should_list_the_newest_runs_with_their_labels_and_dates(self, tmp_path: Path):
        # Arrange
        write_solve_run(tmp_path, "20261001-100000", "FINISHED", [[0.0, 1.0]])
        write_solve_run(tmp_path, "20261002-153045", "FINISHED", [[0.0, 1.0]])

        # Act
        recent = activity(tmp_path).recent

        # Assert
        assert [run.run_id for run in recent] == ["20261002-153045", "20261001-100000"]
        assert recent[0].label == "NSGA-II on ZDT1"
        assert recent[0].launched is not None
        assert (recent[0].launched.month, recent[0].launched.minute) == (10, 30)

    def test_should_keep_only_the_three_newest(self, tmp_path: Path):
        # Arrange
        for day in range(1, 6):
            write_solve_run(tmp_path, f"2026100{day}-100000", "FINISHED", [[0.0, 1.0]])

        # Act
        recent = activity(tmp_path).recent

        # Assert
        assert [run.run_id for run in recent] == [
            "20261005-100000",
            "20261004-100000",
            "20261003-100000",
        ]


class TestTrainingLabel:
    def test_should_name_only_the_first_problems_of_a_training(self):
        # Arrange
        training = FinishedTraining(
            "20261001-100000", "PAES", "Double", tuple(f"RE3{i}" for i in range(1, 8)), ()
        )

        # Act
        label = home_panel._training_label(training)

        # Assert
        assert label == "PAES on RE31, RE32 and 5 more"
