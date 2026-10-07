"""Tests for listing the finished training runs."""

from pathlib import Path

import yaml

from evolver_studio.training_runs import list_finished_trainings

VAR_CONF = "# Evaluation: 100\nEP=1 NHV=2 | --crossover SBX\nEP=3 NHV=4 | --crossover PCX\n"


def _training(
    working_directory: Path, run_id: str, status: str = "FINISHED", var_conf: str | None = VAR_CONF
) -> Path:
    run_dir = working_directory / "cli-runner-runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "status.yaml").write_text(
        yaml.safe_dump(
            {"status": status, "evaluationsDone": 1, "maxEvaluations": 1, "updatedAt": "x"}
        )
    )
    output = f"results/{run_id}"
    (run_dir / "request.yaml").write_text(yaml.safe_dump({"outputDirectory": output}))
    (run_dir / "base_level.yaml").write_text(
        yaml.safe_dump(
            {
                "algorithmName": "NSGA-II",
                "encoding": "Double",
                "trainingProblemNames": ["ZDT4", {"class": "DTLZ2", "args": [12, 2]}],
            }
        )
    )
    if var_conf is not None:
        (working_directory / output).mkdir(parents=True)
        (working_directory / output / "VAR_CONF.txt").write_text(var_conf)
    return run_dir


class TestListFinishedTrainings:
    def test_should_list_a_finished_run_with_its_configurations(self, tmp_path: Path):
        # Arrange
        _training(tmp_path, "20261007-120000")

        # Act
        (training,) = list_finished_trainings(tmp_path)

        # Assert
        assert (training.algorithm, training.encoding) == ("NSGA-II", "Double")
        assert training.problems == ("ZDT4", "DTLZ2")
        assert [c.configuration for c in training.configurations] == [
            "--crossover SBX",
            "--crossover PCX",
        ]
        assert training.label.startswith("20261007-120000 · NSGA-II (Double) · ZDT4, DTLZ2")

    def test_should_list_the_most_recent_first(self, tmp_path: Path):
        # Arrange
        _training(tmp_path, "20261007-110000")
        _training(tmp_path, "20261007-120000")

        # Act
        runs = list_finished_trainings(tmp_path)

        # Assert
        assert [run.run_id for run in runs] == ["20261007-120000", "20261007-110000"]

    def test_should_leave_out_what_has_not_finished_or_has_no_configurations(self, tmp_path: Path):
        # Arrange
        _training(tmp_path, "a-running", status="RUNNING")
        _training(tmp_path, "b-failed", status="FAILED")
        _training(tmp_path, "c-no-file", var_conf=None)
        _training(tmp_path, "d-empty", var_conf="# nothing\n")
        cancelled = _training(tmp_path, "e-cancelled")
        (cancelled / "CANCELLED").write_text("")

        # Act / Assert
        assert list_finished_trainings(tmp_path) == []

    def test_should_list_nothing_without_a_runs_directory(self, tmp_path: Path):
        # Act / Assert
        assert list_finished_trainings(tmp_path) == []
