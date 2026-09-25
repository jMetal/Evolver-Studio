"""Tests for detecting an in-progress training run across app restarts."""

import subprocess
import sys
from pathlib import Path

from evolver_studio.evolver_client import write_pid_file
from evolver_studio.runs import find_active_run, mark_cancelled

STATUS_RUNNING = (
    "status: RUNNING\nevaluationsDone: 10\nmaxEvaluations: 100\nupdatedAt: '2026-01-01'\n"
)
STATUS_FINISHED = (
    "status: FINISHED\nevaluationsDone: 100\nmaxEvaluations: 100\nupdatedAt: '2026-01-01'\n"
)


def _write_run(
    working_directory: Path, run_id: str, status_text: str, output_directory: str
) -> Path:
    run_dir = working_directory / "cli-runner-runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "status.yaml").write_text(status_text)
    (run_dir / "request.yaml").write_text(
        f"baseLevel: base_level.yaml\nmetaSearch: meta_search.yaml\n"
        f"outputDirectory: {output_directory}\n"
    )
    return run_dir


class TestFindActiveRun:
    def test_should_return_none_when_runs_directory_does_not_exist(self, tmp_path: Path):
        """A fresh installation with no prior runs must not raise."""
        # Act
        active_run = find_active_run(tmp_path)

        # Assert
        assert active_run is None

    def test_should_return_none_when_no_run_is_running(self, tmp_path: Path):
        """A finished run must not be reported as active."""
        # Arrange
        _write_run(tmp_path, "20260101-000000", STATUS_FINISHED, "results/x/20260101-000000")

        # Act
        active_run = find_active_run(tmp_path)

        # Assert
        assert active_run is None

    def test_should_return_the_running_run(self, tmp_path: Path):
        """A running run must be reported, with paths resolved against the working directory."""
        # Arrange
        run_dir = _write_run(
            tmp_path, "20260101-000000", STATUS_RUNNING, "results/x/20260101-000000"
        )

        # Act
        active_run = find_active_run(tmp_path)

        # Assert
        assert active_run.run_id == "20260101-000000"
        assert active_run.status_yaml == run_dir / "status.yaml"
        assert active_run.pid_file == run_dir / "pid.txt"
        assert active_run.indicators_csv == tmp_path / "results/x/20260101-000000/INDICATORS.csv"

    def test_should_skip_cancelled_runs(self, tmp_path: Path):
        """A run marked cancelled must not be reported as active, even if status.yaml lags."""
        # Arrange
        run_dir = _write_run(
            tmp_path, "20260101-000000", STATUS_RUNNING, "results/x/20260101-000000"
        )
        mark_cancelled(run_dir)

        # Act
        active_run = find_active_run(tmp_path)

        # Assert
        assert active_run is None

    def test_should_ignore_a_running_status_whose_pid_is_actually_dead(self, tmp_path: Path):
        """A crashed process that never got to write FAILED must not be reported active."""
        # Arrange
        run_dir = _write_run(
            tmp_path, "20260101-000000", STATUS_RUNNING, "results/x/20260101-000000"
        )
        dead_process = subprocess.Popen([sys.executable, "-c", "pass"])
        dead_process.wait(timeout=5)
        write_pid_file(run_dir / "pid.txt", dead_process.pid)

        # Act
        active_run = find_active_run(tmp_path)

        # Assert
        assert active_run is None

    def test_should_return_the_most_recent_running_run(self, tmp_path: Path):
        """With several run directories, the most recent (by run_id) running one wins."""
        # Arrange
        _write_run(tmp_path, "20260101-000000", STATUS_FINISHED, "results/x/20260101-000000")
        _write_run(tmp_path, "20260101-010000", STATUS_RUNNING, "results/x/20260101-010000")

        # Act
        active_run = find_active_run(tmp_path)

        # Assert
        assert active_run.run_id == "20260101-010000"
