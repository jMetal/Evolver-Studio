"""Tests for the tests' own wait for a run to end."""

import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from run_waiting import TIME_FACTOR_VARIABLE, time_factor, wait_for_run

from evolver_studio.evolver_client import is_alive, write_pid_file


def _running(run_dir: Path, pid: int) -> None:
    run_dir.mkdir()
    status = {"status": "RUNNING", "evaluationsDone": 1, "maxEvaluations": 9, "updatedAt": "x"}
    (run_dir / "status.yaml").write_text(yaml.safe_dump(status))
    write_pid_file(run_dir / "pid.txt", pid)


class TestTimeFactor:
    def test_should_be_one_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(TIME_FACTOR_VARIABLE, raising=False)

        assert time_factor() == 1.0

    def test_should_be_read_from_the_environment_and_never_shorten(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(TIME_FACTOR_VARIABLE, "4")
        assert time_factor() == 4.0

        monkeypatch.setenv(TIME_FACTOR_VARIABLE, "0.1")
        assert time_factor() == 1.0


class TestWaitForRun:
    def test_should_stop_a_run_that_does_not_end_in_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Arrange
        monkeypatch.delenv(TIME_FACTOR_VARIABLE, raising=False)
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
        )
        run_dir = tmp_path / "run"
        _running(run_dir, process.pid)

        # Act
        with pytest.raises(pytest.fail.Exception, match="the training did not finish in 1 s"):
            wait_for_run(run_dir, 1, "the training", poll=0.05)

        # Assert
        process.wait(timeout=10)
        assert not is_alive(process.pid)

    def test_should_return_once_the_run_has_ended(self, tmp_path: Path):
        # Arrange
        run_dir = tmp_path / "run"
        _running(run_dir, 1)
        finished = {
            "status": "FINISHED",
            "evaluationsDone": 9,
            "maxEvaluations": 9,
            "updatedAt": "y",
        }
        (run_dir / "status.yaml").write_text(yaml.safe_dump(finished))

        # Act
        start = time.monotonic()
        wait_for_run(run_dir, 10)

        # Assert
        assert time.monotonic() - start < 1
