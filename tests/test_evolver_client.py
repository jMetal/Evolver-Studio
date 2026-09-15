"""Tests for Evolver subprocess integration (status parsing, jar build)."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evolver_studio.evolver_client import (
    RunState,
    build_jar,
    cancel_training,
    read_pid,
    read_status,
    write_pid_file,
)
from evolver_studio.result import Err, Ok


class TestReadStatus:
    def test_should_parse_running_status(self, tmp_path: Path):
        """A RUNNING status.yaml must parse into a RunStatus with no error message."""
        # Arrange
        status_yaml = tmp_path / "status.yaml"
        status_yaml.write_text(
            "status: RUNNING\nevaluationsDone: 100\nmaxEvaluations: 2000\n"
            "updatedAt: '2026-09-15T10:23:41.123456'\n"
        )

        # Act
        status = read_status(status_yaml)

        # Assert
        assert status.state == RunState.RUNNING
        assert status.evaluations_done == 100
        assert status.error_message is None

    def test_should_parse_failed_status_with_error_message(self, tmp_path: Path):
        """A FAILED status.yaml must expose its errorMessage."""
        # Arrange
        status_yaml = tmp_path / "status.yaml"
        status_yaml.write_text(
            "status: FAILED\nevaluationsDone: 0\nmaxEvaluations: 2000\n"
            "updatedAt: '2026-09-15T10:23:41.123456'\nerrorMessage: 'boom'\n"
        )

        # Act
        status = read_status(status_yaml)

        # Assert
        assert status.state == RunState.FAILED
        assert status.error_message == "boom"

    def test_should_return_none_when_status_file_missing(self, tmp_path: Path):
        """Polling before the file exists must not raise."""
        # Arrange
        status_yaml = tmp_path / "missing.yaml"

        # Act
        status = read_status(status_yaml)

        # Assert
        assert status is None

    def test_should_return_none_when_status_file_is_half_written(self, tmp_path: Path):
        """A partially flushed write, missing required keys, must not raise."""
        # Arrange
        status_yaml = tmp_path / "status.yaml"
        status_yaml.write_text("status: RUNNING\nevaluationsDone: 10\n")

        # Act
        status = read_status(status_yaml)

        # Assert
        assert status is None


class TestBuildJar:
    def test_should_return_ok_when_maven_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """A zero exit code must produce Ok(None)."""
        # Arrange
        completed = MagicMock(returncode=0, stdout="", stderr="")
        monkeypatch.setattr(
            "evolver_studio.evolver_client.subprocess.run", lambda *a, **k: completed
        )

        # Act
        result = build_jar(tmp_path)

        # Assert
        assert isinstance(result, Ok)

    def test_should_return_err_with_output_when_maven_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """A non-zero exit code must produce Err with the captured Maven output."""
        # Arrange
        completed = MagicMock(returncode=1, stdout="compile error\n", stderr="")
        monkeypatch.setattr(
            "evolver_studio.evolver_client.subprocess.run", lambda *a, **k: completed
        )

        # Act
        result = build_jar(tmp_path)

        # Assert
        assert isinstance(result, Err)
        assert "compile error" in result.message


class TestPidFile:
    def test_should_round_trip_a_pid(self, tmp_path: Path):
        """A written PID must be read back unchanged."""
        # Arrange
        pid_file = tmp_path / "pid.txt"

        # Act
        write_pid_file(pid_file, 12345)
        pid = read_pid(pid_file)

        # Assert
        assert pid == 12345

    def test_should_return_none_when_pid_file_missing(self, tmp_path: Path):
        """Checking before a process has been launched must not raise."""
        # Arrange
        pid_file = tmp_path / "missing.txt"

        # Act
        pid = read_pid(pid_file)

        # Assert
        assert pid is None

    def test_should_return_none_when_pid_file_is_not_an_integer(self, tmp_path: Path):
        """A corrupted PID file must not raise."""
        # Arrange
        pid_file = tmp_path / "pid.txt"
        pid_file.write_text("not-a-pid")

        # Act
        pid = read_pid(pid_file)

        # Assert
        assert pid is None


class TestCancelTraining:
    def test_should_stop_a_running_process(self, monkeypatch: pytest.MonkeyPatch):
        """SIGTERM must reach and terminate a genuinely running process."""
        # Arrange
        monkeypatch.setattr("evolver_studio.evolver_client.CANCEL_GRACE_PERIOD_SECONDS", 0.2)
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

        # Act
        result = cancel_training(process.pid)
        process.wait(timeout=5)

        # Assert
        assert isinstance(result, Ok)
        assert process.poll() is not None

    def test_should_return_ok_when_process_is_already_gone(self):
        """Cancelling an already-finished process must be a no-op success."""
        # Arrange
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait(timeout=5)

        # Act
        result = cancel_training(process.pid)

        # Assert
        assert isinstance(result, Ok)
