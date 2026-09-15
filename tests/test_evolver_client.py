"""Tests for Evolver subprocess integration (status parsing, jar build)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evolver_studio.evolver_client import RunState, build_jar, read_status
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
