"""Tests for Evolver subprocess integration (status parsing, jar download)."""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from evolver_studio.evolver_client import (
    EVOLVER_VERSION,
    JAR_DIRECTORY,
    JAR_OVERRIDE_VARIABLE,
    RunState,
    cancel_training,
    download_jar,
    is_jar_overridden,
    jar_path,
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


class TestJarPath:
    def test_should_default_to_the_downloaded_release_jar(self, monkeypatch: pytest.MonkeyPatch):
        # Arrange
        monkeypatch.delenv(JAR_OVERRIDE_VARIABLE, raising=False)

        # Act
        jar = jar_path()

        # Assert
        assert jar == JAR_DIRECTORY / f"Evolver-{EVOLVER_VERSION}-jar-with-dependencies.jar"
        assert not is_jar_overridden()

    def test_should_use_the_jar_named_by_the_override_variable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """EVOLVER_JAR lets a developer run a jar built from a local Evolver checkout."""
        # Arrange
        local_jar = tmp_path / "Evolver-2.2-SNAPSHOT-jar-with-dependencies.jar"
        monkeypatch.setenv(JAR_OVERRIDE_VARIABLE, str(local_jar))

        # Act
        jar = jar_path()

        # Assert
        assert jar == local_jar
        assert is_jar_overridden()


class TestDownloadJar:
    """Downloads from file:// URLs, which urllib serves like Maven Central's https:// ones."""

    def _publish(self, directory: Path, content: bytes, sha1: str | None = None) -> str:
        jar = directory / "Evolver.jar"
        jar.write_bytes(content)
        checksum = sha1 if sha1 is not None else hashlib.sha1(content).hexdigest()
        (directory / "Evolver.jar.sha1").write_text(checksum)
        return jar.as_uri()

    def test_should_store_the_jar_when_its_checksum_matches(self, tmp_path: Path):
        # Arrange
        url = self._publish(tmp_path, b"jar content")
        destination = tmp_path / "lib" / "downloaded.jar"
        fractions = []

        # Act
        result = download_jar(destination, fractions.append, url)

        # Assert
        assert isinstance(result, Ok)
        assert destination.read_bytes() == b"jar content"
        assert fractions[-1] == 1.0

    def test_should_discard_the_jar_when_its_checksum_does_not_match(self, tmp_path: Path):
        """A corrupted download must not be left where the app would run it."""
        # Arrange
        url = self._publish(tmp_path, b"jar content", sha1="0" * 40)
        destination = tmp_path / "lib" / "downloaded.jar"

        # Act
        result = download_jar(destination, url=url)

        # Assert
        assert isinstance(result, Err)
        assert "Checksum mismatch" in result.message
        assert list(destination.parent.iterdir()) == []

    def test_should_return_err_when_the_jar_cannot_be_fetched(self, tmp_path: Path):
        # Arrange
        destination = tmp_path / "lib" / "downloaded.jar"

        # Act
        result = download_jar(destination, url=(tmp_path / "missing.jar").as_uri())

        # Assert
        assert isinstance(result, Err)
        assert not destination.exists()


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
