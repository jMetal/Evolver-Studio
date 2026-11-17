"""Tests that the app writes where EVOLVER_STUDIO_DATA says, and in its installation otherwise."""

import os
import subprocess
import sys
from pathlib import Path

from evolver_studio.evolver_client import DATA_DIRECTORY_VARIABLE, STUDIO_HOME

PRINT_DIRECTORIES = (
    "from evolver_studio import evolver_client as c;"
    " print(c.WORKING_DIRECTORY); print(c.JAR_DIRECTORY)"
)


def directories_with(environment: dict[str, str]) -> tuple[Path, Path]:
    """The working and the jar directories of an interpreter that starts with an environment."""
    output = subprocess.run(
        [sys.executable, "-c", PRINT_DIRECTORIES],
        env={
            **{k: v for k, v in os.environ.items() if k != DATA_DIRECTORY_VARIABLE},
            **environment,
        },
        cwd=STUDIO_HOME,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    return Path(output[0]), Path(output[1])


class TestDataDirectory:
    def test_should_use_the_installation_when_the_variable_is_not_set(self):
        # Act
        working, jar = directories_with({})

        # Assert
        assert working == STUDIO_HOME
        assert jar == STUDIO_HOME / "lib"

    def test_should_use_the_directory_the_variable_names(self, tmp_path: Path):
        # Act
        working, jar = directories_with({DATA_DIRECTORY_VARIABLE: str(tmp_path)})

        # Assert
        assert working == tmp_path
        assert jar == tmp_path / "lib"
