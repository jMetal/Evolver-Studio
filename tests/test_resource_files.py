"""Tests for reading Evolver's resource files from its jar."""

import zipfile
from pathlib import Path

import pytest

from evolver_studio.resource_files import (
    POM_PROPERTIES_ENTRY,
    jar_evolver_version,
    meta_optimizer_configuration_text,
    parameter_space_text,
)


@pytest.fixture
def jar(tmp_path: Path) -> Path:
    """A minimal stand-in for Evolver's jar, with its resources at the root."""
    path = tmp_path / "Evolver.jar"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("parameterSpaces/NSGAIIDouble.yaml", "algorithmResult: {}\n")
        archive.writestr("metaOptimizerConfigurations/MetaNSGAII.yaml", "algorithm: NSGA-II\n")
        archive.writestr(
            POM_PROPERTIES_ENTRY, "artifactId=Evolver\ngroupId=org.uma.jmetal\nversion=2.1\n"
        )
    return path


class TestResourceFiles:
    def test_should_read_a_parameter_space_from_the_jar(self, jar: Path):
        # Act
        text = parameter_space_text(jar, "NSGAIIDouble.yaml")

        # Assert
        assert text == "algorithmResult: {}\n"

    def test_should_read_a_meta_optimizer_configuration_from_the_jar(self, jar: Path):
        # Act
        text = meta_optimizer_configuration_text(jar, "MetaNSGAII.yaml")

        # Assert
        assert text == "algorithm: NSGA-II\n"

    def test_should_raise_key_error_for_a_file_the_jar_does_not_have(self, jar: Path):
        # Act / Assert
        with pytest.raises(KeyError):
            parameter_space_text(jar, "Missing.yaml")

    def test_should_read_the_evolver_version_of_the_jar(self, jar: Path):
        # Act
        version = jar_evolver_version(jar)

        # Assert
        assert version == "2.1"

    def test_should_have_no_version_for_a_jar_that_does_not_record_one(self, tmp_path: Path):
        # Arrange
        path = tmp_path / "NoVersion.jar"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("parameterSpaces/NSGAIIDouble.yaml", "algorithmResult: {}\n")

        # Act
        version = jar_evolver_version(path)

        # Assert
        assert version is None
