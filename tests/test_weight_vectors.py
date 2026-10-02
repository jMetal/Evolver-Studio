"""Tests for finding the population sizes the weight vector files allow."""

from pathlib import Path

import pytest

from evolver_studio.weight_vectors import available_population_sizes


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    """A weight vectors directory with a few of Evolver's files, and others that are not."""
    for name in (
        "W2D_100.dat",
        "W2D_300.dat",
        "W2D_1000.dat",
        "W3D_91.dat",
        "notes.txt",
        "W2D.dat",
    ):
        (tmp_path / name).write_text("0.5 0.5\n")
    return tmp_path


class TestAvailablePopulationSizes:
    def test_should_list_the_sizes_for_the_objectives_in_ascending_order(self, directory: Path):
        # Act / Assert
        assert available_population_sizes(directory, 2) == [100, 300, 1000]

    def test_should_not_mix_objectives(self, directory: Path):
        # Act / Assert
        assert available_population_sizes(directory, 3) == [91]

    def test_should_find_nothing_for_objectives_without_a_file(self, directory: Path):
        # Act / Assert
        assert available_population_sizes(directory, 5) == []

    def test_should_find_nothing_in_a_missing_directory(self, tmp_path: Path):
        # Act / Assert
        assert available_population_sizes(tmp_path / "missing", 2) == []

    def test_should_find_the_real_files_copied_from_evolver(self):
        # Act
        sizes = available_population_sizes(
            Path(__file__).parent.parent / "resources/weightVectors", 3
        )

        # Assert
        assert 91 in sizes
        assert sizes == sorted(sizes)
