"""Tests that the resource files copied from Evolver (resources/) stay in sync with it."""

from pathlib import Path

import pytest

EVOLVER_HOME = Path("/Users/ajnebro/Softw/Evolver")
STUDIO_HOME = Path(__file__).parent.parent

# Directories under resources/ copied from Evolver (see resources/README.md).
COPIED_DIRECTORIES = ("referenceFronts", "referenceFrontsTSP", "tspInstances", "weightVectors")
# Reference fronts deliberately left out: MaF's many-objective fronts are ~73 MB of the ~99 MB.
EXCLUDED_FILE_PREFIXES = ("referenceFronts/MaF",)


def _relative_files(resources_dir: Path) -> set[str]:
    """Return the copied files under a resources/ directory, relative to it.

    Args:
        resources_dir: A resources/ directory, either Evolver's or this repo's.

    Returns:
        Paths like "referenceFronts/ZDT4.csv", excluding EXCLUDED_FILE_PREFIXES.
    """
    return {
        path.relative_to(resources_dir).as_posix()
        for directory in COPIED_DIRECTORIES
        for path in (resources_dir / directory).rglob("*")
        if path.is_file()
        and not path.relative_to(resources_dir).as_posix().startswith(EXCLUDED_FILE_PREFIXES)
    }


class TestResourcesMatchEvolverCheckout:
    @pytest.fixture(autouse=True)
    def _require_checkout(self):
        if not (EVOLVER_HOME / "resources").is_dir():
            pytest.skip(f"Evolver checkout not found at {EVOLVER_HOME}")

    def test_should_copy_every_resource_file_of_evolver(self):
        """A reference front or weight vector file added to Evolver must not go unnoticed."""
        # Act
        missing = sorted(
            _relative_files(EVOLVER_HOME / "resources") - _relative_files(STUDIO_HOME / "resources")
        )

        # Assert
        assert missing == [], (
            f"File(s) in Evolver's resources/ not copied here: {missing}. Copy them, or add them "
            "to EXCLUDED_FILE_PREFIXES (with a reason)."
        )

    def test_should_not_keep_files_evolver_no_longer_has(self):
        """A file removed from Evolver should be removed from the copy too."""
        # Act
        extra = sorted(
            _relative_files(STUDIO_HOME / "resources") - _relative_files(EVOLVER_HOME / "resources")
        )

        # Assert
        assert extra == []

    def test_should_keep_copies_identical_to_evolver(self):
        """A reference front corrected in Evolver would otherwise stay stale here."""
        # Arrange
        studio_resources = STUDIO_HOME / "resources"
        evolver_resources = EVOLVER_HOME / "resources"
        common = _relative_files(studio_resources) & _relative_files(evolver_resources)

        # Act
        different = sorted(
            name
            for name in common
            if (studio_resources / name).read_bytes() != (evolver_resources / name).read_bytes()
        )

        # Assert
        assert different == []
