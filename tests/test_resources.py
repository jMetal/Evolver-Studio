"""Tests that the resource files copied from Evolver (resources/) match their manifest."""

from pathlib import Path

from evolver_studio.bundled_resources import (
    RESOURCES_DIRECTORY,
    copied_files,
    is_copied,
    read_manifest,
    sha256,
    write_manifest,
)


class TestIsCopied:
    def test_should_copy_reference_fronts_and_weight_vectors(self):
        # Act / Assert
        assert is_copied("referenceFronts/ZDT4.csv")
        assert is_copied("weightVectors/W3D_100.dat")

    def test_should_leave_out_maf_fronts_and_other_directories(self):
        # Act / Assert
        assert not is_copied("referenceFronts/MaF07.15D.csv")
        assert not is_copied("extremePointsFronts/ZDT1.csv")


class TestManifest:
    def test_should_round_trip_the_checksums_of_the_copied_files(self, tmp_path: Path):
        # Arrange
        (tmp_path / "referenceFronts").mkdir()
        (tmp_path / "referenceFronts" / "ZDT4.csv").write_text("0.0 1.0\n")
        manifest_file = tmp_path / "SHA256SUMS"

        # Act
        count = write_manifest(tmp_path, manifest_file)

        # Assert
        assert count == 1
        assert read_manifest(manifest_file) == {
            "referenceFronts/ZDT4.csv": sha256(tmp_path / "referenceFronts" / "ZDT4.csv")
        }


class TestCopiedResources:
    """The copy in this repository, against the manifest scripts/sync_resources.py wrote."""

    def test_should_hold_exactly_the_files_of_the_manifest(self):
        """A file lost or added by hand would go unnoticed otherwise."""
        # Act / Assert
        assert copied_files() == sorted(read_manifest())

    def test_should_keep_every_file_unchanged(self):
        """Edited copies would no longer be Evolver's; refresh them with `make sync-resources`."""
        # Act
        changed = [
            name
            for name, digest in read_manifest().items()
            if sha256(RESOURCES_DIRECTORY / name) != digest
        ]

        # Assert
        assert changed == []

    def test_should_copy_only_files_that_belong_in_the_copy(self):
        # Act / Assert
        assert all(is_copied(name) for name in copied_files())
