"""Tests that the resource files copied from Evolver (resources/) match their manifest."""

from pathlib import Path

from evolver_studio.bundled_resources import (
    RESOURCES_DIRECTORY,
    copied_files,
    is_copied,
    provision_resources,
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


class TestProvisionResources:
    @staticmethod
    def make_source(directory: Path, front: str = "0.0 1.0\n") -> Path:
        source = directory / "source"
        (source / "referenceFronts").mkdir(parents=True, exist_ok=True)
        (source / "referenceFronts" / "ZDT4.csv").write_text(front)
        write_manifest(source, source / "SHA256SUMS")
        return source

    def test_should_copy_the_resources_into_an_empty_data_directory(self, tmp_path: Path):
        # Arrange
        source = self.make_source(tmp_path)
        data = tmp_path / "data"

        # Act
        copied = provision_resources(data, source)

        # Assert
        assert copied
        assert (data / "resources" / "referenceFronts" / "ZDT4.csv").read_text() == "0.0 1.0\n"
        assert (data / "resources" / "SHA256SUMS").read_bytes() == (
            source / "SHA256SUMS"
        ).read_bytes()

    def test_should_not_copy_again_when_the_manifest_is_the_same(self, tmp_path: Path):
        # Arrange
        source = self.make_source(tmp_path)
        data = tmp_path / "data"
        provision_resources(data, source)
        (data / "resources" / "referenceFronts" / "ZDT4.csv").write_text("edited\n")

        # Act
        copied = provision_resources(data, source)

        # Assert
        assert not copied
        assert (data / "resources" / "referenceFronts" / "ZDT4.csv").read_text() == "edited\n"

    def test_should_copy_again_keeping_the_files_the_user_added(self, tmp_path: Path):
        # Arrange
        source = self.make_source(tmp_path)
        data = tmp_path / "data"
        provision_resources(data, source)
        own_front = data / "resources" / "referenceFronts" / "Own.csv"
        own_front.write_text("2.0 3.0\n")
        self.make_source(tmp_path, front="0.5 0.5\n")

        # Act
        copied = provision_resources(data, source)

        # Assert
        assert copied
        assert (data / "resources" / "referenceFronts" / "ZDT4.csv").read_text() == "0.5 0.5\n"
        assert own_front.read_text() == "2.0 3.0\n"

    def test_should_copy_nothing_when_the_data_directory_is_the_installation(self, tmp_path: Path):
        # Arrange
        installation = tmp_path / "installation"
        source = self.make_source(installation).rename(installation / "resources")

        # Act
        copied = provision_resources(installation, source)

        # Assert
        assert not copied
        assert sorted(path.name for path in installation.iterdir()) == ["resources"]
