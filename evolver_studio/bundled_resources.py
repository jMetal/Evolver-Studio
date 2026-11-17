"""The Evolver resource files copied into this repository's resources/ directory.

Evolver's jar does not include the reference fronts, weight vectors and TSP instances its
training runs read (through paths relative to their working directory, such as
resources/referenceFronts/ZDT4.csv), so they are copied from the source of the Evolver release
the app runs by scripts/sync_resources.py, which also records their checksums in
resources/SHA256SUMS.
"""

import hashlib
import shutil
from pathlib import Path

RESOURCES_DIRECTORY = Path(__file__).resolve().parent.parent / "resources"
MANIFEST_FILE = RESOURCES_DIRECTORY / "SHA256SUMS"
# Directories under resources/ copied from Evolver (see resources/README.md).
COPIED_DIRECTORIES = ("referenceFronts", "referenceFrontsTSP", "tspInstances", "weightVectors")
# Files deliberately left out: MaF's many-objective fronts are ~73 MB of the ~99 MB.
EXCLUDED_FILE_PREFIXES = ("referenceFronts/MaF",)


def is_copied(relative_path: str) -> bool:
    """Tell whether a file of Evolver's resources/ directory belongs in the copy.

    Args:
        relative_path: A path relative to resources/, like "referenceFronts/ZDT4.csv".

    Returns:
        True if it lies under one of COPIED_DIRECTORIES and is not excluded.
    """
    in_copied_directory = relative_path.startswith(
        tuple(f"{directory}/" for directory in COPIED_DIRECTORIES)
    )
    return in_copied_directory and not relative_path.startswith(EXCLUDED_FILE_PREFIXES)


def copied_files(resources_directory: Path = RESOURCES_DIRECTORY) -> list[str]:
    """Return the files present under the copied directories, relative to resources/.

    Args:
        resources_directory: The resources/ directory to scan.

    Returns:
        Sorted paths like "referenceFronts/ZDT4.csv".
    """
    return sorted(
        path.relative_to(resources_directory).as_posix()
        for directory in COPIED_DIRECTORIES
        for path in (resources_directory / directory).rglob("*")
        if path.is_file()
    )


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(manifest_file: Path = MANIFEST_FILE) -> dict[str, str]:
    """Read a SHA256SUMS manifest.

    Args:
        manifest_file: The manifest, one "<sha256>  <relative path>" line per file.

    Returns:
        The SHA-256 digest of each listed file, by relative path.
    """
    return {
        name: digest
        for digest, name in (line.split("  ", 1) for line in manifest_file.read_text().splitlines())
    }


def write_manifest(
    resources_directory: Path = RESOURCES_DIRECTORY, manifest_file: Path = MANIFEST_FILE
) -> int:
    """Record the checksum of every copied file in a SHA256SUMS manifest.

    Args:
        resources_directory: The resources/ directory holding the copy.
        manifest_file: Where to write the manifest.

    Returns:
        The number of files listed.
    """
    files = copied_files(resources_directory)
    manifest_file.write_text(
        "".join(f"{sha256(resources_directory / name)}  {name}\n" for name in files)
    )
    return len(files)


def provision_resources(data_directory: Path, source: Path = RESOURCES_DIRECTORY) -> bool:
    """Make a data directory hold the resources/ the JVM reads, copying them if it does not.

    The JVM resolves resources/... against its working directory, which is the data directory
    when the app does not run from its own installation (see EVOLVER_STUDIO_DATA). The files
    are copied over, never removed: whatever the user added there stays. The manifest is copied
    last, so an interrupted copy is done again.

    Args:
        data_directory: The directory whose resources/ is to hold the files.
        source: The resources/ directory to copy from.

    Returns:
        True if the files were copied, False if the data directory already held them (or is
        the installation itself).
    """
    target = data_directory / "resources"
    if target.resolve() == source.resolve():
        return False
    manifest, copied_manifest = source / MANIFEST_FILE.name, target / MANIFEST_FILE.name
    if copied_manifest.is_file() and copied_manifest.read_bytes() == manifest.read_bytes():
        return False
    shutil.copytree(
        source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns(MANIFEST_FILE.name)
    )
    shutil.copy2(manifest, copied_manifest)
    return True
