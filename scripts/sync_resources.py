"""Refresh resources/ from the source archive of the Evolver release the app runs.

Replaces the copied reference fronts, weight vectors and TSP instances (see
evolver_studio/bundled_resources.py) with those of release EVOLVER_VERSION, downloaded from
GitHub, and rewrites resources/SHA256SUMS. Run it (`make sync-resources`) whenever
EVOLVER_VERSION changes.
"""

import io
import shutil
import tarfile
import urllib.request

from evolver_studio.bundled_resources import (
    COPIED_DIRECTORIES,
    MANIFEST_FILE,
    RESOURCES_DIRECTORY,
    is_copied,
    write_manifest,
)
from evolver_studio.evolver_client import EVOLVER_VERSION

SOURCE_ARCHIVE_URL = (
    f"https://codeload.github.com/jMetal/Evolver/tar.gz/refs/tags/v{EVOLVER_VERSION}"
)


def main() -> None:
    print(f"Downloading {SOURCE_ARCHIVE_URL}")
    with urllib.request.urlopen(SOURCE_ARCHIVE_URL) as response:
        archive_bytes = response.read()
    for directory in COPIED_DIRECTORIES:
        shutil.rmtree(RESOURCES_DIRECTORY / directory, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            # Member names look like "Evolver-2.1/resources/referenceFronts/ZDT4.csv".
            parts = member.name.split("/", 2)
            if not member.isfile() or len(parts) < 3 or parts[1] != "resources":
                continue
            if is_copied(parts[2]):
                target = RESOURCES_DIRECTORY / parts[2]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
                target.chmod(member.mode & 0o777)
    count = write_manifest()
    print(f"Copied {count} files from Evolver {EVOLVER_VERSION}; wrote {MANIFEST_FILE.name}")


if __name__ == "__main__":
    main()
