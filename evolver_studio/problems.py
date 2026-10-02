"""Finding the reference front files of a problem among the resources copied from Evolver.

Evolver's reference fronts are named after their problem, with the number of objectives as a suffix
when the problem has fronts for several (ZDT1.csv, DTLZ2.3D.csv); the TSP instances' fronts live in
their own directory. Paths are returned relative to the JVM's working directory, as a request's
`referenceFrontFileName` expects them.
"""

import re
from pathlib import Path

from evolver_studio.bundled_resources import RESOURCES_DIRECTORY

REFERENCE_FRONT_DIRECTORIES = ("referenceFronts", "referenceFrontsTSP")


def reference_front_candidates(
    problem: str, resources_directory: Path = RESOURCES_DIRECTORY
) -> list[str]:
    """List the reference front files that belong to a problem.

    Args:
        problem: A problem name (ZDT1) or a fully-qualified class name, whose last segment (the
            class's simple name) names the fronts.
        resources_directory: The resources/ directory to look in.

    Returns:
        Their paths relative to the directory that holds resources/, such as
        "resources/referenceFronts/DTLZ2.3D.csv": the file without a dimension first, then the
        others by number of objectives.
    """
    simple_name = problem.rsplit(".", 1)[-1]
    pattern = re.compile(rf"{re.escape(simple_name)}(?:\.(\d+)D)?\.csv")
    found: list[tuple[int, str]] = []
    for directory in REFERENCE_FRONT_DIRECTORIES:
        folder = resources_directory / directory
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                dimension = int(match.group(1)) if match.group(1) else 0
                found.append((dimension, f"{resources_directory.name}/{directory}/{path.name}"))
    return [path for _, path in sorted(found)]


def reference_front_dimension(front_file: Path) -> int | None:
    """Count the objectives of a reference front: the columns of its first line.

    Args:
        front_file: A reference front file (comma-separated, no header).

    Returns:
        The number of objectives, or None if the file cannot be read or is empty.
    """
    try:
        with front_file.open() as lines:
            first_line = next(lines, "").strip()
    except OSError:
        return None
    return len(first_line.split(",")) if first_line else None


def default_reference_front(candidates: list[str]) -> str | None:
    """Pick the reference front to start from, when there is no doubt about it.

    Args:
        candidates: The files `reference_front_candidates` found.

    Returns:
        The only candidate, or None when there are none or several (the problem has fronts for
        different numbers of objectives, and which one fits depends on the problem's instance).
    """
    return candidates[0] if len(candidates) == 1 else None
