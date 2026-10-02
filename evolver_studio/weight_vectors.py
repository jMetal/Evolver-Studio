"""The weight vector files the decomposition-based algorithms (MOEA/D, RVEA) read.

They are named after the number of objectives and the population size they serve
(`W<objectives>D_<population size>.dat`), and an algorithm fails when none matches its population,
so the sizes a problem can be solved with are the ones that have a file.
"""

import re
from pathlib import Path

WEIGHT_VECTOR_FILE_PATTERN = re.compile(r"W(\d+)D_(\d+)\.dat")


def available_population_sizes(directory: Path, objectives: int) -> list[int]:
    """List the population sizes there is a weight vector file for.

    Args:
        directory: The weight vector files directory.
        objectives: The number of objectives of the problem.

    Returns:
        The sizes, ascending; empty if the directory does not exist or has none for that number of
        objectives.
    """
    if not directory.is_dir():
        return []
    sizes = []
    for path in directory.iterdir():
        match = WEIGHT_VECTOR_FILE_PATTERN.fullmatch(path.name)
        if match and int(match.group(1)) == objectives:
            sizes.append(int(match.group(2)))
    return sorted(sizes)
