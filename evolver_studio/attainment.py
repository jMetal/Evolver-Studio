"""Empirical attainment functions (EAF) of the runs of a validation study, for two objectives.

The EAF of an algorithm on a problem says, for each point of the objective space, in what share
of its runs the front reached it (some solution dominates or equals it). Its summary attainment
surfaces draw the points reached by at least one run (the best), by half of them (the median)
and by all (the worst). The difference of the EAFs of two algorithms shows *where* in the
objective space one reaches more often than the other, which a quality indicator, a single
number, cannot (López-Ibáñez, Paquete and Stützle, 2010).

Computed by moocore, which supports EAF differences for two objectives only.

Nothing here depends on Streamlit.
"""

import moocore
import numpy as np
import pandas as pd

BEST = "best"
MEDIAN = "median"
WORST = "worst"


def _stacked(fronts: dict[int, pd.DataFrame]) -> np.ndarray:
    """The fronts of the runs as one matrix: f1, f2 and the run's position, from 1."""
    blocks = [
        np.column_stack([front[["f1", "f2"]].to_numpy(dtype=float), np.full(len(front), set_id)])
        for set_id, front in enumerate(fronts.values(), start=1)
        if len(front)
    ]
    return np.vstack(blocks) if blocks else np.empty((0, 3))


def attainment_surfaces(fronts: dict[int, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The best, median and worst attainment surfaces of an algorithm's runs on a problem.

    Args:
        fronts: The front (columns f1, f2) of each run, by run number.

    Returns:
        The points of each surface (columns f1, f2), sorted by f1, by `BEST`, `MEDIAN` and
        `WORST`; empty when there are no fronts.
    """
    data = _stacked(fronts)
    runs = len(fronts)
    if not len(data) or runs == 0:
        return {}
    levels = {BEST: 100.0 / runs, MEDIAN: 50.0, WORST: 100.0}
    eaf = moocore.eaf(data[:, :2], data[:, 2], percentiles=sorted(set(levels.values())))
    surfaces = {}
    for name, level in levels.items():
        points = eaf[np.isclose(eaf[:, 2], level)][:, :2]
        surfaces[name] = pd.DataFrame(points, columns=["f1", "f2"]).sort_values("f1")
    return surfaces


def eaf_differences(left: dict[int, pd.DataFrame], right: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Where the left algorithm's runs reach more often than the right one's, and by how much.

    Args:
        left: The front (columns f1, f2) of each run of one algorithm, by run number.
        right: The same, of the other; both should have the same number of runs.

    Returns:
        Rectangles of the objective space, columns `x0`, `y0`, `x1`, `y1` (corners; an infinite
        one is unbounded) and `difference`: the share of the left runs that reach it minus the
        share of the right ones, from -1 to 1. Only the rectangles with a difference; empty when
        either has no front.
    """
    columns = ["x0", "y0", "x1", "y1", "difference"]
    left_data, right_data = _stacked(left), _stacked(right)
    if not len(left_data) or not len(right_data):
        return pd.DataFrame(columns=columns)
    rectangles = moocore.eafdiff(left_data, right_data, rectangles=True)
    table = pd.DataFrame(rectangles, columns=columns)
    table["difference"] = table["difference"] / max(len(left), len(right))
    return table[table["difference"] != 0].reset_index(drop=True)
