"""The hypervolume of a front, computed as Evolver computes its hypervolume indicators.

Evolver measures a front against the reference front of its problem (`cli.solving`'s
`SolveRunner`): it normalizes the front with the minimum and maximum of each objective of the
reference front, and jMetal's `PISAHypervolume` then clips each normalized value to [0, 1] and
measures the volume dominated up to the point (1, ..., 1). Its `NormalizedHypervolume` is
1 - HV(front) / HV(reference front), and `HypervolumeMinus` is -HV(front). This is the HV of those
same terms: for every front of a validation study, 1 - hypervolume(front) / hypervolume(reference
front) equals the NormalizedHypervolume Evolver wrote, to the last digits.

Evolver registers no maximized hypervolume, so Evolver-Studio computes it from the fronts a study
keeps (`run-K/FUN.csv`). The volume is computed by moocore, in C: fast for any number of
objectives.

Nothing here depends on Streamlit.
"""

import moocore
import numpy as np


def hypervolume(front: np.ndarray, reference_front: np.ndarray) -> float:
    """The hypervolume of a front, normalized with a reference front as Evolver does.

    Args:
        front: The objective values of the solutions, one row per solution; dominated ones do
            not change the result.
        reference_front: The problem's reference front, with the same number of objectives.

    Returns:
        The hypervolume, between 0 and 1 (higher is better).

    Raises:
        ValueError: If the fronts have different numbers of objectives.
    """
    front = np.asarray(front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    if front.size == 0:
        return 0.0
    if front.shape[1] != reference_front.shape[1]:
        raise ValueError(
            f"The front has {front.shape[1]} objectives and the reference front "
            f"{reference_front.shape[1]}"
        )
    lower = reference_front.min(axis=0)
    span = reference_front.max(axis=0) - lower
    # An objective the reference front does not vary in normalizes to 0, as no division is made.
    span[span == 0] = 1.0
    normalized = np.clip((front - lower) / span, 0.0, 1.0)
    return float(moocore.hypervolume(normalized, ref=np.ones(front.shape[1])))
