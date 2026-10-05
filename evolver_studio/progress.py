"""Estimating how far a run is: the time left, and how to write a duration."""


def estimate_remaining_seconds(
    elapsed_seconds: float, evaluations_done: int, evaluations_total: int
) -> float | None:
    """Estimate the time a run still needs, assuming it keeps its pace.

    Args:
        elapsed_seconds: The time since the run started.
        evaluations_done: The evaluations performed so far.
        evaluations_total: The evaluations of the whole run.

    Returns:
        The seconds left, or None while nothing is done yet (there is no pace to extrapolate).
    """
    if evaluations_done <= 0 or elapsed_seconds <= 0:
        return None
    remaining = max(evaluations_total - evaluations_done, 0)
    return elapsed_seconds * remaining / evaluations_done


def format_duration(seconds: float) -> str:
    """Write a duration in the largest units that read well: "8 s", "2 min 5 s", "1 h 3 min".

    Args:
        seconds: The duration.

    Returns:
        The text, rounded to the second (to the minute from an hour on).
    """
    total = round(seconds)
    if total < 60:
        return f"{total} s"
    minutes, rest = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} min {rest} s" if rest else f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"
