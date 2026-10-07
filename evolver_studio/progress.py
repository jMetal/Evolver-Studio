"""How a run in progress is described: which evaluation it is at, of how many."""

from evolver_studio.evolver_client import RunStatus


def running_label(status: RunStatus | None, runs: int) -> str:
    """Describe a run in progress in one line, e.g. "Running · evaluation 12,000 of 25,000".

    Evolver counts the evaluations over all the independent runs, so with several runs the label
    also says which one is in progress.

    Args:
        status: The run's status, or None while it has written none.
        runs: The number of independent runs of the request.

    Returns:
        The label; "Running…" alone while the status says nothing yet.
    """
    if status is None or status.evaluations_done <= 0 or status.max_evaluations <= 0:
        return "Running…"
    done = min(status.evaluations_done, status.max_evaluations)
    label = f"Running · evaluation {done:,} of {status.max_evaluations:,}"
    if runs > 1:
        per_run = status.max_evaluations // runs
        current = min(done // max(per_run, 1) + 1, runs)
        label += f" · run {current} of {runs}"
    return label
