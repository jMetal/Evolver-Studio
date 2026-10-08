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


def study_running_label(status: RunStatus | None) -> str:
    """Describe a validation study in progress, e.g. "Running · 3 of 12 jobs finished".

    The study's status counts jobs where a run's counts evaluations.

    Args:
        status: The study's status, or None while it has written none.

    Returns:
        The label; "Running…" alone while the status says nothing yet.
    """
    if status is None or status.max_evaluations <= 0:
        return "Running…"
    done = min(status.evaluations_done, status.max_evaluations)
    return f"Running · {done} of {status.max_evaluations} jobs finished"


def training_progress(status: RunStatus) -> tuple[float, str]:
    """The fraction of a training run done, and what to say about it.

    Args:
        status: The run's status.

    Returns:
        A value in [0, 1] and its text: the evaluations done of those to do, or the computing time
        spent of the limit when the run is limited by time (it has no evaluations to do).
    """
    if status.max_computing_time_minutes:
        elapsed = status.elapsed_minutes or 0.0
        fraction = min(elapsed / status.max_computing_time_minutes, 1.0)
        return fraction, (
            f"{elapsed:.1f} of {status.max_computing_time_minutes:g} min · "
            f"{status.evaluations_done} evaluations"
        )
    fraction = min(status.evaluations_done / max(status.max_evaluations, 1), 1.0)
    return fraction, f"{status.evaluations_done}/{status.max_evaluations}"
