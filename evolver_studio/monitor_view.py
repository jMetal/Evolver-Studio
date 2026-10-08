"""The tabs that monitor a training run, while it runs and once it has finished.

They read the files Evolver appends to through `training_monitor`, which follows them from where
it stopped reading, so that a run of hours does not slow the page down.
"""

import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from evolver_studio.evolver_client import RunStatus
from evolver_studio.live_front import build_front_figure
from evolver_studio.progress import training_progress
from evolver_studio.results import (
    deduplicate_consecutive_checkpoints,
    last_n_checkpoints,
    load_indicators,
)
from evolver_studio.slider_state import next_slider_value
from evolver_studio.training_monitor import (
    KEPT_RECENT_CHECKPOINTS,
    CheckpointReader,
    VarConfFollower,
    build_convergence_figure,
    build_population_figure,
    format_duration,
    summarize_run,
    tail_text,
)
from evolver_studio.training_results import TrainedConfiguration

SAMPLES_KEPT = 60
LOG_LINES = 40
FOLLOW_LATEST = "Follow the latest checkpoint"


class MonitorState:
    """What a monitor remembers between polls: the files followed and the pace seen.

    Attributes:
        started_at: When the run started, or None if it is not known.
        front: The follower of INDICATORS.csv.
        population: The follower of POPULATION_INDICATORS.csv, or None when the run does not
            write its population.
        var_conf: The follower of VAR_CONF.txt.
        log_file: The runner's log.
        kept_checkpoints: How many of the latest checkpoints are kept whole to browse; a finished
            run can keep all of them.
        samples: What the polls saw each time the evaluations changed: the monotonic time and
            the evaluations done.
    """

    def __init__(
        self,
        started_at: datetime | None,
        indicators_csv: Path,
        population_csv: Path | None,
        var_conf_file: Path,
        log_file: Path,
        kept_checkpoints: int = KEPT_RECENT_CHECKPOINTS,
    ) -> None:
        self.started_at = started_at
        self.front = CheckpointReader(indicators_csv, kept_checkpoints)
        self.population = (
            CheckpointReader(population_csv, kept_checkpoints) if population_csv else None
        )
        self.var_conf = VarConfFollower(var_conf_file)
        self.log_file = log_file
        self.samples: deque[tuple[float, int]] = deque(maxlen=SAMPLES_KEPT)

    def poll(self) -> None:
        """Read what Evolver appended to the files since the last poll."""
        self.front.poll()
        if self.population is not None:
            self.population.poll()
        self.var_conf.poll()

    def record(self, status: RunStatus) -> None:
        """Remember the evaluations a poll saw, when they are not the ones of the last one."""
        if not self.samples or self.samples[-1][1] != status.evaluations_done:
            self.samples.append((time.monotonic(), status.evaluations_done))


def sync_last_n_slider_value(slider_key: str, available: int) -> None:
    """Pre-seed the "last N" slider's session-state value before it's created.

    Called with no `value=` kwarg on the slider itself, since Streamlit
    re-applies `value=` whenever `max_value` changes (not only on first
    creation), which would silently override a value the user had chosen.

    Args:
        slider_key: The slider's widget key.
        available: The current number of distinct checkpoints to show by
            default, before the user narrows it.
    """
    tracked_key = f"{slider_key}_auto_tracked_available"
    next_value, next_tracked = next_slider_value(
        st.session_state.get(slider_key), st.session_state.get(tracked_key), available
    )
    st.session_state[slider_key] = next_value
    st.session_state[tracked_key] = next_tracked


def render_front_with_slider(history: pd.DataFrame, slider_key: str, chart_key: str) -> None:
    """Draw the front evolution, letting the viewer narrow it to the last N checkpoints.

    Early checkpoints often have much larger indicator values than later,
    converged ones, which can swamp the late-stage detail in a combined plot.

    Args:
        history: Deduplicated checkpoint history to show.
        slider_key: Stable widget key so the chosen N persists across reruns.
        chart_key: Stable widget key for the chart itself.
    """
    available = history["Evaluation"].nunique()
    if available <= 1:
        # st.slider rejects min_value == max_value; nothing to narrow down yet anyway.
        st.plotly_chart(build_front_figure(history), width="stretch", key=chart_key)
        return
    sync_last_n_slider_value(slider_key, available)
    n = st.slider("Show last N fronts", min_value=1, max_value=available, key=slider_key)
    figure = build_front_figure(last_n_checkpoints(history, n))
    st.plotly_chart(figure, width="stretch", key=chart_key)


def render_indicator_front(indicators_csv: Path, run_id: str) -> None:
    """Plot the indicator front's evolution across all checkpoints written.

    Args:
        indicators_csv: Path to the completed run's INDICATORS.csv.
        run_id: The run's identifier, to key the "last N" slider.
    """
    history = load_indicators(indicators_csv)
    if history.empty:
        st.info("No indicator data was written.")
        return
    deduplicated = deduplicate_consecutive_checkpoints(history)
    render_front_with_slider(deduplicated, f"final_last_n_slider_{run_id}", "final_indicator_front")


def run_started_at(run_id: str) -> datetime | None:
    """When a run started, from its timestamp-based identifier."""
    try:
        return datetime.strptime(run_id, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def tab_names(with_population: bool) -> list[str]:
    """The tabs of the monitor of a run in progress."""
    return [
        "Overview",
        "Front",
        *(["Population"] if with_population else []),
        "Convergence",
        "Best so far",
        "Log",
    ]


def render_overview(state: MonitorState, status: RunStatus | None, process_alive: bool) -> None:
    """Show how far the run is, its pace, the time left and the best values found so far.

    Args:
        state: The monitor's state.
        status: The run's latest status, or None before it writes one.
        process_alive: Whether the run's process is alive.
    """
    if status is None:
        st.progress(0.0, text="Starting…")
        return
    st.progress(*training_progress(status))
    summary = summarize_run(
        status,
        state.started_at,
        list(state.samples),
        datetime.now(),
        time.monotonic(),
        process_alive,
    )
    columns = st.columns(4)
    columns[0].metric("Elapsed", format_duration(summary.elapsed_seconds))
    done = f"{summary.evaluations_done:,}"
    if summary.evaluations_total:
        done += f" / {summary.evaluations_total:,}"
    columns[1].metric("Meta-evaluations", done)
    rate = summary.evaluations_per_minute
    columns[2].metric("Pace", f"{rate:,.0f} / min" if rate else "—")
    left = format_duration(summary.remaining_seconds)
    columns[3].metric("Time left (estimate)", f"~ {left}" if summary.remaining_seconds else left)
    if summary.stuck:
        st.warning(
            f"No progress for {format_duration(summary.seconds_without_progress)}, much longer "
            "than between its updates, and the process is alive. It may be stuck: look at the "
            "Log tab."
        )
    _render_best_values(state)


def _render_best_values(state: MonitorState) -> None:
    summaries = state.front.summaries
    if not summaries:
        st.caption("No checkpoint yet: Evolver writes the first one after the update frequency.")
        return
    first, latest = summaries[0], summaries[-1]
    st.markdown("**Best of the front so far** (lower is better), and its change since the first:")
    columns = st.columns(max(len(latest.minimum), 1))
    for column, indicator in zip(columns, latest.minimum, strict=False):
        change = latest.minimum[indicator] - first.minimum[indicator]
        column.metric(
            indicator,
            f"{latest.minimum[indicator]:.4g}",
            delta=f"{change:.3g}" if change else None,
            delta_color="inverse",
        )
    minutes = state.var_conf.minutes.get(latest.evaluation)
    when = f" · {format_duration(minutes * 60)} of computing" if minutes is not None else ""
    st.caption(
        f"Last checkpoint: evaluation {latest.evaluation:,}{when} · {latest.size} "
        f"configurations on the front · {len(summaries)} checkpoints."
    )


def render_population(state: MonitorState, key: str) -> None:
    """Show the meta-optimizer's population at a checkpoint, with the front over it.

    Args:
        state: The monitor's state.
        key: Prefixes the widgets' keys.
    """
    if state.population is None:
        st.info(
            "This run does not write its population. Tick **Show the meta-optimizer's "
            "population while it runs** before launching to see it here."
        )
        return
    summaries = state.population.summaries
    kept = _kept_evaluations(state.population)
    if not summaries or not kept:
        st.caption("No checkpoint of the population yet.")
        return
    follow = st.checkbox(FOLLOW_LATEST, value=True, key=f"{key}_follow")
    if follow or len(kept) == 1:
        evaluation = kept[-1]
    else:
        evaluation = st.select_slider(
            "Checkpoint (meta-evaluations)", options=kept, value=kept[-1], key=f"{key}_checkpoint"
        )
    log_scale = st.checkbox(
        "Logarithmic axes",
        key=f"{key}_log",
        help="The first checkpoints are far from the later ones: a logarithmic scale shows both.",
    )
    indicators = state.population.indicators
    if len(indicators) < 2:
        st.info("The population viewer needs two meta-objectives.")
        return
    figure = build_population_figure(
        state.population.checkpoint(evaluation),
        state.front.checkpoint(evaluation),
        state.front.first,
        (indicators[0], indicators[1]),
        log_scale=log_scale,
    )
    st.plotly_chart(figure, width="stretch", key=f"{key}_chart")
    st.caption(
        f"{len(summaries)} checkpoints of the population; the latest {len(kept) - 1} and the "
        "first are kept to browse."
    )


def _kept_evaluations(reader: CheckpointReader) -> list[int]:
    blocks = [reader.first, *reader.recent()]
    return sorted({int(b["Evaluation"].iloc[0]) for b in blocks if b is not None and len(b)})


def render_convergence(state: MonitorState, key: str) -> None:
    """Show how the best, the median and the worst of the front improve over the run.

    Args:
        state: The monitor's state.
        key: Prefixes the widgets' keys.
    """
    summaries = state.front.summaries
    if not summaries:
        st.caption("No checkpoint yet.")
        return
    columns = st.columns(2)
    by_time = columns[0].radio(
        "x axis", ["Meta-evaluations", "Computing time"], horizontal=True, key=f"{key}_x"
    )
    log_scale = columns[1].checkbox("Logarithmic values", key=f"{key}_log")
    minutes = state.var_conf.minutes if by_time == "Computing time" else None
    figure = build_convergence_figure(summaries, minutes=minutes, log_scale=log_scale)
    st.plotly_chart(figure, width="stretch", key=f"{key}_chart")


def render_best_configurations(
    state: MonitorState, key: str, on_validate: Callable[[TrainedConfiguration], None] | None = None
) -> None:
    """List the configurations of the latest front, and let one be copied or downloaded.

    Args:
        state: The monitor's state.
        key: Prefixes the widgets' keys.
        on_validate: Called with the chosen configuration when the viewer asks to validate it;
            without it there is no such button.
    """
    latest = state.var_conf.latest
    if latest is None or not latest.configurations:
        st.caption("No checkpoint yet.")
        return
    when = f"evaluation {latest.evaluation:,}" if latest.evaluation is not None else "the last one"
    st.caption(
        f"The {len(latest.configurations)} configurations of the front at {when}: what the "
        "training has found so far, in the order of the file."
    )
    table = pd.DataFrame(
        [{**c.objectives, "configuration": c.configuration} for c in latest.configurations]
    )
    st.dataframe(table, hide_index=True, width="stretch", height="content")
    index = st.selectbox(
        "Configuration",
        range(len(latest.configurations)),
        format_func=lambda i: f"{i + 1}. {latest.configurations[i].label}",
        key=f"{key}_choice",
    )
    chosen = latest.configurations[index].configuration
    st.code(chosen, language=None, wrap_lines=True)
    columns = st.columns(2)
    columns[0].download_button(
        "Download this configuration (.txt)",
        chosen + "\n",
        file_name=f"configuration_{index + 1}.txt",
        key=f"{key}_download",
        help="The format a solve request and Validation take: --parameter value pairs.",
    )
    if on_validate is not None and columns[1].button(
        "Validate this configuration",
        key=f"{key}_validate",
        help="Opens Validation with it as the tuned configuration, ready to compare with the "
        "default configurations of other algorithms. (You can also copy it from the box above "
        "and paste it there.)",
    ):
        on_validate(latest.configurations[index])


def render_log(state: MonitorState) -> None:
    """Show the last lines of what Evolver prints.

    Args:
        state: The monitor's state.
    """
    text = tail_text(state.log_file, LOG_LINES)
    if text is None:
        st.caption("No log yet.")
        return
    st.caption(f"The last {LOG_LINES} lines of `{state.log_file.name}`.")
    st.code(text or "(empty)", language=None, wrap_lines=True)
