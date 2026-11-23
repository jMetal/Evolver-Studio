"""Plotly figures of the fronts a solve run found."""

from collections.abc import Mapping

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

REFERENCE_FRONT_COLOR = "lightgrey"
INFEASIBLE_COLOR = "#D94A57"


def build_front_figure(
    fronts: dict[int, pd.DataFrame],
    reference_front: pd.DataFrame | None = None,
    violations: Mapping[int, pd.Series] | None = None,
) -> go.Figure:
    """Plot the fronts of some runs, over the reference front.

    Two objectives give a scatter, three a 3D scatter, and more a parallel-coordinates plot (one
    line per solution, colored by run); the reference front is shown for the first two. For a
    problem with constraints the infeasible solutions are drawn apart, in red; the
    parallel-coordinates plot leaves them out unless there are no others.

    Args:
        fronts: The objective values of each run, by run number, with columns f1, f2, ... as
            `solve_results.read_run_fronts` returns them.
        reference_front: The problem's reference front, with the same columns, or None.
        violations: The violation of each solution of each run, for a problem with constraints
            (0 if it is feasible); None or empty for one without.

    Returns:
        The figure; empty when there are no fronts.
    """
    if not fronts:
        return go.Figure()
    violations = violations or {}
    combined = pd.concat(
        [
            front.assign(Run=run, Infeasible=(violations[run] > 0) if run in violations else False)
            for run, front in fronts.items()
        ]
    )
    infeasible = combined.pop("Infeasible").astype(bool)
    objectives = [column for column in combined.columns if column != "Run"]
    combined["Run"] = combined["Run"].astype(str)
    if len(objectives) >= 4:
        shown = combined if infeasible.all() else combined[~infeasible]
        return _parallel_coordinates(shown, objectives)
    combined.loc[infeasible, "Run"] += " · infeasible"
    if len(objectives) == 3:
        figure = _scatter_3d(combined, reference_front)
    else:
        # With constraints a front is often disconnected: a line would join its parts.
        figure = _scatter_2d(combined, reference_front, "markers" if violations else "lines")
    for trace in figure.data:
        if trace.name.endswith("infeasible"):
            trace.marker.update(color=INFEASIBLE_COLOR, symbol="x")
    return figure


def _scatter_2d(
    combined: pd.DataFrame, reference_front: pd.DataFrame | None, reference_mode: str
) -> go.Figure:
    figure = px.scatter(combined, x="f1", y="f2", color="Run")
    if reference_front is not None:
        ordered = reference_front.sort_values("f1")
        figure.add_trace(
            go.Scatter(
                x=ordered["f1"],
                y=ordered["f2"],
                mode=reference_mode,
                name="Reference front",
                line={"color": REFERENCE_FRONT_COLOR},
                marker={"color": REFERENCE_FRONT_COLOR, "size": 3},
            )
        )
        figure.data = (figure.data[-1], *figure.data[:-1])
    return figure


def _scatter_3d(combined: pd.DataFrame, reference_front: pd.DataFrame | None) -> go.Figure:
    figure = px.scatter_3d(combined, x="f1", y="f2", z="f3", color="Run")
    figure.update_traces(marker={"size": 3})
    if reference_front is not None:
        figure.add_trace(
            go.Scatter3d(
                x=reference_front["f1"],
                y=reference_front["f2"],
                z=reference_front["f3"],
                mode="markers",
                name="Reference front",
                marker={"size": 1.5, "color": REFERENCE_FRONT_COLOR},
            )
        )
    return figure


def _parallel_coordinates(combined: pd.DataFrame, objectives: list[str]) -> go.Figure:
    runs = sorted(combined["Run"].unique(), key=int)
    colored = combined.assign(RunNumber=combined["Run"].astype(int))
    figure = px.parallel_coordinates(
        colored, dimensions=objectives, color="RunNumber", labels={"RunNumber": "Run"}
    )
    figure.update_layout(coloraxis_colorbar={"tickvals": [int(run) for run in runs]})
    return figure
