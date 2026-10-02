"""Plotly figures of the fronts a solve run found."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

REFERENCE_FRONT_COLOR = "lightgrey"


def build_front_figure(
    fronts: dict[int, pd.DataFrame], reference_front: pd.DataFrame | None = None
) -> go.Figure:
    """Plot the fronts of some runs, over the reference front.

    Two objectives give a scatter, three a 3D scatter, and more a parallel-coordinates plot (one
    line per solution, colored by run); the reference front is shown for the first two.

    Args:
        fronts: The objective values of each run, by run number, with columns f1, f2, ... as
            `solve_results.read_run_fronts` returns them.
        reference_front: The problem's reference front, with the same columns, or None.

    Returns:
        The figure; empty when there are no fronts.
    """
    if not fronts:
        return go.Figure()
    combined = pd.concat([front.assign(Run=run) for run, front in fronts.items()])
    objectives = [column for column in combined.columns if column != "Run"]
    combined["Run"] = combined["Run"].astype(str)
    if len(objectives) >= 4:
        return _parallel_coordinates(combined, objectives)
    if len(objectives) == 3:
        return _scatter_3d(combined, reference_front)
    return _scatter_2d(combined, reference_front)


def _scatter_2d(combined: pd.DataFrame, reference_front: pd.DataFrame | None) -> go.Figure:
    figure = px.scatter(combined, x="f1", y="f2", color="Run")
    if reference_front is not None:
        figure.add_trace(
            go.Scatter(
                x=reference_front["f1"],
                y=reference_front["f2"],
                mode="lines",
                name="Reference front",
                line={"color": REFERENCE_FRONT_COLOR},
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
