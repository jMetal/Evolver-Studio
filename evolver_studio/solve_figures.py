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
        ordered = reference_front.sort_values("f1")
        figure.add_trace(
            go.Scatter(
                x=ordered["f1"],
                y=ordered["f2"],
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


def build_live_front_figure(
    solutions: pd.DataFrame, reference_front: pd.DataFrame | None = None
) -> go.Figure:
    """Plot the solutions of the run in progress, over the reference front.

    The dominated solutions (when Evolver writes the whole population) are drawn small and grey,
    the non-dominated ones in color. Two objectives give a scatter, three a 3D scatter, and more a
    parallel-coordinates plot of the non-dominated solutions.

    Args:
        solutions: The current solutions, with the columns Run, Evaluations, NonDominated and
            f1, f2, ... as `solve_results.read_current_front` returns them.
        reference_front: The problem's reference front, with columns f1, f2, ..., or None.

    Returns:
        The figure, titled with the run and its evaluations so far.
    """
    objectives = [column for column in solutions.columns if column.startswith("f")]
    run, evaluations = int(solutions["Run"].iloc[0]), int(solutions["Evaluations"].iloc[0])
    nondominated = solutions[solutions["NonDominated"] == 1]
    dominated = solutions[solutions["NonDominated"] == 0]
    if len(objectives) >= 4:
        figure = _parallel_coordinates(
            nondominated.assign(Run=run).astype({"Run": str}), objectives
        )
    elif len(objectives) == 3:
        figure = go.Figure()
        _add_reference(figure, reference_front, 3)
        _add_solutions(figure, dominated, "Dominated", "lightgrey", 2, 3)
        _add_solutions(figure, nondominated, "Non-dominated", "#1f77b4", 3, 3)
    else:
        figure = go.Figure()
        _add_reference(figure, reference_front, 2)
        _add_solutions(figure, dominated, "Dominated", "lightgrey", 5, 2)
        _add_solutions(figure, nondominated, "Non-dominated", "#1f77b4", 7, 2)
        figure.update_layout(xaxis_title="f1", yaxis_title="f2")
    figure.update_layout(title=f"Run {run}, {evaluations} evaluations")
    return figure


def _add_reference(
    figure: go.Figure, reference_front: pd.DataFrame | None, dimensions: int
) -> None:
    if reference_front is None:
        return
    if dimensions == 3:
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
        return
    # Some fronts (LZ09) are not sorted: a line through them would zigzag.
    ordered = reference_front.sort_values("f1")
    figure.add_trace(
        go.Scatter(
            x=ordered["f1"],
            y=ordered["f2"],
            mode="lines",
            name="Reference front",
            line={"color": REFERENCE_FRONT_COLOR},
        )
    )


def _add_solutions(
    figure: go.Figure, solutions: pd.DataFrame, name: str, color: str, size: float, dimensions: int
) -> None:
    if solutions.empty:
        return
    marker = {"size": size, "color": color}
    if dimensions == 3:
        figure.add_trace(
            go.Scatter3d(
                x=solutions["f1"],
                y=solutions["f2"],
                z=solutions["f3"],
                mode="markers",
                name=name,
                marker=marker,
            )
        )
        return
    figure.add_trace(
        go.Scatter(x=solutions["f1"], y=solutions["f2"], mode="markers", name=name, marker=marker)
    )
