"""The charts of the analysis of a validation study.

Boxplots of an indicator, a heatmap of the A12 effect sizes against the pivot, the computing time
of each algorithm, Demšar's critical difference plot, and the fronts of chosen runs: for two and
three objectives a grid with a panel for the reference front and one per algorithm, and for more,
a parallel-coordinates chart for the reference front and one per algorithm, normalized with the
reference front, which each shows in gray behind the algorithm's front.

Nothing here depends on Streamlit.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evolver_studio.solve_figures import REFERENCE_FRONT_COLOR

COLORS = px.colors.qualitative.Plotly
PANEL_HEIGHT = 260
GRID_COLUMNS = 3
GRID_2D_ROW_HEIGHT = 340
GRID_3D_ROW_HEIGHT = 480
# The reference front in its own panel; in the algorithms' panels it is REFERENCE_FRONT_COLOR.
REFERENCE_PANEL_COLOR = "dimgray"


def boxplot_figure(runs: pd.DataFrame, indicator: str) -> go.Figure:
    """The values of an indicator over the runs, a box per algorithm, a panel per problem."""
    figure = px.box(
        runs,
        x="contender",
        y=indicator,
        color="contender",
        facet_col="problem",
        facet_col_wrap=3,
        labels={"contender": "", indicator: indicator},
    )
    figure.update_yaxes(matches=None, showticklabels=True)
    figure.update_xaxes(showticklabels=False)
    figure.for_each_annotation(lambda a: a.update(text=a.text.split("=", 1)[-1]))
    figure.update_layout(showlegend=True, height=320 * -(-runs["problem"].nunique() // 3))
    return figure


def effect_size_figure(comparison: pd.DataFrame, pivot: str) -> go.Figure:
    """The A12 of the pivot against each algorithm on each problem, as a heatmap.

    Args:
        comparison: What `validation_stats.compare_with_pivot` returns.
        pivot: The pivot, for the title.

    Returns:
        Problems in rows, algorithms in columns; blue where the pivot is better (A12 above 0.5),
        red where it is worse, the value and the mark of the Wilcoxon test in each cell.
    """
    marks = {"pivot better": "+", "pivot worse": "-", "no significant difference": "="}
    table = comparison.pivot(index="problem", columns="contender", values="a12")
    text = comparison.assign(
        label=comparison["a12"].map("{:.2f}".format) + " " + comparison["verdict"].map(marks)
    ).pivot(index="problem", columns="contender", values="label")
    problems = list(dict.fromkeys(comparison["problem"]))
    contenders = list(dict.fromkeys(comparison["contender"]))
    table = table.reindex(index=problems, columns=contenders)
    text = text.reindex(index=problems, columns=contenders)
    figure = go.Figure(
        go.Heatmap(
            z=table.to_numpy(),
            x=contenders,
            y=problems,
            text=text.to_numpy(),
            texttemplate="%{text}",
            colorscale="RdBu",
            zmin=0,
            zmax=1,
            colorbar={"title": "A12"},
            hovertemplate="%{y} · %{x}<br>A12 = %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"A12 of {pivot} against each algorithm",
        yaxis={"autorange": "reversed"},
        height=max(300, 40 * len(problems) + 140),
    )
    return figure


def cost_figure(runs: pd.DataFrame) -> go.Figure:
    """The median computing time of a run of each algorithm on each problem, in seconds."""
    times = (
        runs.groupby(["problem", "contender"], sort=False)["TimeMs"]
        .median()
        .div(1000)
        .reset_index()
    )
    figure = px.bar(
        times,
        x="problem",
        y="TimeMs",
        color="contender",
        barmode="group",
        labels={"TimeMs": "Median time of a run (s)", "problem": "", "contender": ""},
    )
    figure.update_layout(height=420)
    return figure


def critical_difference_figure(ranks: pd.Series, difference: float, groups: list[list[str]]):
    """Demšar's critical difference plot.

    An axis of average ranks (1, the best, on the left), each algorithm hanging from its rank,
    the critical difference as a segment above the axis, and below it a bar joining each group of
    algorithms that are not significantly different.

    Args:
        ranks: The average ranks, best first.
        difference: Nemenyi's critical difference.
        groups: The groups of `validation_stats.nonsignificant_groups`.

    Returns:
        The figure.
    """
    ordered = ranks.sort_values(kind="stable")
    count = len(ordered)
    figure = go.Figure()
    axis_y = 0.0
    figure.add_shape(type="line", x0=1, x1=count, y0=axis_y, y1=axis_y, line={"color": "black"})
    for tick in range(1, count + 1):
        figure.add_shape(
            type="line", x0=tick, x1=tick, y0=axis_y, y1=axis_y + 0.08, line={"color": "black"}
        )
        figure.add_annotation(x=tick, y=axis_y + 0.18, text=str(tick), showarrow=False)
    figure.add_shape(
        type="line", x0=1, x1=1 + difference, y0=0.55, y1=0.55, line={"color": "black", "width": 2}
    )
    figure.add_annotation(
        x=1 + difference / 2, y=0.72, text=f"CD = {difference:.2f}", showarrow=False
    )
    half = (count + 1) // 2
    for position, (name, rank) in enumerate(ordered.items()):
        left = position < half
        level = -0.35 - 0.3 * (position if left else count - 1 - position)
        edge = 0.6 if left else count + 0.4
        figure.add_shape(
            type="line", x0=rank, x1=rank, y0=axis_y, y1=level, line={"color": "black"}
        )
        figure.add_shape(type="line", x0=rank, x1=edge, y0=level, y1=level, line={"color": "black"})
        figure.add_annotation(
            x=edge,
            y=level,
            text=f"{name} ({rank:.2f})",
            showarrow=False,
            xanchor="right" if left else "left",
        )
    for index, group in enumerate(groups):
        low, high = ordered[group[0]], ordered[group[-1]]
        y = -0.12 - 0.08 * index
        figure.add_shape(type="line", x0=low - 0.03, x1=high + 0.03, y0=y, y1=y, line={"width": 5})
    lowest = -0.35 - 0.3 * half
    figure.update_xaxes(visible=False, range=[0.6 - 0.25 * count, count + 0.4 + 0.25 * count])
    figure.update_yaxes(visible=False, range=[lowest - 0.2, 0.9])
    figure.update_layout(height=max(260, 70 * half + 160), plot_bgcolor="white")
    return figure


def fronts_grid_figure(
    fronts: dict[str, pd.DataFrame],
    reference_front: pd.DataFrame | None,
    reference_behind: bool = True,
    columns: int = GRID_COLUMNS,
) -> go.Figure:
    """The fronts of some algorithms on a problem with two or three objectives, a panel each.

    The first panel is the reference front, and then one per algorithm, in the order given (the
    pivot first), all on the same axes so that they can be compared. Two objectives give
    scatters, three 3D scatters.

    Args:
        fronts: The objective values (columns f1, f2, ...) of the front of each algorithm, by
            the label to show it with.
        reference_front: The problem's reference front, with the same columns, or None.
        reference_behind: Whether each algorithm's panel also shows the reference front, faint.
        columns: Panels per row.

    Returns:
        The figure; empty when there are no fronts.
    """
    if not fronts:
        return go.Figure()
    objectives = list(next(iter(fronts.values())).columns)
    if len(objectives) >= 4:
        raise ValueError("Four objectives or more: use parallel_fronts_figures")
    three = len(objectives) == 3
    panels: list[tuple[str, pd.DataFrame | None, str | None]] = []
    if reference_front is not None:
        panels.append(("Reference front", None, None))
    for index, (label, front) in enumerate(fronts.items()):
        panels.append((label, front, COLORS[index % len(COLORS)]))
    columns = min(columns, len(panels))
    rows = -(-len(panels) // columns)
    figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[label for label, _, _ in panels],
        specs=[[{"type": "scene" if three else "xy"}] * columns for _ in range(rows)],
        shared_xaxes="all" if not three else False,
        shared_yaxes="all" if not three else False,
        horizontal_spacing=0.06,
        vertical_spacing=0.12 / rows if not three else 0.04,
    )
    everything = pd.concat(
        [f[objectives] for f in fronts.values()]
        + ([reference_front[objectives]] if reference_front is not None else [])
    )
    for position, (_, front, color) in enumerate(panels):
        row, column = divmod(position, columns)
        place = {"row": row + 1, "col": column + 1}
        if reference_front is not None and (front is None or reference_behind):
            figure.add_trace(
                _reference_trace(reference_front, three, faint=front is not None), **place
            )
        if front is not None:
            figure.add_trace(_front_trace(front, three, color), **place)
    figure.update_layout(showlegend=False)
    if three:
        ranges = {
            axis: [float(everything[objective].min()), float(everything[objective].max())]
            for axis, objective in zip(("xaxis", "yaxis", "zaxis"), objectives, strict=True)
        }
        for index in range(1, len(panels) + 1):
            scene = "scene" if index == 1 else f"scene{index}"
            figure.update_layout(
                {
                    scene: {
                        axis: {"title": objective, "range": ranges[axis]}
                        for axis, objective in zip(
                            ("xaxis", "yaxis", "zaxis"), objectives, strict=True
                        )
                    }
                }
            )
        figure.update_layout(height=GRID_3D_ROW_HEIGHT * rows)
    else:
        figure.update_xaxes(title_text="f1", row=rows)
        figure.update_yaxes(title_text="f2", col=1)
        figure.update_layout(height=GRID_2D_ROW_HEIGHT * rows + 60)
    return figure


def _reference_trace(reference_front: pd.DataFrame, three: bool, faint: bool):
    color = REFERENCE_FRONT_COLOR if faint else REFERENCE_PANEL_COLOR
    if three:
        return go.Scatter3d(
            x=reference_front["f1"],
            y=reference_front["f2"],
            z=reference_front["f3"],
            mode="markers",
            name="Reference front",
            # Dense: light enough for the surface's shape to show through its own points.
            marker={"size": 1.5, "color": color, "opacity": 0.4 if faint else 0.25},
        )
    ordered = reference_front.sort_values("f1")
    return go.Scatter(
        x=ordered["f1"],
        y=ordered["f2"],
        mode="lines" if faint else "lines+markers",
        name="Reference front",
        line={"color": color},
        marker={"size": 3, "color": color},
    )


def _front_trace(front: pd.DataFrame, three: bool, color: str):
    if three:
        return go.Scatter3d(
            x=front["f1"],
            y=front["f2"],
            z=front["f3"],
            mode="markers",
            marker={"size": 2.5, "color": color},
        )
    return go.Scatter(
        x=front["f1"], y=front["f2"], mode="markers", marker={"color": color, "size": 6}
    )


def parallel_fronts_figures(
    fronts: dict[str, pd.DataFrame], reference_front: pd.DataFrame | None
) -> list[go.Figure]:
    """For four objectives or more: a parallel-coordinates chart per algorithm.

    Each shows the algorithm's front in color over the reference front in gray, every objective
    normalized with the reference front's bounds (0 its lowest value, 1 its highest), on the same
    scale in every chart so that they can be compared.

    Args:
        fronts: The objective values (columns f1, f2, ...) of the front of each algorithm, by
            the label to show it with.
        reference_front: The problem's reference front, with the same columns, or None (the
            fronts' own bounds normalize them then).

    Returns:
        The reference front's figure, when there is one, and then a figure per algorithm, in the
        order of `fronts`.
    """
    if not fronts:
        return []
    objectives = list(next(iter(fronts.values())).columns)
    bounds = reference_front if reference_front is not None else pd.concat(fronts.values())
    lower = bounds[objectives].min()
    span = (bounds[objectives].max() - lower).replace(0, 1.0)
    top = max(1.0, *(float(((f[objectives] - lower) / span).max().max()) for f in fronts.values()))
    figures = []
    if reference_front is not None:
        reference = (reference_front[objectives] - lower) / span
        figure = go.Figure(
            go.Parcoords(
                line={"color": REFERENCE_PANEL_COLOR},
                dimensions=[
                    {"label": objective, "values": reference[objective], "range": [0, top]}
                    for objective in objectives
                ],
            )
        )
        figure.update_layout(
            title={"text": "Reference front", "x": 0.01},
            height=PANEL_HEIGHT + 80,
            margin={"t": 80},
        )
        figures.append(figure)
    for index, (label, front) in enumerate(fronts.items()):
        color = COLORS[index % len(COLORS)]
        parts = [front[objectives].assign(_shown=1.0)]
        if reference_front is not None:
            parts.insert(0, reference_front[objectives].assign(_shown=0.0))
        data = pd.concat(parts, ignore_index=True)
        normalized = (data[objectives] - lower) / span
        figure = go.Figure(
            go.Parcoords(
                line={
                    "color": data["_shown"],
                    "colorscale": [[0, REFERENCE_FRONT_COLOR], [1, color]],
                    "cmin": 0,
                    "cmax": 1,
                },
                dimensions=[
                    {"label": objective, "values": normalized[objective], "range": [0, top]}
                    for objective in objectives
                ],
            )
        )
        figure.update_layout(
            title={"text": label, "x": 0.01}, height=PANEL_HEIGHT + 80, margin={"t": 80}
        )
        figures.append(figure)
    return figures
