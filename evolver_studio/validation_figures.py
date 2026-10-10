"""The charts of the analysis of a validation study.

Boxplots of an indicator, a heatmap of the A12 effect sizes against the pivot, the computing time
of each algorithm, Demšar's critical difference plot, and the fronts of chosen runs: overlaid for
two and three objectives, and for more, a parallel-coordinates chart per algorithm, normalized
with the reference front, which each shows in gray behind the algorithm's front.

Nothing here depends on Streamlit.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from evolver_studio.solve_figures import REFERENCE_FRONT_COLOR

COLORS = px.colors.qualitative.Plotly
PANEL_HEIGHT = 260


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


def fronts_figure(
    fronts: dict[str, pd.DataFrame], reference_front: pd.DataFrame | None
) -> go.Figure:
    """The fronts of some algorithms on a problem with two or three objectives, overlaid.

    Two objectives give a scatter over the reference front's line, three a 3D scatter over its
    points (an algorithm is hidden or shown by clicking its name in the legend). For more, see
    `parallel_fronts_figures`.

    Args:
        fronts: The objective values (columns f1, f2, ...) of the front to show of each
            algorithm, by the label to show it with.
        reference_front: The problem's reference front, with the same columns, or None.

    Returns:
        The figure; empty when there are no fronts.
    """
    if not fronts:
        return go.Figure()
    objectives = list(next(iter(fronts.values())).columns)
    if len(objectives) >= 4:
        raise ValueError("Four objectives or more: use parallel_fronts_figures")
    figure = go.Figure()
    three = len(objectives) == 3
    if reference_front is not None:
        ordered = reference_front.sort_values("f1")
        if three:
            figure.add_trace(
                go.Scatter3d(
                    x=ordered["f1"],
                    y=ordered["f2"],
                    z=ordered["f3"],
                    mode="markers",
                    name="Reference front",
                    marker={"size": 2, "color": REFERENCE_FRONT_COLOR, "opacity": 0.4},
                )
            )
        else:
            figure.add_trace(
                go.Scatter(
                    x=ordered["f1"],
                    y=ordered["f2"],
                    mode="lines",
                    name="Reference front",
                    line={"color": REFERENCE_FRONT_COLOR},
                )
            )
    for index, (label, front) in enumerate(fronts.items()):
        color = COLORS[index % len(COLORS)]
        if three:
            figure.add_trace(
                go.Scatter3d(
                    x=front["f1"],
                    y=front["f2"],
                    z=front["f3"],
                    mode="markers",
                    name=label,
                    marker={"size": 3, "color": color},
                )
            )
        else:
            figure.add_trace(
                go.Scatter(
                    x=front["f1"],
                    y=front["f2"],
                    mode="markers",
                    name=label,
                    marker={"color": color, "size": 7},
                )
            )
    if three:
        figure.update_layout(
            scene={"xaxis_title": "f1", "yaxis_title": "f2", "zaxis_title": "f3"}, height=650
        )
    else:
        figure.update_layout(xaxis_title="f1", yaxis_title="f2", height=550)
    return figure


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
        A figure per algorithm, in the order of `fronts`.
    """
    if not fronts:
        return []
    objectives = list(next(iter(fronts.values())).columns)
    bounds = reference_front if reference_front is not None else pd.concat(fronts.values())
    lower = bounds[objectives].min()
    span = (bounds[objectives].max() - lower).replace(0, 1.0)
    top = max(1.0, *(float(((f[objectives] - lower) / span).max().max()) for f in fronts.values()))
    figures = []
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
