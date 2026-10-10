"""The charts of the analysis of a validation study.

Boxplots of an indicator, a heatmap of the A12 effect sizes against the pivot, the computing time
of each algorithm, Demšar's critical difference plot, and the fronts of chosen runs: for two and
three objectives a grid with a panel for the reference front and one per algorithm, and for more,
a parallel-coordinates chart for the reference front and one per algorithm, normalized with the
reference front, which each shows in gray behind the algorithm's front.

Nothing here depends on Streamlit.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evolver_studio.solve_figures import REFERENCE_FRONT_COLOR
from evolver_studio.validation_stats import magnitude

COLORS = px.colors.qualitative.Plotly
PANEL_HEIGHT = 260
EAF_LEVELS = 5
SIMPLEX_POINTS = 3000
COLOR_SCHEME = "color"
GRAY_SCHEME = "gray"
# ColorBrewer's RdBu with 7 classes: from a large effect with the pivot worse (red) to a large one
# with it better (blue), a light neutral gray for a negligible one.
EFFECT_COLORS = ("#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac")
# For black-and-white printing: the larger the effect, the darker, either way.
EFFECT_GRAYS = ("#525252", "#969696", "#d9d9d9", "#f7f7f7", "#d9d9d9", "#969696", "#525252")
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


def effect_size_figure(
    comparison: pd.DataFrame, pivot: str, scheme: str = COLOR_SCHEME
) -> go.Figure:
    """The A12 of the pivot against each algorithm on each problem, as a heatmap.

    The cells are shaded by the magnitude of the effect, as Vargha and Delaney classify it
    (negligible, small, medium, large; `validation_stats.magnitude`), not by the raw value: 0.93
    and 0.99 are both a large effect. In color (ColorBrewer's RdBu, safe for color-blind readers),
    blue where the pivot is better and red where it is worse, a light neutral gray for a
    negligible effect; in grays, for printing in black and white, the darker the larger the
    effect, and the direction is the mark of the cell. Each cell has the A12 and the mark of the
    Wilcoxon test (+, =, -).

    Args:
        comparison: What `validation_stats.compare_with_pivot` returns.
        pivot: The pivot, for the title.
        scheme: `COLOR_SCHEME` or `GRAY_SCHEME`.

    Returns:
        Problems in rows, algorithms in columns.
    """
    marks = {"pivot better": "+", "pivot worse": "-", "no significant difference": "="}
    problems = list(dict.fromkeys(comparison["problem"]))
    contenders = list(dict.fromkeys(comparison["contender"]))
    level = comparison.assign(level=comparison["a12"].map(effect_level))
    table = level.pivot(index="problem", columns="contender", values="level").reindex(
        index=problems, columns=contenders
    )
    a12 = comparison.pivot(index="problem", columns="contender", values="a12").reindex(
        index=problems, columns=contenders
    )
    verdicts = comparison.pivot(index="problem", columns="contender", values="verdict").reindex(
        index=problems, columns=contenders
    )
    colors = EFFECT_COLORS if scheme == COLOR_SCHEME else EFFECT_GRAYS
    # One flat band of the colorscale per level, from -3 (large, pivot worse) to 3.
    colorscale = []
    for index, color in enumerate(colors):
        colorscale += [[index / 7, color], [(index + 1) / 7, color]]
    figure = go.Figure(
        go.Heatmap(
            z=table.to_numpy(dtype=float),
            x=contenders,
            y=problems,
            customdata=a12.to_numpy(dtype=float),
            colorscale=colorscale,
            zmin=-3.5,
            zmax=3.5,
            xgap=2,
            ygap=2,
            colorbar={
                "title": "Effect",
                "tickvals": list(range(-3, 4)),
                "ticktext": [
                    "large, worse",
                    "medium, worse",
                    "small, worse",
                    "negligible",
                    "small, better",
                    "medium, better",
                    "large, better",
                ],
            },
            hovertemplate="%{y} · %{x}<br>A12 = %{customdata:.3f}<extra></extra>",
        )
    )
    for row, problem in enumerate(problems):
        for column, contender in enumerate(contenders):
            value = a12.iloc[row, column]
            if pd.isna(value):
                continue
            dark = abs(table.iloc[row, column]) == 3 or (
                scheme == GRAY_SCHEME and abs(table.iloc[row, column]) >= 2
            )
            figure.add_annotation(
                x=contender,
                y=problem,
                text=f"{value:.2f} {marks.get(verdicts.iloc[row, column], '')}",
                showarrow=False,
                font={"color": "white" if dark else "black"},
            )
    figure.update_layout(
        title=f"A12 of {pivot} against each algorithm",
        yaxis={"autorange": "reversed"},
        height=max(300, 40 * len(problems) + 140),
        plot_bgcolor="white",
    )
    return figure


def effect_level(value: float) -> int:
    """The level of an A12 value, from -3 (a large effect, the pivot worse) to 3 (large, better)."""
    labels = {"negligible": 0, "small": 1, "medium": 2, "large": 3}
    size = labels[magnitude(value)]
    return size if value > 0.5 else -size


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


def attainment_figure(
    surfaces: dict[str, dict[str, pd.DataFrame]],
    reference_front: pd.DataFrame | None,
    columns: int = GRID_COLUMNS,
) -> go.Figure:
    """The best, median and worst attainment surfaces of each algorithm, a panel each.

    Args:
        surfaces: What `attainment.attainment_surfaces` returns, by algorithm (the pivot first).
        reference_front: The problem's reference front, shown faint in every panel, or None.
        columns: Panels per row.

    Returns:
        The figure, all panels on the same axes; empty without surfaces.
    """
    if not surfaces:
        return go.Figure()
    columns = min(columns, len(surfaces))
    rows = -(-len(surfaces) // columns)
    figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=list(surfaces),
        shared_xaxes="all",
        shared_yaxes="all",
        horizontal_spacing=0.06,
        vertical_spacing=0.12 / rows,
    )
    styles = {
        "best": ("dash", "Best (reached by at least one run)"),
        "median": ("solid", "Median (by half of the runs)"),
        "worst": ("dot", "Worst (by every run)"),
    }
    for position, (label, algorithm_surfaces) in enumerate(surfaces.items()):
        row, column = divmod(position, columns)
        place = {"row": row + 1, "col": column + 1}
        color = COLORS[position % len(COLORS)]
        if reference_front is not None:
            reference = _reference_trace(reference_front, False, faint=True)
            reference.update(showlegend=position == 0, legendgroup="reference")
            figure.add_trace(reference, **place)
        for name, surface in algorithm_surfaces.items():
            dash, legend = styles[name]
            figure.add_trace(
                go.Scatter(
                    x=surface["f1"],
                    y=surface["f2"],
                    mode="lines",
                    line={"color": color, "dash": dash, "shape": "hv", "width": 2},
                    name=legend,
                    legendgroup=name,
                    showlegend=position == 0,
                ),
                **place,
            )
    figure.update_xaxes(title_text="f1", row=rows)
    figure.update_yaxes(title_text="f2", col=1)
    figure.update_layout(height=GRID_2D_ROW_HEIGHT * rows + 100, legend={"orientation": "h"})
    return figure


def eaf_difference_figure(
    differences: pd.DataFrame,
    left: str,
    right: str,
    bounds: pd.DataFrame,
    levels: int = EAF_LEVELS,
) -> go.Figure:
    """Where each of two algorithms reaches more often than the other: two panels, side by side.

    The left panel shades the regions the left algorithm reaches more often, the right panel those
    the right one does; the darker, the larger the difference (in steps of 1 / `levels` of the
    runs). As López-Ibáñez et al.'s EAF difference plots, in grays, so that they print.

    Args:
        differences: What `attainment.eaf_differences` returns.
        left: The left algorithm (the pivot).
        right: The right one.
        bounds: Points (columns f1, f2) that give the extent of the axes, e.g. every front.
        levels: How many shades.

    Returns:
        The figure.
    """
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"{left} reaches more often", f"{right} reaches more often"],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )
    low_x, high_x = float(bounds["f1"].min()), float(bounds["f1"].max())
    low_y, high_y = float(bounds["f2"].min()), float(bounds["f2"].max())
    margin_x, margin_y = 0.05 * (high_x - low_x or 1), 0.05 * (high_y - low_y or 1)
    top_x, top_y = high_x + margin_x, high_y + margin_y
    for column, sign in ((1, 1), (2, -1)):
        side = differences[differences["difference"] * sign > 0]
        shade = np.ceil((side["difference"].abs() * levels).round(9)).clip(1, levels)
        for level in range(1, levels + 1):
            rectangles = side[shade == level]
            if rectangles.empty:
                continue
            xs, ys = [], []
            for rect in rectangles.itertuples():
                x0, x1 = rect.x0, min(rect.x1, top_x)
                y0, y1 = rect.y0, min(rect.y1, top_y)
                xs += [x0, x1, x1, x0, x0, None]
                ys += [y0, y0, y1, y1, y0, None]
            gray = int(235 - 190 * level / levels)
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    fill="toself",
                    mode="none",
                    fillcolor=f"rgb({gray},{gray},{gray})",
                    name=f"{100 * (level - 1) // levels}–{100 * level // levels}% more runs",
                    legendgroup=str(level),
                    showlegend=column == 1,
                    hoverinfo="skip",
                ),
                row=1,
                col=column,
            )
    figure.update_xaxes(range=[low_x - margin_x, top_x], title_text="f1")
    figure.update_yaxes(range=[low_y - margin_y, top_y], title_text="f2", col=1)
    figure.update_layout(height=480, plot_bgcolor="white", legend={"orientation": "h"})
    return figure


def bayesian_simplex_figure(samples: np.ndarray, pivot: str, other: str) -> go.Figure:
    """The posterior of a Bayesian signed-rank test, as points of a triangle (a simplex).

    Each point is a sample of the posterior: its distance to each corner says how likely the
    pivot is to be better, practically equivalent or worse. A cloud by one corner is a clear
    answer; spread over the triangle, the problems are too few, or too different, to tell.

    Args:
        samples: The posterior samples (columns: pivot worse, equivalent, pivot better).
        pivot: The pivot.
        other: The other algorithm.

    Returns:
        The figure.
    """
    shown = samples[:: max(1, len(samples) // SIMPLEX_POINTS)]
    figure = go.Figure(
        go.Scatterternary(
            a=shown[:, 2],
            b=shown[:, 1],
            c=shown[:, 0],
            mode="markers",
            marker={"size": 3, "color": COLORS[0], "opacity": 0.3},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        title=f"{pivot} against {other}",
        ternary={
            "aaxis": {"title": f"{pivot} better"},
            "baxis": {"title": "Equivalent"},
            "caxis": {"title": f"{other} better"},
        },
        height=420,
        showlegend=False,
    )
    return figure


def correlation_figure(correlations: pd.DataFrame, scheme: str = COLOR_SCHEME) -> go.Figure:
    """Spearman's correlation between each pair of indicators, as a heatmap from -1 to 1."""
    names = list(correlations.columns)
    colorscale = (
        [[0, EFFECT_COLORS[0]], [0.5, EFFECT_COLORS[3]], [1, EFFECT_COLORS[6]]]
        if scheme == COLOR_SCHEME
        else [[0, EFFECT_GRAYS[0]], [0.5, EFFECT_GRAYS[3]], [1, EFFECT_GRAYS[6]]]
    )
    figure = go.Figure(
        go.Heatmap(
            z=correlations.to_numpy(dtype=float),
            x=names,
            y=names,
            zmin=-1,
            zmax=1,
            colorscale=colorscale,
            text=correlations.round(2).to_numpy(),
            texttemplate="%{text}",
            colorbar={"title": "ρ"},
            xgap=2,
            ygap=2,
            hovertemplate="%{y} · %{x}<br>ρ = %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        yaxis={"autorange": "reversed"},
        height=max(300, 60 * len(names) + 120),
        plot_bgcolor="white",
    )
    return figure
