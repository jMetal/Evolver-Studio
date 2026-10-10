"""Tests for the charts of the analysis of a validation study."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from evolver_studio.validation_figures import (
    critical_difference_figure,
    effect_size_figure,
    fronts_figure,
    parallel_fronts_figures,
)


def _front(objectives: int, points: int = 5) -> pd.DataFrame:
    values = np.random.default_rng(objectives).random((points, objectives))
    return pd.DataFrame(values, columns=[f"f{i}" for i in range(1, objectives + 1)])


class TestFrontsFigure:
    def test_should_overlay_two_objective_fronts_over_the_reference_front(self):
        # Act
        figure = fronts_figure({"A": _front(2), "B": _front(2)}, _front(2, 20))

        # Assert
        assert [trace.name for trace in figure.data] == ["Reference front", "A", "B"]
        assert all(isinstance(trace, go.Scatter) for trace in figure.data)

    def test_should_overlay_three_objective_fronts_in_3d(self):
        # Act
        figure = fronts_figure({"A": _front(3), "B": _front(3)}, _front(3, 20))

        # Assert
        assert all(isinstance(trace, go.Scatter3d) for trace in figure.data)
        assert len(figure.data) == 3

    def test_should_give_a_parallel_coordinates_chart_per_algorithm_for_more(self):
        # Act
        figures = parallel_fronts_figures({"A": _front(5), "B": _front(5)}, _front(5, 20))

        # Assert: each holds the reference front and the algorithm's front, on one scale
        assert len(figures) == 2
        chart = figures[0].data[0]
        assert isinstance(chart, go.Parcoords)
        assert len(chart.dimensions) == 5
        assert len(chart.dimensions[0].values) == 25
        assert figures[0].data[0].dimensions[0].range == figures[1].data[0].dimensions[0].range

    def test_should_be_empty_without_fronts(self):
        # Act / Assert
        assert len(fronts_figure({}, None).data) == 0


class TestOtherFigures:
    def test_should_draw_the_a12_of_each_problem_and_algorithm(self):
        # Arrange
        comparison = pd.DataFrame(
            {
                "problem": ["P1", "P1", "P2", "P2"],
                "contender": ["A", "B", "A", "B"],
                "a12": [0.9, 0.5, 0.2, 0.6],
                "verdict": [
                    "pivot better",
                    "no significant difference",
                    "pivot worse",
                    "no significant difference",
                ],
            }
        )

        # Act
        heatmap = effect_size_figure(comparison, "tuned").data[0]

        # Assert
        assert list(heatmap.y) == ["P1", "P2"]
        assert heatmap.text[1][0] == "0.20 -"

    def test_should_draw_a_bar_per_group_of_the_critical_difference_plot(self):
        # Arrange
        ranks = pd.Series({"A": 1.0, "B": 1.5, "C": 3.5})

        # Act
        figure = critical_difference_figure(ranks, 1.0, [["A", "B"]])

        # Assert: one thick bar among the shapes, and a label per algorithm
        assert sum(1 for shape in figure.layout.shapes if shape.line.width == 5) == 1
        labels = [a.text for a in figure.layout.annotations]
        assert {"A (1.00)", "B (1.50)", "C (3.50)"} <= set(labels)
