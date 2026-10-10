"""Tests for the charts of the analysis of a validation study."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from evolver_studio.validation_figures import (
    EFFECT_GRAYS,
    GRAY_SCHEME,
    attainment_figure,
    bayesian_simplex_figure,
    correlation_figure,
    critical_difference_figure,
    eaf_difference_figure,
    effect_level,
    effect_size_figure,
    fronts_grid_figure,
    parallel_fronts_figures,
)


def _front(objectives: int, points: int = 5) -> pd.DataFrame:
    values = np.random.default_rng(objectives).random((points, objectives))
    return pd.DataFrame(values, columns=[f"f{i}" for i in range(1, objectives + 1)])


class TestFrontsFigure:
    def test_should_give_the_reference_front_and_then_each_front_a_panel_of_its_own(self):
        # Act
        figure = fronts_grid_figure({"A": _front(2), "B": _front(2)}, _front(2, 20))

        # Assert: panel 1 the reference front; panels 2 and 3 each front over the reference
        titles = [annotation.text for annotation in figure.layout.annotations]
        assert titles == ["Reference front", "A", "B"]
        assert [trace.xaxis for trace in figure.data] == ["x", "x2", "x2", "x3", "x3"]
        assert all(isinstance(trace, go.Scatter) for trace in figure.data)

    def test_should_leave_the_reference_front_out_of_the_fronts_panels_if_asked(self):
        # Act
        figure = fronts_grid_figure(
            {"A": _front(2), "B": _front(2)}, _front(2, 20), reference_behind=False
        )

        # Assert
        assert len(figure.data) == 3

    def test_should_draw_three_objectives_in_3d_panels_on_the_same_ranges(self):
        # Act
        figure = fronts_grid_figure({"A": _front(3), "B": _front(3)}, _front(3, 20))

        # Assert
        assert all(isinstance(trace, go.Scatter3d) for trace in figure.data)
        assert figure.layout.scene.xaxis.range == figure.layout.scene3.xaxis.range

    def test_should_give_a_parallel_coordinates_chart_per_algorithm_for_more(self):
        # Act
        figures = parallel_fronts_figures({"A": _front(5), "B": _front(5)}, _front(5, 20))

        # Assert: the reference front first; then each front over it, on one scale
        assert len(figures) == 3
        assert figures[0].layout.title.text == "Reference front"
        chart = figures[1].data[0]
        assert isinstance(chart, go.Parcoords)
        assert len(chart.dimensions) == 5
        assert len(chart.dimensions[0].values) == 25
        assert figures[1].data[0].dimensions[0].range == figures[2].data[0].dimensions[0].range

    def test_should_be_empty_without_fronts(self):
        # Act / Assert
        assert len(fronts_grid_figure({}, None).data) == 0


class TestOtherFigures:
    COMPARISON = pd.DataFrame(
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

    def test_should_shade_the_a12_by_the_size_of_the_effect(self):
        # Act
        heatmap = effect_size_figure(self.COMPARISON, "tuned").data[0]

        # Assert: large better, negligible; large worse, small better
        assert heatmap.z.tolist() == [[3, 0], [-3, 1]]
        assert list(heatmap.y) == ["P1", "P2"]

    def test_should_write_the_a12_and_the_mark_in_each_cell(self):
        # Act
        figure = effect_size_figure(self.COMPARISON, "tuned")

        # Assert
        assert [a.text for a in figure.layout.annotations] == [
            "0.90 +",
            "0.50 =",
            "0.20 -",
            "0.60 =",
        ]

    def test_should_offer_grays_that_print_in_black_and_white(self):
        # Act
        colorscale = effect_size_figure(self.COMPARISON, "tuned", GRAY_SCHEME).data[0].colorscale

        # Assert: the two sides share each gray
        colors = [color for _, color in colorscale]
        assert set(colors) == set(EFFECT_GRAYS)

    @pytest.mark.parametrize(
        ("value", "level"), [(0.5, 0), (0.55, 0), (0.6, 1), (0.66, 2), (0.9, 3), (0.1, -3)]
    )
    def test_should_classify_the_effect_as_vargha_and_delaney(self, value, level):
        # Act / Assert
        assert effect_level(value) == level

    def test_should_draw_the_attainment_surfaces_of_each_algorithm_a_panel_each(self):
        # Arrange
        surfaces = {
            name: {"best": _front(2), "median": _front(2), "worst": _front(2)} for name in "AB"
        }

        # Act
        figure = attainment_figure(surfaces, _front(2, 20))

        # Assert: per panel, the faint reference front and three surfaces
        assert [a.text for a in figure.layout.annotations] == ["A", "B"]
        assert len(figure.data) == 8

    def test_should_shade_each_side_of_the_eaf_differences(self):
        # Arrange
        differences = pd.DataFrame(
            {
                "x0": [0.0, 0.5],
                "y0": [0.0, 0.5],
                "x1": [1.0, np.inf],
                "y1": [1.0, np.inf],
                "difference": [0.8, -0.2],
            }
        )

        # Act
        figure = eaf_difference_figure(differences, "tuned", "B", _front(2))

        # Assert: one shade on the left panel, one on the right, infinite corners clipped
        assert [trace.xaxis for trace in figure.data] == ["x", "x2"]
        assert all(
            np.isfinite([v for v in trace.x if v is not None]).all() for trace in figure.data
        )

    def test_should_draw_the_posterior_in_a_triangle(self):
        # Act
        figure = bayesian_simplex_figure(np.array([[0.1, 0.2, 0.7]] * 10), "tuned", "B")

        # Assert
        assert list(figure.data[0].a) == [0.7] * 10

    def test_should_draw_the_correlations_from_minus_one_to_one(self):
        # Act
        heatmap = correlation_figure(
            pd.DataFrame([[1, -0.5], [-0.5, 1]], index=["A", "B"], columns=["A", "B"])
        ).data[0]

        # Assert
        assert (heatmap.zmin, heatmap.zmax) == (-1, 1)

    def test_should_draw_a_bar_per_group_of_the_critical_difference_plot(self):
        # Arrange
        ranks = pd.Series({"A": 1.0, "B": 1.5, "C": 3.5})

        # Act
        figure = critical_difference_figure(ranks, 1.0, [["A", "B"]])

        # Assert: one thick bar among the shapes, and a label per algorithm
        assert sum(1 for shape in figure.layout.shapes if shape.line.width == 5) == 1
        labels = [a.text for a in figure.layout.annotations]
        assert {"A (1.00)", "B (1.50)", "C (3.50)"} <= set(labels)
