"""Tests for the figures of a solve run's fronts."""

import pandas as pd

from evolver_studio.solve_figures import build_front_figure


def _front(objectives: int, points: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"f{i}": [round(0.1 * (j + i), 2) for j in range(points)]
            for i in range(1, objectives + 1)
        }
    )


class TestBuildFrontFigure:
    def test_should_draw_a_2d_scatter_per_run_under_the_reference_front(self):
        # Act
        figure = build_front_figure({1: _front(2), 2: _front(2)}, _front(2, 10))

        # Assert: the reference front first, so it is drawn behind
        assert [trace.name for trace in figure.data] == ["Reference front", "1", "2"]
        assert figure.data[0].type == "scatter"

    def test_should_draw_a_3d_scatter_with_the_reference_front(self):
        # Act
        figure = build_front_figure({1: _front(3)}, _front(3, 10))

        # Assert
        assert {trace.type for trace in figure.data} == {"scatter3d"}
        assert "Reference front" in [trace.name for trace in figure.data]

    def test_should_draw_parallel_coordinates_for_many_objectives(self):
        # Act
        figure = build_front_figure({1: _front(5), 2: _front(5)})

        # Assert
        assert [trace.type for trace in figure.data] == ["parcoords"]
        assert len(figure.data[0].dimensions) == 5

    def test_should_draw_without_a_reference_front(self):
        # Act
        figure = build_front_figure({1: _front(2)})

        # Assert
        assert [trace.name for trace in figure.data] == ["1"]

    def test_should_return_an_empty_figure_when_there_are_no_fronts(self):
        # Act / Assert
        assert len(build_front_figure({}).data) == 0
