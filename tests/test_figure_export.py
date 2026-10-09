"""Tests for saving the charts as PNG and PDF."""

import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio.figure_export import can_export_figures, figure_image


@pytest.fixture
def exporting() -> None:
    if not can_export_figures():
        pytest.skip("Kaleido cannot draw images here (no Kaleido or no Chrome)")


def _figure() -> go.Figure:
    return go.Figure(go.Scatter(x=[1, 2, 3], y=[3, 1, 2]))


class TestFigureImage:
    def test_should_draw_a_png(self, exporting):
        # Act
        image = figure_image(_figure(), "png")

        # Assert
        assert image.startswith(b"\x89PNG")

    def test_should_draw_a_pdf(self, exporting):
        # Act
        image = figure_image(_figure(), "pdf")

        # Assert
        assert image.startswith(b"%PDF")


def _page() -> None:
    import plotly.graph_objects as go

    from evolver_studio.figure_export import render_chart

    render_chart(go.Figure(go.Bar(x=["a"], y=[1])), "chart", "my_chart", width="stretch")


class TestRenderChart:
    def test_should_offer_to_save_the_chart_as_png_and_pdf(self, exporting):
        # Act
        app = AppTest.from_function(_page, default_timeout=60).run()

        # Assert
        assert not app.exception
        labels = [button.proto.label for button in app.get("download_button")]
        assert labels == ["PNG", "PDF"]
