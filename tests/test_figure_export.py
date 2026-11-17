"""Tests for saving the charts as PNG and PDF."""

import plotly.graph_objects as go
import pytest
from kaleido.errors import ChromeNotFoundError, JavascriptError
from streamlit.testing.v1 import AppTest

from evolver_studio.figure_export import IMAGE_ATTEMPTS, can_export_figures, figure_image


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


class TestFigureImageRetries:
    """Kaleido sometimes fails with a JavaScript error that drawing the chart again avoids."""

    @staticmethod
    def failing_to_image(monkeypatch, errors: list[Exception]) -> list[int]:
        """Make Figure.to_image raise each of `errors` in turn, then return an image."""
        calls: list[int] = []

        def to_image(self, **arguments):
            calls.append(1)
            if len(calls) <= len(errors):
                raise errors[len(calls) - 1]
            return b"image"

        monkeypatch.setattr(go.Figure, "to_image", to_image)
        return calls

    def test_should_draw_again_after_a_javascript_error(self, monkeypatch):
        # Arrange
        calls = self.failing_to_image(monkeypatch, [JavascriptError("not defined")] * 2)

        # Act
        image = figure_image(_figure(), "png")

        # Assert
        assert image == b"image"
        assert len(calls) == 3

    def test_should_give_up_after_the_attempts_are_used(self, monkeypatch):
        # Arrange
        calls = self.failing_to_image(monkeypatch, [JavascriptError("not defined")] * 10)

        # Act / Assert
        with pytest.raises(JavascriptError):
            figure_image(_figure(), "png")
        assert len(calls) == IMAGE_ATTEMPTS

    def test_should_not_retry_other_errors(self, monkeypatch):
        # Arrange
        calls = self.failing_to_image(monkeypatch, [ChromeNotFoundError("no chrome")])

        # Act / Assert
        with pytest.raises(ChromeNotFoundError):
            figure_image(_figure(), "png")
        assert len(calls) == 1


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
