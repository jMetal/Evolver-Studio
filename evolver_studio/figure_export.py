"""Charts that can be saved: every Plotly chart of the app, with buttons to save it as PNG or PDF.

The images are made by Kaleido, which draws the chart in the Chrome (or Chromium) installed on the
computer, and only when a button is pressed: the monitor of a training redraws its charts every few
seconds, and making the images there would slow it down. Without Kaleido or a Chrome, the buttons
are not offered, and the camera of the chart's toolbar still saves it as PNG.
"""

from collections.abc import Callable

import plotly.graph_objects as go
import streamlit as st

PNG_SCALE = 2
# The size of a chart that does not set its own, in pixels.
DEFAULT_WIDTH = 1000
DEFAULT_HEIGHT = 600


def figure_image(figure: go.Figure, image_format: str) -> bytes:
    """Draw a chart as an image.

    Args:
        figure: The chart.
        image_format: "png" or "pdf".

    Returns:
        The image.
    """
    return figure.to_image(
        format=image_format,
        width=figure.layout.width or DEFAULT_WIDTH,
        height=figure.layout.height or DEFAULT_HEIGHT,
        scale=PNG_SCALE if image_format == "png" else 1,
    )


@st.cache_resource(show_spinner=False)
def can_export_figures() -> bool:
    """Whether Kaleido can draw images: it is installed and finds a Chrome. Tried once."""
    try:
        figure_image(go.Figure(), "png")
    except Exception:  # noqa: BLE001 - any failure means the images cannot be made
        return False
    return True


def render_chart(figure: go.Figure, key: str, file_name: str, **plotly_chart_arguments) -> None:
    """Draw a chart, with buttons to save it as PNG or PDF.

    Args:
        figure: The chart.
        key: The chart's widget key; the buttons' keys derive from it.
        file_name: The name of the files saved, without the extension.
        **plotly_chart_arguments: What else `st.plotly_chart` takes (e.g. width="stretch").
    """
    st.plotly_chart(figure, key=key, **plotly_chart_arguments)
    if not can_export_figures():
        return
    png_column, pdf_column, _ = st.columns([1, 1, 6])
    for column, image_format in ((png_column, "png"), (pdf_column, "pdf")):
        column.download_button(
            image_format.upper(),
            _drawer(figure, image_format),
            file_name=f"{file_name}.{image_format}",
            mime="image/png" if image_format == "png" else "application/pdf",
            key=f"{key}_save_{image_format}",
            on_click="ignore",
            type="tertiary",
            icon=":material/download:",
            help=f"Save this chart as {image_format.upper()}.",
        )


def _drawer(figure: go.Figure, image_format: str) -> Callable[[], bytes]:
    """What makes the image when its button is pressed (a download button calls it then)."""
    return lambda: figure_image(figure, image_format)
