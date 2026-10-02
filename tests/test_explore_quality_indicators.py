"""Tests for the Explore › Quality indicators page."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from evolver_studio.catalogue import QUALITY_INDICATORS

PAGE_SCRIPT = Path(__file__).resolve().parent.parent / "pages" / "explore_quality_indicators.py"


class TestQualityIndicatorsPage:
    def test_should_show_one_row_per_indicator_of_the_catalogue(self):
        # Arrange
        app = AppTest.from_file(str(PAGE_SCRIPT), default_timeout=30)

        # Act
        app.run()

        # Assert
        assert not app.exception
        table = next(m.value for m in app.markdown if "| Name in a request |" in m.value)
        for indicator in QUALITY_INDICATORS:
            assert f"| `{indicator.registry_name}` |" in table
