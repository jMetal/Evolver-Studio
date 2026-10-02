"""Tests for the Explore › Meta-optimizers page."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio.evolver_client import jar_path

PAGE_SCRIPT = Path(__file__).resolve().parent.parent / "pages" / "explore_meta_optimizers.py"
META_KEY = "explorer_meta_algorithm"


@pytest.fixture
def app() -> AppTest:
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    return AppTest.from_file(str(PAGE_SCRIPT), default_timeout=60).run()


class TestMetaOptimizersPage:
    def test_should_show_nothing_until_a_meta_optimizer_is_chosen(self, app: AppTest):
        # Assert
        assert not app.exception
        assert app.selectbox(key=META_KEY).value is None
        assert len(app.tabs) == 0

    def test_should_show_a_tab_per_encoding_of_a_meta_optimizer_with_both(self, app: AppTest):
        """NSGA-II has an operator catalogue for each encoding, in Evolver 2.1 and later."""
        # Act
        app.selectbox(key=META_KEY).select("NSGA-II").run()

        # Assert
        assert [tab.label for tab in app.tabs] == ["Flat", "Tree"]
        assert len(app.dataframe) == 2

    def test_should_list_the_parameters_of_one_with_operators_hardcoded_in_java(self, app: AppTest):
        """SMPSO has no operator catalogue: only its name list is shown."""
        # Act
        app.selectbox(key=META_KEY).select("SMPSO").run()

        # Assert
        assert [tab.label for tab in app.tabs] == ["Flat"]
        assert len(app.dataframe) == 0
