"""Tests for the Explore page."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio.evolver_client import jar_path

EXPLORE_SCRIPT = Path(__file__).resolve().parent.parent / "pages" / "explore.py"
BASE_FILTER_KEY = "explorer_base_filter"


class TestExplorePage:
    @pytest.fixture
    def app(self) -> AppTest:
        if not jar_path().is_file():
            pytest.skip(f"Evolver jar not found at {jar_path()}")
        return AppTest.from_file(str(EXPLORE_SCRIPT), default_timeout=60).run()

    @staticmethod
    def base_table(app: AppTest):
        # The base algorithm's table comes first; the meta-optimizers' follow, in their expanders.
        return app.dataframe[0]

    @staticmethod
    def parameter_names(app: AppTest) -> list[str]:
        table = TestExplorePage.base_table(app)
        return [name.replace(" ", "").removeprefix("└") for name in table.value["Parameter"]]

    def test_should_open_without_exceptions(self, app: AppTest):
        # Assert
        assert not app.exception

    def test_should_show_one_row_per_parameter_of_the_selected_space(self, app: AppTest):
        """NSGAIIDouble.yaml, the default selection, has 34 parameters."""
        # Act
        names = self.parameter_names(app)

        # Assert
        assert len(names) == 34
        assert names[0] == "algorithmResult"

    def test_should_say_when_a_conditional_parameter_applies(self, app: AppTest):
        # Act
        table = self.base_table(app).value
        row = table[table["Parameter"].str.endswith("sbxDistributionIndex")].iloc[0]

        # Assert
        assert row["Active if"] == "crossover = SBX"

    def test_should_keep_the_ancestors_of_the_filtered_parameters(self, app: AppTest):
        # Act
        app.text_input(key=BASE_FILTER_KEY).input("knnDistanceArchiveK").run()

        # Assert
        assert self.parameter_names(app) == [
            "algorithmResult",
            "archiveType",
            "knnDistanceArchiveK",
        ]

    def test_should_name_the_algorithm_encoding_and_file_above_the_table(self, app: AppTest):
        # Act
        captions = [caption.value for caption in app.caption]

        # Assert
        assert "Parameter space of NSGA-II (Double) — NSGAIIDouble.yaml" in captions
