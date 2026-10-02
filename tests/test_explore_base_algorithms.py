"""Tests for the Explore › Base algorithms page."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio.catalogue import is_older_than_catalogue
from evolver_studio.evolver_client import jar_path
from evolver_studio.resource_files import jar_evolver_version

PAGE_SCRIPT = Path(__file__).resolve().parent.parent / "pages" / "explore_base_algorithms.py"
ALGORITHM_KEY = "explorer_algorithm"
BASE_FILTER_KEY = "explorer_base_filter"


@pytest.fixture
def app() -> AppTest:
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    return AppTest.from_file(str(PAGE_SCRIPT), default_timeout=60).run()


def _parameter_names(app: AppTest) -> list[str]:
    return [name.replace(" ", "").removeprefix("└") for name in app.dataframe[0].value["Parameter"]]


class TestOpening:
    def test_should_open_without_exceptions(self, app: AppTest):
        # Assert
        assert not app.exception

    def test_should_show_no_parameter_space_until_an_algorithm_is_chosen(self, app: AppTest):
        # Assert
        assert app.selectbox(key=ALGORITHM_KEY).value is None
        assert len(app.dataframe) == 0

    def test_should_warn_only_when_the_jar_is_older_than_the_catalogue(self, app: AppTest):
        """Evolver 2.1's parameter spaces lack what the catalogue's Evolver has (e.g. RVEA's)."""
        # Arrange
        version = jar_evolver_version(jar_path())
        expected = version is not None and is_older_than_catalogue(version)

        # Act
        warned = any("the jar in use is Evolver" in warning.value for warning in app.warning)

        # Assert
        assert warned is expected


class TestNSGAIIDouble:
    """NSGA-II's default encoding is Double, whose space (NSGAIIDouble.yaml) has 34 parameters."""

    @pytest.fixture
    def nsgaii(self, app: AppTest) -> AppTest:
        return app.selectbox(key=ALGORITHM_KEY).select("NSGA-II").run()

    def test_should_show_one_row_per_parameter_of_the_selected_space(self, nsgaii: AppTest):
        # Act
        names = _parameter_names(nsgaii)

        # Assert
        assert len(names) == 34
        assert names[0] == "algorithmResult"

    def test_should_say_when_a_conditional_parameter_applies(self, nsgaii: AppTest):
        # Act
        table = nsgaii.dataframe[0].value
        row = table[table["Parameter"].str.endswith("sbxDistributionIndex")].iloc[0]

        # Assert
        assert row["Active if"] == "crossover = SBX"

    def test_should_keep_the_ancestors_of_the_filtered_parameters(self, nsgaii: AppTest):
        # Act
        nsgaii.text_input(key=BASE_FILTER_KEY).input("knnDistanceArchiveK").run()

        # Assert
        assert _parameter_names(nsgaii) == ["algorithmResult", "archiveType", "knnDistanceArchiveK"]

    def test_should_name_the_algorithm_encoding_and_file_above_the_table(self, nsgaii: AppTest):
        # Act
        captions = [caption.value for caption in nsgaii.caption]

        # Assert
        assert "Parameter space of NSGA-II (Double) — NSGAIIDouble.yaml" in captions


class TestSingleEncodingAlgorithm:
    def test_should_not_ask_for_the_encoding_of_an_algorithm_with_only_one(self, app: AppTest):
        """RVEA has only the Double encoding."""
        # Act
        app.selectbox(key=ALGORITHM_KEY).select("RVEA").run()

        # Assert
        assert [box.label for box in app.selectbox] == ["Algorithm"]
        assert len(app.dataframe) == 1
