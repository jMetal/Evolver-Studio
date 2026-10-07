"""Tests for the Explore › Problems page."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio.evolver_client import WORKING_DIRECTORY, describe, jar_path
from evolver_studio.problem_catalogue import parse_problem_catalogue
from evolver_studio.result import Ok

PAGE_SCRIPT = Path(__file__).resolve().parent.parent / "pages" / "explore_problems.py"
ENCODING_KEY = "explore_problems_encoding"
FAMILY_KEY = "explore_problems_family"
FILTER_KEY = "explore_problems_filter"


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    result = describe(WORKING_DIRECTORY, jar_path())
    assert isinstance(result, Ok)
    return result.value


@pytest.fixture
def app(manifest: dict) -> AppTest:
    return AppTest.from_file(str(PAGE_SCRIPT), default_timeout=60).run()


def _shown(app: AppTest) -> list[str]:
    return list(app.dataframe[0].value["Problem"])


class TestOpening:
    def test_should_open_without_exceptions(self, app: AppTest):
        # Assert
        assert not app.exception

    def test_should_list_every_registered_problem(self, app: AppTest, manifest: dict):
        # Assert
        assert sorted(_shown(app)) == sorted(manifest["problems"])

    def test_should_describe_the_problems_when_the_jar_has_a_catalogue(
        self, app: AppTest, manifest: dict
    ):
        """Evolver 2.3's manifest has no catalogue: the page then lists the names alone."""
        # Act
        columns = list(app.dataframe[0].value.columns)

        # Assert
        if parse_problem_catalogue(manifest) is None:
            assert columns == ["Problem"]
        else:
            assert "Encoding" in columns and "Arguments" in columns


class TestFilters:
    @pytest.fixture
    def described(self, app: AppTest, manifest: dict) -> AppTest:
        if parse_problem_catalogue(manifest) is None:
            pytest.skip("The Evolver jar in use has no problem catalogue")
        return app

    def test_should_keep_only_the_problems_of_the_chosen_encoding(self, described: AppTest):
        # Act
        app = described.selectbox(key=ENCODING_KEY).select("Binary").run()

        # Assert
        assert "ZDT5" in _shown(app)
        assert set(app.dataframe[0].value["Encoding"]) == {"Binary"}
        assert any("Algorithms that solve Binary problems" in c.value for c in app.caption)

    def test_should_keep_only_the_problems_of_the_chosen_family(self, described: AppTest):
        # Act
        app = described.selectbox(key=FAMILY_KEY).select("DTLZ").run()

        # Assert
        assert set(app.dataframe[0].value["Family"]) == {"DTLZ"}

    def test_should_filter_by_name_ignoring_case(self, described: AppTest):
        # Act
        app = described.text_input(key=FILTER_KEY).input("kroab").run()

        # Assert
        assert _shown(app) and all("kroab" in name.lower() for name in _shown(app))

    def test_should_show_the_arguments_and_reference_fronts_of_a_problem(self, described: AppTest):
        # Act
        app = described.text_input(key=FILTER_KEY).input("DTLZ2").run()
        table = app.dataframe[0].value.set_index("Problem")

        # Assert
        assert table.loc["DTLZ2", "Arguments"] == "numberOfVariables=12, numberOfObjectives=3"
        assert "DTLZ2.3D.csv" in table.loc["DTLZ2", "Reference fronts"]
