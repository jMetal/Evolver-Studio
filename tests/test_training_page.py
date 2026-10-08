"""Tests for the Training page's checks of the training set's problems."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio import evolver_client
from evolver_studio.evolver_client import WORKING_DIRECTORY, describe, jar_path
from evolver_studio.problem_catalogue import parse_problem_catalogue

PAGE_SCRIPT = Path(__file__).resolve().parent.parent / "pages" / "training.py"


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """The page, with no run in progress (its runs would live in a temporary directory)."""
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    monkeypatch.setattr(evolver_client, "WORKING_DIRECTORY", tmp_path)
    return AppTest.from_file(str(PAGE_SCRIPT), default_timeout=60).run()


def _launch_button(app: AppTest):
    return next(button for button in app.button if button.label == "Launch training")


def _needs_problem_catalogue() -> None:
    result = describe(WORKING_DIRECTORY, jar_path())
    if parse_problem_catalogue(getattr(result, "value", {})) is None:
        pytest.skip("The Evolver jar in use does not describe the problems")


class TestTrainingSetProblems:
    def test_should_open_ready_to_launch_with_zdt4_chosen(self, app: AppTest):
        # Assert: NSGA-II, Double, on ZDT4
        assert not app.exception
        assert not app.error
        assert app.multiselect(key="train_problems_Double").value == ["ZDT4"]
        assert not _launch_button(app).disabled

    def test_should_offer_a_list_with_only_the_problems_of_the_encoding(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Act
        options = list(app.multiselect(key="train_problems_Double").options)

        # Assert
        assert "ZDT4" in options and "DTLZ2" in options
        assert "ZDT5" not in options and "KroAB100TSP" not in options

    def test_should_add_a_row_for_each_problem_chosen(self, app: AppTest):
        # Act
        app.multiselect(key="train_problems_Double").select("DTLZ2").run()

        # Assert
        table = next(d.value for d in app.dataframe if "reference_front" in d.value.columns)
        assert list(table["problem"]) == ["ZDT4", "DTLZ2"]
        assert not app.exception

    def test_should_ask_for_a_problem_when_none_is_chosen(self, app: AppTest):
        # Act
        app.multiselect(key="train_problems_Double").unselect("ZDT4").run()

        # Assert
        assert any("Choose at least one training problem" in e.value for e in app.error)
        assert _launch_button(app).disabled

    def test_should_offer_the_problems_of_another_encoding_when_it_changes(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Act
        app.selectbox(key="train_encoding_NSGA-II").select("Binary").run()

        # Assert: nothing is chosen yet, and the list is that of binary problems
        multiselect = app.multiselect(key="train_problems_Binary")
        assert multiselect.value == []
        assert "ZDT5" in multiselect.options and "ZDT4" not in multiselect.options
        assert _launch_button(app).disabled

    def test_should_refuse_a_problem_of_another_encoding_than_the_algorithms(self, app: AppTest):
        # Arrange: a binary problem typed in, as a class name may be
        _needs_problem_catalogue()
        app.session_state["train_problems_Double"] = ["ZDT4", "ZDT5"]

        # Act
        app.run()

        # Assert
        assert any("ZDT5 is a Binary problem" in error.value for error in app.error)
        assert _launch_button(app).disabled


class TestProblemBrowser:
    """The listing of Explore › Problems, to add problems to the training set."""

    @staticmethod
    def _browser_table(app: AppTest):
        return next(d.value for d in app.dataframe if "Family" in d.value.columns)

    def test_should_keep_the_listing_hidden_until_asked_for(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Assert
        assert not any("Family" in d.value.columns for d in app.dataframe)
        assert app.toggle(key="train_browse_Double").value is False

    def test_should_list_the_problems_of_the_encoding_with_their_details(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Act
        app.toggle(key="train_browse_Double").set_value(True).run()

        # Assert: Double problems only, with their arguments and fronts, and no encoding filter
        table = self._browser_table(app)
        assert set(table["Encoding"]) == {"Double"}
        assert "ZDT4" in set(table["Problem"])
        assert {"Objectives", "Variables", "Arguments", "Reference fronts"} <= set(table.columns)
        assert not any(box.key == "train_browser_Double_encoding" for box in app.selectbox)
        assert not app.exception

    def test_should_filter_the_listing_by_family_and_name(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()
        app.toggle(key="train_browse_Double").set_value(True).run()

        # Act
        app.selectbox(key="train_browser_Double_family").select("DTLZ").run()
        app.text_input(key="train_browser_Double_filter").input("DTLZ2").run()

        # Assert
        assert list(self._browser_table(app)["Problem"]) == ["DTLZ2"]

    def test_should_not_offer_to_add_until_rows_are_selected(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Act
        app.toggle(key="train_browse_Double").set_value(True).run()

        # Assert
        add = app.button(key="train_browser_add_Double")
        assert add.disabled and add.label == "Add the selected problems"


class TestMetaOptimizerEncoding:
    """The selector of flat or tree, before the box with the meta-optimizer's operator flags."""

    @staticmethod
    def _preview(app: AppTest) -> dict:
        import yaml

        meta = next(
            c.value for c in app.code if "algorithm: " in c.value and "encoding: " in c.value
        )
        return yaml.safe_load(meta)

    def test_should_offer_flat_and_tree_for_a_meta_optimizer_that_supports_both(self, app: AppTest):
        # Assert: NSGA-II, the default meta-optimizer
        radio = app.radio(key="train_meta_encoding_NSGA-II")
        assert list(radio.options) == ["Flat", "Tree"]
        assert radio.value == "Flat"
        assert self._preview(app)["encoding"] == "flat"

    def test_should_start_the_operator_flags_from_the_example_of_the_chosen_encoding(
        self, app: AppTest
    ):
        # Act
        app.radio(key="train_meta_encoding_NSGA-II").set_value("Tree").run()

        # Assert: the tree encoding has its own operators, not SBX and polynomial mutation
        flags = app.text_area(key="meta_operator_flags_NSGA-II_tree").value
        assert "mutationDistributionIndex" in flags
        assert "sbxDistributionIndex" not in flags
        assert self._preview(app)["encoding"] == "tree"
        assert not app.exception

    def test_should_offer_only_flat_to_a_meta_optimizer_without_tree_support(self, app: AppTest):
        # Act
        app.selectbox(key="train_meta_algorithm").select("SPEA2").run()

        # Assert
        radio = app.radio(key="train_meta_encoding_SPEA2")
        assert list(radio.options) == ["Flat"]
        assert any("supports only the flat encoding" in c.value for c in app.caption)

    def test_should_not_ask_random_search_for_operator_flags_in_the_tree_encoding(
        self, app: AppTest
    ):
        # Act
        app.selectbox(key="train_meta_algorithm").select("RandomSearch").run()
        app.radio(key="train_meta_encoding_RandomSearch").set_value("Tree").run()

        # Assert: nothing but the scalars is written
        preview = self._preview(app)
        assert preview["encoding"] == "tree"
        assert set(preview) <= {"algorithm", "encoding", "metaMaxEvaluations", "numberOfCores"}
        assert not app.exception


class TestMetaStoppingCondition:
    @staticmethod
    def _meta(app: AppTest) -> dict:
        import yaml

        return yaml.safe_load(
            next(c.value for c in app.code if "algorithm: " in c.value and "encoding: " in c.value)
        )

    def test_should_stop_by_evaluations_by_default(self, app: AppTest):
        # Assert
        assert app.radio(key="train_meta_stop_by").value == "Evaluations"
        assert app.number_input(key="train_meta_evals").value == 2000
        meta = self._meta(app)
        assert meta["metaMaxEvaluations"] == 2000
        assert "metaMaxComputingTimeMinutes" not in meta

    def test_should_stop_by_computing_time_when_chosen(self, app: AppTest):
        # Act
        app.radio(key="train_meta_stop_by").set_value("Computing time").run()
        app.number_input(key="train_meta_minutes").set_value(2.5).run()

        # Assert: only the time is written, as Evolver accepts one limit or the other
        meta = self._meta(app)
        assert meta["metaMaxComputingTimeMinutes"] == 2.5
        assert "metaMaxEvaluations" not in meta
        assert not any(box.key == "train_meta_evals" for box in app.number_input)
        assert not app.exception

    def test_should_combine_with_the_tree_encoding(self, app: AppTest):
        # Act
        app.radio(key="train_meta_encoding_NSGA-II").set_value("Tree").run()
        app.radio(key="train_meta_stop_by").set_value("Computing time").run()

        # Assert
        meta = self._meta(app)
        assert meta["encoding"] == "tree"
        assert meta["metaMaxComputingTimeMinutes"] == 10.0
        assert not _launch_button(app).disabled
