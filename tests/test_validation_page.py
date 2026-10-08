"""Tests for the Validation page, running real studies in a temporary working directory."""

import time
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from evolver_studio import evolver_client
from evolver_studio.evolver_client import WORKING_DIRECTORY, describe, jar_path
from evolver_studio.problem_catalogue import parse_problem_catalogue
from evolver_studio.resource_files import default_configuration_text
from evolver_studio.runs import RunPhase, run_phase

PAGE_SCRIPT = Path(__file__).resolve().parent.parent / "pages" / "validation.py"
RESOURCES = Path(__file__).resolve().parent.parent / "resources"
RUN_TIMEOUT_SECONDS = 120


def _jar_describes_problems() -> bool:
    result = describe(WORKING_DIRECTORY, jar_path())
    return parse_problem_catalogue(getattr(result, "value", {})) is not None


@pytest.fixture
def working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory that shares the resources, which the page and its worker run in."""
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    (tmp_path / "resources").symlink_to(RESOURCES)
    monkeypatch.setattr(evolver_client, "WORKING_DIRECTORY", tmp_path)
    return tmp_path


@pytest.fixture
def app(working_directory: Path) -> AppTest:
    return AppTest.from_file(str(PAGE_SCRIPT), default_timeout=60).run()


def _needs_problem_catalogue() -> None:
    if not _jar_describes_problems():
        pytest.skip("The Evolver jar in use does not describe the problems")


def _finished_training(working_directory: Path) -> None:
    """A finished training run of NSGA-II (Double) with two configurations."""
    default = default_configuration_text(jar_path(), "NSGAIIDoubleDefault.txt").splitlines()[0]
    run_dir = working_directory / "cli-runner-runs" / "20260101-000000"
    run_dir.mkdir(parents=True)
    (run_dir / "status.yaml").write_text(
        yaml.safe_dump(
            {"status": "FINISHED", "evaluationsDone": 1, "maxEvaluations": 1, "updatedAt": "x"}
        )
    )
    (run_dir / "request.yaml").write_text(yaml.safe_dump({"outputDirectory": "results/t"}))
    (run_dir / "base_level.yaml").write_text(
        yaml.safe_dump(
            {"algorithmName": "NSGA-II", "encoding": "Double", "trainingProblemNames": ["ZDT4"]}
        )
    )
    (working_directory / "results" / "t").mkdir(parents=True)
    (working_directory / "results" / "t" / "VAR_CONF.txt").write_text(
        f"# Evaluation: 100\nEP=0.1 NHV=0.2 | {default}\nEP=0.3 NHV=0.4 | {default}\n"
    )


class TestOpening:
    def test_should_open_without_exceptions(self, app: AppTest):
        # Assert
        assert not app.exception

    def test_should_say_it_needs_a_newer_evolver_when_the_jar_has_no_problem_catalogue(
        self, app: AppTest
    ):
        # Act
        needs_newer = any("2.4 or later" in info.value for info in app.info)

        # Assert
        assert needs_newer is not _jar_describes_problems()

    def test_should_offer_the_problems_of_the_chosen_encoding(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Act
        app.selectbox(key="validation_encoding").select("Binary").run()

        # Assert
        options = list(app.multiselect(key="validation_problems_Binary").options)
        assert "ZDT5" in options and "ZDT1" not in options

    def test_should_offer_the_algorithms_with_a_default_for_the_encoding(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Act
        app.selectbox(key="validation_encoding").select("Permutation").run()

        # Assert
        defaults = app.multiselect(key="validation_defaults_Permutation")
        assert list(defaults.options) == ["NSGA-II", "PAES"]
        assert defaults.value == ["NSGA-II"]

    def test_should_not_let_a_study_run_without_problems_or_a_tuned_configuration(
        self, app: AppTest
    ):
        # Arrange
        _needs_problem_catalogue()

        # Assert
        assert next(b for b in app.button if b.label == "Run the study").disabled
        assert any("Choose at least one problem" in w.value for w in app.warning)


class TestAdjustingAnAlgorithm:
    @staticmethod
    def _crossover_probability(app: AppTest):
        return next(
            widget
            for widget in app.number_input
            if widget.key.startswith("validation_adjusted_Double_NSGA-II_")
            and widget.key.endswith("_crossoverProbability")
        )

    def test_should_start_from_the_default_configuration(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Assert: the form is there, and nothing is marked as adjusted
        assert self._crossover_probability(app).value is not None
        assert not any("(adjusted)" in caption.value for caption in app.caption)

    def test_should_compare_with_the_adjusted_configuration_under_its_own_name(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Act
        self._crossover_probability(app).set_value(0.5).run()

        # Assert
        assert not app.exception
        assert any(
            "NSGA-II (adjusted)" in caption.value and "crossoverProbability" in caption.value
            for caption in app.caption
        )

    def test_should_go_back_to_the_default_configuration(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()
        self._crossover_probability(app).set_value(0.5).run()

        # Act
        next(
            b
            for b in app.button
            if b.label == "Reset to the default configuration"
            and b.key.startswith("validation_adjusted_Double_NSGA-II")
        ).click().run()

        # Assert
        assert not any("(adjusted)" in caption.value for caption in app.caption)


class TestTunedConfiguration:
    def test_should_offer_a_configuration_a_training_run_found(
        self, app: AppTest, working_directory: Path
    ):
        # Arrange
        _needs_problem_catalogue()
        _finished_training(working_directory)

        # Act
        app = AppTest.from_file(str(PAGE_SCRIPT), default_timeout=60).run()

        # Assert
        options = app.selectbox(key="validation_configuration_20260101-000000").options
        assert len(options) == 2
        assert "EP=0.1 · NHV=0.2" in options[0]

    def test_should_reject_a_pasted_value_the_space_does_not_allow(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()
        app.radio(key="validation_tuned_source").set_value("A configuration I paste").run()

        # Act
        app.text_area(key="validation_pasted_configuration").input(
            "--algorithmResult nonsense"
        ).run()

        # Assert
        assert any("does not allow the value of algorithmResult" in e.value for e in app.error)


class TestRunAStudy:
    def test_should_run_a_study_and_show_its_comparison(
        self, app: AppTest, working_directory: Path
    ):
        # Arrange: ZDT1 and DTLZ2, the tuned NSGA-II (from a training run) against the default
        # NSGA-II and MOEA/D, which with a population of 91 has a weight vector file for the
        # three objectives of DTLZ2 but not for the two of ZDT1
        _needs_problem_catalogue()
        _finished_training(working_directory)
        app = AppTest.from_file(str(PAGE_SCRIPT), default_timeout=60).run()
        app.multiselect(key="validation_problems_Double").select("ZDT1").select("DTLZ2").run()
        app.multiselect(key="validation_defaults_Double").select("MOEA/D").run()
        app.number_input(key="validation_population").set_value(91)
        app.number_input(key="validation_evaluations").set_value(2000)
        app.number_input(key="validation_runs").set_value(4)
        app.number_input(key="validation_processes").set_value(3).run()
        assert not app.exception
        assert any("MOEA/D is left out of ZDT1" in w.value for w in app.warning)

        # Act
        next(b for b in app.button if b.label == "Run the study").click().run()
        study_dir = next((working_directory / "validation-runs").iterdir())
        deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
        while run_phase(study_dir) in (RunPhase.STARTING, RunPhase.RUNNING):
            assert time.monotonic() < deadline, "the study did not finish"
            time.sleep(0.3)
        app.run()
        app.selectbox(key="validation_history").select(study_dir.name).run()

        # Assert
        assert not app.exception
        assert run_phase(study_dir) is RunPhase.FINISHED
        assert [tab.label for tab in app.tabs][-5:] == [
            "Summary",
            "Comparison",
            "Boxplots",
            "Runs",
            "Details",
        ]
        medians = next(d for d in app.dataframe if "NSGA-II (tuned)" in d.value.columns)
        assert list(medians.value.index) == ["ZDT1", "DTLZ2"]
        # 2 contenders on ZDT1, 3 on DTLZ2, 4 runs each
        assert len(next(d for d in app.dataframe if "Seed" in d.value.columns).value) == 20


class TestProblemBrowser:
    """The listing of Explore › Problems, to add problems to the study."""

    @staticmethod
    def _browser_table(app: AppTest):
        return next(d.value for d in app.dataframe if "Family" in d.value.columns)

    def test_should_keep_the_listing_hidden_until_asked_for(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()

        # Assert
        assert app.toggle(key="validation_browse_Double").value is False
        assert not any("Family" in d.value.columns for d in app.dataframe)

    def test_should_list_the_problems_of_the_encoding_the_study_uses(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()
        app.selectbox(key="validation_encoding").select("Binary").run()

        # Act
        app.toggle(key="validation_browse_Binary").set_value(True).run()

        # Assert: binary problems only, and no encoding filter: the study fixes it
        table = self._browser_table(app)
        assert set(table["Encoding"]) == {"Binary"}
        assert "ZDT5" in set(table["Problem"])
        assert not any(box.key == "validation_browser_Binary_encoding" for box in app.selectbox)
        assert not app.exception

    def test_should_filter_the_listing_by_name(self, app: AppTest):
        # Arrange
        _needs_problem_catalogue()
        app.toggle(key="validation_browse_Double").set_value(True).run()

        # Act
        app.text_input(key="validation_browser_Double_filter").input("WFG").run()

        # Assert
        names = list(self._browser_table(app)["Problem"])
        assert names and all("WFG" in name for name in names)
