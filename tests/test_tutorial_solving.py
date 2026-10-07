"""Tests for tutorial S2, "Solving a problem with a configurable algorithm"."""

from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from evolver_studio import evolver_client, tutorial_solving
from evolver_studio.catalogue import BASE_ALGORITHMS
from evolver_studio.evolver_client import jar_path
from evolver_studio.solve_form import form_state_from_request
from evolver_studio.solve_request import solve_request_to_yaml
from evolver_studio.tutorial_solving import (
    DEFAULT_ON_ZDT1,
    DEFAULT_ON_ZDT2_FIVE_RUNS,
    MOEAD_ON_DTLZ2,
    OTHER_ON_ZDT1,
    RVEA_VARIANTS,
    prepared_request,
)

APP_SCRIPT = Path(__file__).resolve().parent.parent / "app.py"
RESOURCES = Path(__file__).resolve().parent.parent / "resources"
PREPARED_RUNS = [
    DEFAULT_ON_ZDT1,
    OTHER_ON_ZDT1,
    DEFAULT_ON_ZDT2_FIVE_RUNS,
    MOEAD_ON_DTLZ2,
    *RVEA_VARIANTS.values(),
]


@pytest.fixture
def jar() -> Path:
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    return jar_path()


class TestPreparedRuns:
    @pytest.mark.parametrize("prepared", PREPARED_RUNS, ids=lambda run: run.key)
    def test_should_make_a_valid_request_with_a_fixed_seed_and_the_two_indicators(
        self, prepared, jar: Path
    ):
        # Act
        request = prepared_request(prepared, jar, "out")

        # Assert
        assert request.validation_errors() == []
        assert request.seed == 1
        assert request.indicator_names == ["Epsilon", "NormalizedHypervolume"]
        assert "\n" not in request.configuration

    @pytest.mark.parametrize("prepared", PREPARED_RUNS, ids=lambda run: run.key)
    def test_should_be_restorable_in_the_run_algorithm_form(self, prepared, jar: Path):
        """The button 'Open in Run algorithm' relies on it."""
        # Arrange
        request = yaml.safe_load(solve_request_to_yaml(prepared_request(prepared, jar, "out")))

        # Act
        state = form_state_from_request(request, BASE_ALGORITHMS, [prepared.problem], 0)

        # Assert
        assert state is not None

    def test_should_ask_the_decomposition_algorithms_for_their_weight_vectors(self, jar: Path):
        # Act
        moead = prepared_request(MOEAD_ON_DTLZ2, jar, "out")
        nsgaii = prepared_request(DEFAULT_ON_ZDT1, jar, "out")

        # Assert
        assert moead.extra_config == {"weightVectorFilesDirectory": "resources/weightVectors"}
        assert nsgaii.extra_config is None

    def test_should_read_the_default_configuration_of_each_variant_of_rvea(self, jar: Path):
        # Act
        configurations = {
            name: prepared_request(prepared, jar, "out").configuration
            for name, prepared in RVEA_VARIANTS.items()
        }

        # Assert
        assert "--replacement rvea " in configurations["RVEA"] + " "
        assert "--replacement rveaStar" in configurations["RVEA*"]
        assert "--replacement iRVEA" in configurations["iRVEA"]


class TestTutorialS2WalkThrough:
    """Runs the real app through S2, with its runs in a temporary working directory."""

    @pytest.fixture
    def app(self, jar: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
        (tmp_path / "resources").symlink_to(RESOURCES)
        monkeypatch.setattr(evolver_client, "WORKING_DIRECTORY", tmp_path)
        app = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
        app.switch_page("pages/tutorials.py").run()
        app.button(key="tutorial_open_solving").click().run()
        return app

    @staticmethod
    def _go_to_step(app: AppTest, title: str) -> None:
        while app.subheader[0].value != title:
            next(button for button in app.button if button.label == "Next →").click().run()

    @staticmethod
    def _run(app: AppTest, key: str) -> None:
        app.button(key=f"tutorial_solving_button_{key}").click().run()

    def test_should_render_every_step_without_exceptions(self, app: AppTest):
        # Act
        subheaders = []
        for _ in tutorial_solving.STEPS:
            subheaders.append(app.subheader[0].value)
            assert not app.exception
            next_button = next(button for button in app.button if button.label == "Next →")
            if not next_button.disabled:
                next_button.click().run()

        # Assert
        assert subheaders == [step.title for step in tutorial_solving.STEPS]

    def test_should_run_the_first_step_and_get_the_values_the_text_announces(self, app: AppTest):
        # Arrange
        self._go_to_step(app, "Step 1: the default NSGA-II on ZDT1")

        # Act
        self._run(app, "default")

        # Assert
        assert not app.exception
        assert any("are the ones you got" in success.value for success in app.success)
        assert len(app.get("plotly_chart")) == 1

    def test_should_check_the_answer_about_the_crossover(self, app: AppTest):
        # Arrange
        self._go_to_step(app, "Step 1: the default NSGA-II on ZDT1")

        # Act
        app.radio(key="tutorial_solving_crossover_quiz").set_value("SBX").run()

        # Assert
        assert any("Right" in success.value for success in app.success)

    def test_should_list_the_files_and_filter_the_solutions_of_the_first_run(self, app: AppTest):
        # Arrange
        self._go_to_step(app, "Step 1: the default NSGA-II on ZDT1")
        self._run(app, "default")
        next(button for button in app.button if button.label == "Next →").click().run()

        # Act
        slider = app.slider(key="tutorial_solving_f1_range")
        low, high = slider.value
        slider.set_value((low, (low + high) / 2)).run()

        # Assert
        assert any("run-1/FUN.csv" in write.value for write in app.markdown)
        assert any("solutions." in caption.value for caption in app.caption)

    def test_should_compare_the_two_configurations_once_both_have_run(self, app: AppTest):
        # Arrange
        self._go_to_step(app, "Step 1: the default NSGA-II on ZDT1")
        self._run(app, "default")
        self._go_to_step(app, "Step 3: a different configuration")

        # Act
        self._run(app, "other")

        # Assert
        assert not app.exception
        assert any("are the ones you got" in success.value for success in app.success)
        comparison = [frame for frame in app.dataframe if "" in frame.value.columns]
        assert comparison
        assert list(comparison[0].value[""]) == ["Standard configuration", "Other configuration"]

    def test_should_run_five_runs_and_summarize_them(self, app: AppTest):
        # Arrange
        self._go_to_step(app, "Step 4: more runs")

        # Act
        self._run(app, "runs")

        # Assert
        assert not app.exception
        assert any("are the ones you got" in success.value for success in app.success)
        assert [len(frame.value) for frame in app.dataframe] == [5, 3]

    def test_should_run_moead_and_the_variant_of_rvea_chosen(self, app: AppTest):
        # Arrange
        self._go_to_step(app, "Step 5: more objectives, other algorithms")

        # Act
        self._run(app, "moead")
        app.radio(key="tutorial_solving_rvea_variant").set_value("iRVEA").run()
        self._run(app, "irvea")

        # Assert
        assert not app.exception
        assert len([s for s in app.success if "are the ones you got" in s.value]) == 2

    def test_should_go_back_to_the_list_of_tutorials_from_the_last_step(self, app: AppTest):
        """A link to the Tutorials page does nothing from inside it, so it is a button."""
        # Arrange
        self._go_to_step(app, "Try it yourself")

        # Act
        app.button(key="tutorial_solving_all_tutorials").click().run()

        # Assert
        assert not app.exception
        assert [header.value for header in app.subheader][:1] == ["Introductory"]
        assert len([button for button in app.button if button.label == "Start"]) == 2
