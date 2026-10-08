"""Tests for tutorial "Analyzing training results", on the training run that ships with the app."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio import tutorial_analysis
from evolver_studio.evolver_client import jar_path

APP_SCRIPT = Path(__file__).resolve().parent.parent / "app.py"
TITLE = "Analyzing training results"


class TestTrainingOfTheTutorial:
    def test_should_ship_a_finished_training_with_its_population(self):
        # Act
        training = tutorial_analysis.tutorial_training()

        # Assert
        assert (training.algorithm, training.encoding) == ("NSGA-II", "Double")
        assert training.has_population
        assert training.configurations
        assert training.outcome is not None


class TestWalkThrough:
    @pytest.fixture
    def app(self) -> AppTest:
        if not jar_path().is_file():
            pytest.skip(f"Evolver jar not found at {jar_path()}")
        app = AppTest.from_file(str(APP_SCRIPT), default_timeout=60).run()
        app.switch_page("pages/tutorials.py").run()
        key = "tutorial_open_analyzing_training_results"
        next(button for button in app.button if button.key == key).click().run()
        return app

    def test_should_render_every_step_without_exceptions(self, app: AppTest):
        # Act
        subheaders = []
        for _ in tutorial_analysis.STEPS:
            subheaders.append(app.subheader[0].value)
            assert not app.exception
            next_button = next(button for button in app.button if button.label == "Next →")
            if not next_button.disabled:
                next_button.click().run()

        # Assert
        assert subheaders == [step.title for step in tutorial_analysis.STEPS]

    def test_should_show_the_parameter_table_of_the_front(self, app: AppTest):
        # Act
        while app.subheader[0].value != "Step 5: what the configurations have in common":
            next(button for button in app.button if button.label == "Next →").click().run()

        # Assert
        assert not app.exception
        assert any("Agreement" in getattr(d.value, "columns", []) for d in app.dataframe)
