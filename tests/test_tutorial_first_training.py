"""Tests for tutorial "Your first guided training"."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio import evolver_client, tutorial_first_training
from evolver_studio.evolver_client import jar_path
from evolver_studio.tutorial_first_training import (
    META_EVALUATIONS,
    PROBLEM,
    latest_tutorial_training,
    training_form_state,
)

APP_SCRIPT = Path(__file__).resolve().parent.parent / "app.py"
RESOURCES = Path(__file__).resolve().parent.parent / "resources"
OPEN_KEY = "tutorial_open_first_training"


@pytest.fixture
def working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    (tmp_path / "resources").symlink_to(RESOURCES)
    monkeypatch.setattr(evolver_client, "WORKING_DIRECTORY", tmp_path)
    return tmp_path


def _open() -> AppTest:
    app = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
    app.switch_page("pages/tutorials.py").run()
    app.button(key=OPEN_KEY).click().run()
    return app


class TestTrainingForm:
    def test_should_fill_the_training_page_with_the_training_of_e3(self, working_directory: Path):
        # Arrange
        app = AppTest.from_file(str(APP_SCRIPT), default_timeout=60)
        for key, value in training_form_state().items():
            app.session_state[key] = value

        # Act
        app.switch_page("pages/training.py").run()

        # Assert: the form shows it, with the population written
        assert not app.exception
        assert app.selectbox(key="train_base_algorithm").value == "NSGA-II"
        assert list(app.multiselect(key="train_problems_Double").value) == [PROBLEM]
        assert app.number_input(key="train_meta_evals").value == META_EVALUATIONS
        assert app.checkbox(key="train_write_population").value is True


class TestWalkThrough:
    def test_should_render_every_step_with_the_shipped_training(self, working_directory: Path):
        # Arrange: no training of the reader yet
        assert latest_tutorial_training() is None
        app = _open()

        # Act
        subheaders = []
        for _ in tutorial_first_training.STEPS:
            subheaders.append(app.subheader[0].value)
            assert not app.exception, subheaders[-1]
            next_button = next(button for button in app.button if button.label == "Next →")
            if not next_button.disabled:
                next_button.click().run()

        # Assert
        assert subheaders == [step.title for step in tutorial_first_training.STEPS]

    def test_should_compare_the_configuration_found_with_the_default(self, working_directory: Path):
        # Arrange
        app = _open()
        while app.subheader[0].value != "Step 5: is it better than the default?":
            next(button for button in app.button if button.label == "Next →").click().run()

        # Act: run both
        for button in [b for b in app.button if b.label == "Run it here"]:
            button.click().run()
        for button in [b for b in app.button if b.label == "Run it here"][1:]:
            button.click().run()

        # Assert
        assert not app.exception
        assert any("Median NHV over the" in m.value for m in app.markdown)


class TestOpenTraining:
    def test_should_open_the_training_page_with_the_training_in_its_form(
        self, working_directory: Path
    ):
        # Arrange
        app = _open()
        while app.subheader[0].value != "Step 3: launch it, and follow it":
            next(button for button in app.button if button.label == "Next →").click().run()

        # Act
        next(b for b in app.button if b.label == "Open Training with this training").click().run()

        # Assert: the Training page, with the form of the tutorial's training
        assert not app.exception
        assert app.title[0].value == "Training"
        assert app.number_input(key="train_meta_evals").value == META_EVALUATIONS
        assert app.checkbox(key="train_write_population").value is True
