"""Tests for tutorial "Analyzing a validation", with the real study of "Validating a
configuration"."""

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio import evolver_client, tutorial_validation_analysis
from evolver_studio.evolver_client import WORKING_DIRECTORY, describe, jar_path
from evolver_studio.problem_catalogue import parse_problem_catalogue
from evolver_studio.runs import RunPhase, run_phase
from evolver_studio.tutorial_validation import STUDY_ID

APP_SCRIPT = Path(__file__).resolve().parent.parent / "app.py"
RESOURCES = Path(__file__).resolve().parent.parent / "resources"
RUN_TIMEOUT_SECONDS = 180
OPEN_KEY = "tutorial_open_analyzing_a_validation"


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    if (
        parse_problem_catalogue(getattr(describe(WORKING_DIRECTORY, jar_path()), "value", {}))
        is None
    ):
        pytest.skip("The Evolver jar in use does not describe the problems")
    (tmp_path / "resources").symlink_to(RESOURCES)
    monkeypatch.setattr(evolver_client, "WORKING_DIRECTORY", tmp_path)
    app = AppTest.from_file(str(APP_SCRIPT), default_timeout=120).run()
    app.switch_page("pages/tutorials.py").run()
    app.button(key=OPEN_KEY).click().run()
    return app


def _next(app: AppTest) -> None:
    next(button for button in app.button if button.label == "Next →").click().run()


class TestWalkThrough:
    def test_should_ask_to_run_the_study_while_there_is_none(self, app: AppTest):
        # Act
        _next(app)

        # Assert
        assert not app.exception
        assert any(button.label == "Run the study" for button in app.button)

    def test_should_read_the_study_in_every_step(self, app: AppTest, tmp_path: Path):
        # Arrange: run the study from the introduction
        app.button(key="tutorial_validation_run").click().run()
        directory = tmp_path / "validation-runs" / STUDY_ID
        deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
        while run_phase(directory) in (RunPhase.STARTING, RunPhase.RUNNING):
            assert time.monotonic() < deadline, "the study did not finish"
            time.sleep(0.3)
        app.run()

        # Act
        subheaders = []
        for _ in tutorial_validation_analysis.STEPS:
            subheaders.append(app.subheader[0].value)
            assert not app.exception, subheaders[-1]
            next_button = next(button for button in app.button if button.label == "Next →")
            if not next_button.disabled:
                next_button.click().run()

        # Assert
        assert subheaders == [step.title for step in tutorial_validation_analysis.STEPS]
