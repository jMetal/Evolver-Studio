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


class TestTrainingSetProblems:
    def test_should_open_ready_to_launch_the_default_training_set(self, app: AppTest):
        # Assert: NSGA-II, Double, on ZDT4
        assert not app.exception
        assert not app.error
        assert not _launch_button(app).disabled

    def test_should_refuse_a_problem_of_another_encoding_than_the_algorithms(self, app: AppTest):
        # Arrange
        result = describe(WORKING_DIRECTORY, jar_path())
        if parse_problem_catalogue(getattr(result, "value", {})) is None:
            pytest.skip("The Evolver jar in use does not describe the problems")

        # Act: ZDT4 is real-valued
        app.selectbox(key="train_encoding_NSGA-II").select("Binary").run()

        # Assert
        assert any("ZDT4 is a Double problem" in error.value for error in app.error)
        assert _launch_button(app).disabled
