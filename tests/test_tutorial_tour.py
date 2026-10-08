"""Tests for tutorial "A tour of Evolver-Studio"."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio import tutorial_tour
from evolver_studio.evolver_client import jar_path

APP_SCRIPT = Path(__file__).resolve().parent.parent / "app.py"


class TestWalkThrough:
    @pytest.fixture
    def app(self) -> AppTest:
        if not jar_path().is_file():
            pytest.skip(f"Evolver jar not found at {jar_path()}")
        app = AppTest.from_file(str(APP_SCRIPT), default_timeout=60).run()
        app.switch_page("pages/tutorials.py").run()
        next(button for button in app.button if button.key == "tutorial_open_tour").click().run()
        return app

    def test_should_render_every_step_without_exceptions(self, app: AppTest):
        # Act
        subheaders = []
        for _ in tutorial_tour.STEPS:
            subheaders.append(app.subheader[0].value)
            assert not app.exception
            next_button = next(button for button in app.button if button.label == "Next →")
            if not next_button.disabled:
                next_button.click().run()

        # Assert
        assert subheaders == [step.title for step in tutorial_tour.STEPS]

    def test_should_count_what_evolver_offers_from_its_manifest(self, app: AppTest):
        # Act
        while app.subheader[0].value != "Step 2: what Evolver offers":
            next(button for button in app.button if button.label == "Next →").click().run()

        # Assert
        counts = {metric.label: int(metric.value) for metric in app.metric}
        assert not app.exception
        assert counts["Base algorithms"] >= 6 and counts["Problems"] > 50
