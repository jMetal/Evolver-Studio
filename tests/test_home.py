"""Tests for the app's sections and its home page."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evolver_studio.sections import HOME_PAGE, SECTIONS

ROOT = Path(__file__).resolve().parent.parent
APP_SCRIPT = ROOT / "app.py"
PAGES = [page for section in SECTIONS for page in section.pages]
PLACEHOLDER_MARK = "Not implemented yet"


class TestSections:
    @pytest.mark.parametrize("path", [HOME_PAGE] + [page.path for page in PAGES])
    def test_should_point_every_page_at_an_existing_script(self, path: str):
        # Assert
        assert (ROOT / path).is_file()

    @pytest.mark.parametrize("page", PAGES, ids=[page.title for page in PAGES])
    def test_should_mark_as_available_exactly_the_pages_that_are_not_placeholders(self, page):
        # Act
        is_placeholder = PLACEHOLDER_MARK in (ROOT / page.path).read_text()

        # Assert
        assert page.available is not is_placeholder


class TestHomePage:
    @pytest.fixture
    def app(self) -> AppTest:
        return AppTest.from_file(str(APP_SCRIPT), default_timeout=30).run()

    def test_should_open_the_app_on_the_home_page_without_exceptions(self, app: AppTest):
        # Assert
        assert not app.exception
        assert app.title[0].value == "Evolver-Studio"

    def test_should_link_to_every_page(self, app: AppTest):
        # Act
        labels = [link.proto.label for link in app.get("page_link")]

        # Assert
        for section in SECTIONS:
            if section.name == "Learn":  # its tutorials are listed one by one
                continue
            for page in section.pages:
                assert page.title in labels

    def test_should_group_the_cards_under_a_heading_for_each_section(self, app: AppTest):
        # Act
        steps = [h.value for h in app.get("html") if 'class="studio-step"' in h.value]

        # Assert
        for section in SECTIONS:
            assert any(section.name in step for step in steps) or any(
                section.name in subheader.value for subheader in app.subheader
            )
