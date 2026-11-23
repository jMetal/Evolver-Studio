"""Tests for how the app is laid out for Streamlit."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_should_not_have_a_pages_folder():
    """Streamlit serves a folder called pages/ by itself, without app.py and so without the menu,
    when the first thing opened after the server starts is the URL of a page."""
    # Assert
    assert not (ROOT / "pages").exists() or not list((ROOT / "pages").glob("*.py"))
    assert list((ROOT / "app_pages").glob("*.py"))
