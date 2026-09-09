"""Smoke tests for the optional Streamlit GUI (idem/gui/app.py).

These use Streamlit's own AppTest harness to actually execute the
script in a simulated session — catching import errors, exceptions
during a normal render, and basic structural regressions — without
spinning up a real browser or server. Deeper interaction testing
(clicking Validate/Run, typing into the editor) is left out for now:
the underlying logic they trigger (idem.config, idem.runner) already
has full coverage elsewhere, so this file's job is just to confirm the
GUI wires it up without crashing.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "idem" / "gui" / "app.py")


def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception


def test_app_shows_title():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert any("Idem" in t.value for t in at.title)


def test_app_prefills_editor_with_default_template():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert len(at.text_area) == 1
    assert "model:" in at.text_area[0].value
    assert "questions:" in at.text_area[0].value


def test_validate_button_reports_valid_config():
    at = AppTest.from_file(APP_PATH)
    at.run()
    validate_button = next(b for b in at.button if b.label == "Validate")
    validate_button.click().run()
    assert not at.exception
    assert any("valid" in s.value.lower() for s in at.success)


def test_validate_button_reports_invalid_config():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_area[0].set_value("not: valid\nquestions: []\n")
    validate_button = next(b for b in at.button if b.label == "Validate")
    validate_button.click().run()
    assert not at.exception
    assert len(at.error) > 0
