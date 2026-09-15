from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import streamlit as st
from streamlit.testing.v1 import AppTest

from brasileirao.api import FootballDataError
from brasileirao.data import make_demo_matches


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_missing_secret_stops_without_request_or_alternative_source(monkeypatch):
    st.cache_data.clear()
    monkeypatch.delenv("API_TOKEN", raising=False)
    request = Mock()
    monkeypatch.setattr("brasileirao.api.fetch_brasileirao_matches", request)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert "API_TOKEN" in app.error[0].value
    request.assert_not_called()
    assert len(app.text_input) == 0
    assert len(app.radio) == 0
    assert len(app.get("file_uploader")) == 0
    assert len(app.dataframe) == 0


def test_api_error_stops_instead_of_showing_demo(monkeypatch):
    st.cache_data.clear()
    request = Mock(side_effect=FootballDataError("Acesso negado (HTTP 403)."))
    monkeypatch.setattr("brasileirao.api.fetch_brasileirao_matches", request)
    app = AppTest.from_file(str(APP))
    app.secrets["API_TOKEN"] = "cloud-test-token"
    app.run()
    assert not app.exception
    assert "403" in app.error[0].value
    assert len(app.dataframe) == 0


def test_api_success_cache_refresh_and_season_change(monkeypatch):
    st.cache_data.clear()
    # Artificial fixtures are confined to tests; the application calls only the API client.
    matches = make_demo_matches(played_matchdays=2)
    matches.attrs["fetched_at"] = "2026-09-15T12:00:00+00:00"
    request = Mock(return_value=matches)
    monkeypatch.setattr("brasileirao.api.fetch_brasileirao_matches", request)
    app = AppTest.from_file(str(APP), default_timeout=45)
    app.secrets["API_TOKEN"] = "cloud-test-token"
    app.run()
    assert not app.exception
    assert not app.error
    request.assert_called_once_with("cloud-test-token", datetime.now().year)
    assert len(app.tabs) == 5
    assert len(app.dataframe) == 4
    assert any('Risco de <span class="risk-accent">degola.' in element.value for element in app.markdown)
    assert all("cloud-test-token" not in element.value for element in app.markdown)
    app.run()
    assert request.call_count == 1
    app.button[0].click().run()
    assert request.call_count == 2
    app.number_input[0].set_value(2025).run()
    assert not app.exception
    request.assert_called_with("cloud-test-token", 2025)
    assert request.call_count == 3
