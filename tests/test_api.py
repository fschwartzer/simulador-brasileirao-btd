from copy import deepcopy
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from brasileirao.api import FootballDataError, fetch_brasileirao_matches
from brasileirao.config import resolve_api_token


@pytest.fixture
def api_payload():
    return {"matches": [{
        "id": 101, "matchday": 1, "utcDate": "2026-01-28T22:00:00Z",
        "status": "FINISHED", "homeTeam": {"id": 1, "shortName": "Clube A"},
        "awayTeam": {"id": 2, "name": "Clube B"},
        "score": {"fullTime": {"home": 2, "away": 1}},
    }, {
        "id": 102, "matchday": 2, "utcDate": "2026-02-04T22:00:00Z",
        "status": "TIMED", "homeTeam": {"id": 2, "name": "Clube B"},
        "awayTeam": {"id": 1, "shortName": "Clube A"},
        "score": {"fullTime": {"home": None, "away": None}},
    }]}


def mock_response(monkeypatch, payload, status=200):
    response = Mock(status_code=status)
    response.json.return_value = payload
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError("upstream failure")
    request = Mock(return_value=response)
    monkeypatch.setattr("brasileirao.api.requests.get", request)
    return request


def test_secret_precedes_environment_and_is_trimmed():
    assert resolve_api_token({"API_TOKEN": " cloud "}, {"API_TOKEN": "local"}) == "cloud"
    assert resolve_api_token({}, {"API_TOKEN": "local"}) == "local"
    assert resolve_api_token({}, {}) == ""
    assert resolve_api_token({"API_TOKEN": None}, {}) == ""
    assert resolve_api_token({"API_TOKEN": " "}, {"API_TOKEN": "local"}) == ""


def test_api_normalizes_and_records_provenance(monkeypatch, api_payload):
    request = mock_response(monkeypatch, api_payload)
    matches = fetch_brasileirao_matches(" secret-for-test ", 2026)
    request.assert_called_once_with(
        "https://api.football-data.org/v4/competitions/BSA/matches",
        headers={"X-Auth-Token": "secret-for-test"}, params={"season": 2026}, timeout=20.0,
    )
    assert matches["home_id"].tolist() == ["1", "2"]
    assert matches.loc[0, "home_goals"] == 2
    assert pd.isna(matches.loc[1, "home_goals"])
    assert matches.attrs["source"] == "football-data.org"
    assert matches.attrs["season"] == 2026
    assert pd.Timestamp(matches.attrs["fetched_at"]).tzinfo is not None


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_http_failures_do_not_return_data(monkeypatch, status):
    mock_response(monkeypatch, {}, status)
    with pytest.raises(FootballDataError, match=str(status)):
        fetch_brasileirao_matches("test-token", 2026)


@pytest.mark.parametrize("payload", [None, [], {}, {"matches": None}, {"matches": []}, {"matches": [None]}, {"matches": [{"score": "invalid"}]}])
def test_invalid_response_is_rejected(monkeypatch, payload):
    mock_response(monkeypatch, payload)
    with pytest.raises(FootballDataError):
        fetch_brasileirao_matches("test-token", 2026)


def test_inconsistent_finished_game_is_rejected(monkeypatch, api_payload):
    malformed = deepcopy(api_payload)
    malformed["matches"][0]["score"]["fullTime"]["home"] = None
    mock_response(monkeypatch, malformed)
    with pytest.raises(FootballDataError, match="incompletas ou inconsistentes"):
        fetch_brasileirao_matches("test-token", 2026)


def test_network_error_does_not_expose_request_details(monkeypatch):
    monkeypatch.setattr("brasileirao.api.requests.get", Mock(side_effect=requests.Timeout("sensitive-detail")))
    with pytest.raises(FootballDataError) as error:
        fetch_brasileirao_matches("test-token", 2026)
    assert "sensitive-detail" not in str(error.value)


def test_missing_token_does_not_call_api(monkeypatch):
    request = mock_response(monkeypatch, {})
    with pytest.raises(FootballDataError, match="API_TOKEN"):
        fetch_brasileirao_matches(" ", 2026)
    request.assert_not_called()
