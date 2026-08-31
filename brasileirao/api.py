from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from .data import validate_matches


BASE_URL = "https://api.football-data.org/v4"


class FootballDataError(RuntimeError):
    """Erro traduzido da football-data.org."""


def _extract_goals(match: dict[str, Any]) -> tuple[int | None, int | None]:
    score = match.get("score") or {}
    full_time = score.get("fullTime") or {}
    return full_time.get("home"), full_time.get("away")


def fetch_brasileirao_matches(token: str, season: int, timeout: float = 20.0) -> pd.DataFrame:
    """Obtém a temporada completa do Brasileirão Série A no plano gratuito."""

    if not token or not token.strip():
        raise FootballDataError("Informe uma chave da football-data.org.")
    try:
        response = requests.get(
            f"{BASE_URL}/competitions/BSA/matches",
            headers={"X-Auth-Token": token.strip()},
            params={"season": int(season)},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise FootballDataError(f"Falha de rede ao consultar a API: {exc}") from exc

    if response.status_code == 401:
        raise FootballDataError("Chave inválida ou ausente (HTTP 401).")
    if response.status_code == 403:
        raise FootballDataError("A conta não tem acesso a essa temporada/competição (HTTP 403).")
    if response.status_code == 429:
        raise FootballDataError("Limite de requisições atingido; aguarde a renovação da cota (HTTP 429).")
    try:
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise FootballDataError(f"Resposta inválida da API (HTTP {response.status_code}).") from exc

    rows: list[dict[str, Any]] = []
    for match in payload.get("matches", []):
        home_goals, away_goals = _extract_goals(match)
        home = match.get("homeTeam") or {}
        away = match.get("awayTeam") or {}
        rows.append(
            {
                "match_id": match.get("id"),
                "matchday": match.get("matchday"),
                "utc_date": pd.to_datetime(match.get("utcDate"), utc=True, errors="coerce"),
                "status": match.get("status"),
                "home_id": home.get("id"),
                "home_team": home.get("shortName") or home.get("name"),
                "away_id": away.get("id"),
                "away_team": away.get("shortName") or away.get("name"),
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )
    if not rows:
        raise FootballDataError("A API não retornou partidas para a temporada informada.")
    return validate_matches(pd.DataFrame(rows))


def parse_uploaded_csv(file: Any) -> pd.DataFrame:
    """Lê CSV no mesmo contrato interno usado pelo cliente da API."""

    try:
        frame = pd.read_csv(file)
    except Exception as exc:  # pandas expõe diferentes erros por engine/encoding
        raise FootballDataError(f"Não foi possível ler o CSV: {exc}") from exc
    return validate_matches(frame)
