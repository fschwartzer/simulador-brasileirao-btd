from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "matchday",
    "status",
    "home_id",
    "home_team",
    "away_id",
    "away_team",
    "home_goals",
    "away_goals",
}
FINISHED_STATUSES = {"FINISHED", "AWARDED"}


class DataValidationError(ValueError):
    """Erro de contrato dos dados de partidas."""


def validate_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Valida e normaliza o contrato usado pelo modelo.

    Placares ausentes são permitidos somente em partidas ainda não encerradas.
    IDs são convertidos para texto para evitar junções silenciosamente inválidas.
    """

    missing = REQUIRED_COLUMNS.difference(matches.columns)
    if missing:
        raise DataValidationError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    df = matches.copy()
    df["home_id"] = df["home_id"].astype("string")
    df["away_id"] = df["away_id"].astype("string")
    df["home_team"] = df["home_team"].astype("string").str.strip()
    df["away_team"] = df["away_team"].astype("string").str.strip()
    df["status"] = df["status"].astype("string").str.upper().str.strip()
    df["matchday"] = pd.to_numeric(df["matchday"], errors="coerce").astype("Int64")
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").astype("Int64")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").astype("Int64")

    if df[["home_id", "away_id", "home_team", "away_team"]].isna().any().any():
        raise DataValidationError("Toda partida precisa ter IDs e nomes dos dois clubes.")
    if (df["home_id"] == df["away_id"]).any():
        raise DataValidationError("Uma partida não pode ter o mesmo clube nos dois lados.")
    if df["matchday"].isna().any() or (df["matchday"] < 1).any():
        raise DataValidationError("Rodada ausente ou inválida.")

    finished = df["status"].isin(FINISHED_STATUSES)
    if df.loc[finished, ["home_goals", "away_goals"]].isna().any().any():
        raise DataValidationError("Partida encerrada sem placar final.")
    goals = df.loc[finished, ["home_goals", "away_goals"]]
    if (goals < 0).any().any():
        raise DataValidationError("Gols não podem ser negativos.")

    if "match_id" in df.columns and df["match_id"].notna().any():
        duplicated = df.loc[df["match_id"].notna(), "match_id"].duplicated()
        if duplicated.any():
            raise DataValidationError("Há IDs de partidas duplicados.")

    return df.sort_values(["matchday", "utc_date"] if "utc_date" in df else ["matchday"]).reset_index(drop=True)


def team_catalog(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["home_id", "home_team"]].rename(columns={"home_id": "team_id", "home_team": "team"})
    away = matches[["away_id", "away_team"]].rename(columns={"away_id": "team_id", "away_team": "team"})
    teams = pd.concat([home, away], ignore_index=True).drop_duplicates()
    conflicts = teams.groupby("team_id")["team"].nunique()
    if (conflicts > 1).any():
        bad = conflicts[conflicts > 1].index.tolist()
        raise DataValidationError(f"ID associado a mais de um nome de clube: {bad}")
    return teams.sort_values("team").reset_index(drop=True)


def split_at_matchday(matches: pd.DataFrame, cutoff: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa informação observada e jogos a simular sem olhar além do corte."""

    is_observed = matches["status"].isin(FINISHED_STATUSES) & (matches["matchday"] <= cutoff)
    observed = matches.loc[is_observed].copy()
    remaining = matches.loc[~is_observed].copy()
    return observed, remaining


def standings_from_results(results: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Calcula a classificação pelos quatro primeiros critérios solicitados da CBF."""

    ids = teams["team_id"].astype(str).tolist()
    idx = {team_id: i for i, team_id in enumerate(ids)}
    n = len(ids)
    played = np.zeros(n, dtype=int)
    points = np.zeros(n, dtype=int)
    wins = np.zeros(n, dtype=int)
    draws = np.zeros(n, dtype=int)
    losses = np.zeros(n, dtype=int)
    gf = np.zeros(n, dtype=int)
    ga = np.zeros(n, dtype=int)

    for row in results.itertuples(index=False):
        h, a = idx[str(row.home_id)], idx[str(row.away_id)]
        hg, ag = int(row.home_goals), int(row.away_goals)
        played[[h, a]] += 1
        gf[h] += hg
        ga[h] += ag
        gf[a] += ag
        ga[a] += hg
        if hg > ag:
            points[h] += 3
            wins[h] += 1
            losses[a] += 1
        elif hg < ag:
            points[a] += 3
            wins[a] += 1
            losses[h] += 1
        else:
            points[[h, a]] += 1
            draws[[h, a]] += 1

    table = teams.copy().reset_index(drop=True)
    table["J"] = played
    table["P"] = points
    table["V"] = wins
    table["E"] = draws
    table["D"] = losses
    table["GP"] = gf
    table["GC"] = ga
    table["SG"] = gf - ga
    table = table.sort_values(["P", "V", "SG", "GP", "team"], ascending=[False, False, False, False, True])
    table.insert(0, "Pos", np.arange(1, len(table) + 1))
    return table.reset_index(drop=True)


def make_demo_matches(played_matchdays: int = 12, seed: int = 1952) -> pd.DataFrame:
    """Gera calendário completo artificial para demonstração offline."""

    names = [
        "Athletico-PR", "Atlético-MG", "Bahia", "Botafogo", "Chapecoense",
        "Corinthians", "Coritiba", "Cruzeiro", "Flamengo", "Fluminense",
        "Grêmio", "Internacional", "Mirassol", "Palmeiras", "RB Bragantino",
        "Remo", "Santos", "São Paulo", "Vasco", "Vitória",
    ]
    team_ids = [f"demo-{i:02d}" for i in range(1, len(names) + 1)]
    rotation = list(range(len(names)))
    first_leg: list[tuple[int, int, int]] = []
    for matchday in range(1, len(names)):
        for k in range(len(names) // 2):
            left, right = rotation[k], rotation[-(k + 1)]
            home, away = (left, right) if (matchday + k) % 2 else (right, left)
            first_leg.append((matchday, home, away))
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]

    fixtures = first_leg + [(md + 19, away, home) for md, home, away in first_leg]
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 0.28, len(names))
    rows = []
    for match_id, (matchday, home, away) in enumerate(fixtures, start=1):
        finished = matchday <= played_matchdays
        if finished:
            hg = int(rng.poisson(np.exp(0.35 + strength[home] - 0.35 * strength[away])))
            ag = int(rng.poisson(np.exp(0.08 + strength[away] - 0.35 * strength[home])))
        else:
            hg = ag = pd.NA
        rows.append(
            {
                "match_id": f"demo-match-{match_id}",
                "matchday": matchday,
                "utc_date": pd.Timestamp("2026-01-28", tz="UTC") + pd.to_timedelta(int(matchday - 1) * 7, unit="D"),
                "status": "FINISHED" if finished else "SCHEDULED",
                "home_id": team_ids[home],
                "home_team": names[home],
                "away_id": team_ids[away],
                "away_team": names[away],
                "home_goals": hg,
                "away_goals": ag,
            }
        )
    return validate_matches(pd.DataFrame(rows))
