from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import standings_from_results
from .model import DavidsonModel


@dataclass(frozen=True)
class SimulationResult:
    distributions: pd.DataFrame
    summary: pd.DataFrame
    n_simulations: int
    relegated_slots: int


class ScorelineSampler:
    """Amostra placares condicionais ao resultado BTD.

    Pseudoplacares funcionam como suavização nas primeiras rodadas. Cada placar
    real recebe peso três, de forma que os dados substituam gradualmente o prior.
    """

    _PRIOR = {
        0: [(1, 0)] * 3 + [(2, 0)] * 2 + [(2, 1)] * 3 + [(3, 0), (3, 1), (3, 2)],
        1: [(0, 0)] * 3 + [(1, 1)] * 5 + [(2, 2)] * 2 + [(3, 3)],
        2: [(0, 1)] * 3 + [(0, 2)] * 2 + [(1, 2)] * 3 + [(0, 3), (1, 3), (2, 3)],
    }

    def __init__(self, completed: pd.DataFrame, observed_weight: int = 3):
        if observed_weight < 1:
            raise ValueError("observed_weight deve ser positivo.")
        pools = {key: list(value) for key, value in self._PRIOR.items()}
        for row in completed.itertuples(index=False):
            hg, ag = int(row.home_goals), int(row.away_goals)
            outcome = 0 if hg > ag else (1 if hg == ag else 2)
            pools[outcome].extend([(hg, ag)] * observed_weight)
        self.pools = {key: np.asarray(value, dtype=np.int16) for key, value in pools.items()}

    def sample(self, outcome: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        home_goals = np.empty(len(outcome), dtype=np.int16)
        away_goals = np.empty(len(outcome), dtype=np.int16)
        for value, pool in self.pools.items():
            mask = outcome == value
            count = int(mask.sum())
            if count:
                draw = pool[rng.integers(0, len(pool), size=count)]
                home_goals[mask] = draw[:, 0]
                away_goals[mask] = draw[:, 1]
        return home_goals, away_goals


def simulate_season(
    observed: pd.DataFrame,
    remaining: pd.DataFrame,
    teams: pd.DataFrame,
    model: DavidsonModel,
    n_simulations: int = 10_000,
    relegated_slots: int = 4,
    seed: int = 1970,
) -> SimulationResult:
    """Simula o restante e ordena por P, V, SG e GP, nessa sequência."""

    if n_simulations < 1:
        raise ValueError("n_simulations deve ser positivo.")
    n_teams = len(teams)
    if not 1 <= relegated_slots < n_teams:
        raise ValueError("Quantidade de rebaixados incompatível com o número de clubes.")

    teams = teams.reset_index(drop=True).copy()
    ids = teams["team_id"].astype(str).tolist()
    idx = {team_id: i for i, team_id in enumerate(ids)}
    current = standings_from_results(observed, teams).set_index("team_id").reindex(ids)

    points = np.repeat(current["P"].to_numpy(dtype=np.int16)[None, :], n_simulations, axis=0)
    wins = np.repeat(current["V"].to_numpy(dtype=np.int16)[None, :], n_simulations, axis=0)
    goals_for = np.repeat(current["GP"].to_numpy(dtype=np.int16)[None, :], n_simulations, axis=0)
    goals_against = np.repeat(current["GC"].to_numpy(dtype=np.int16)[None, :], n_simulations, axis=0)

    rng = np.random.default_rng(seed)
    scorelines = ScorelineSampler(observed)
    rows = np.arange(n_simulations)
    for fixture in remaining.itertuples(index=False):
        home_id, away_id = str(fixture.home_id), str(fixture.away_id)
        if home_id not in idx or away_id not in idx:
            raise ValueError("Jogo restante contém clube fora do catálogo.")
        h, a = idx[home_id], idx[away_id]
        p_home, p_draw, _ = model.probabilities(home_id, away_id)
        uniforms = rng.random(n_simulations)
        outcome = np.where(uniforms < p_home, 0, np.where(uniforms < p_home + p_draw, 1, 2))
        hg, ag = scorelines.sample(outcome, rng)

        goals_for[:, h] += hg
        goals_against[:, h] += ag
        goals_for[:, a] += ag
        goals_against[:, a] += hg
        home_win = outcome == 0
        draw = outcome == 1
        away_win = outcome == 2
        points[:, h] += 3 * home_win + draw
        points[:, a] += 3 * away_win + draw
        wins[:, h] += home_win
        wins[:, a] += away_win

    goal_difference = goals_for - goals_against
    # Se os quatro critérios ainda empatarem, jitter apenas evita viés alfabético.
    # Cartões e confronto direto não estão disponíveis no endpoint gratuito usado.
    residual_tie_break = rng.random((n_simulations, n_teams))
    order = np.lexsort(
        (residual_tie_break, -goals_for, -goal_difference, -wins, -points),
        axis=1,
    )
    positions = np.empty_like(order, dtype=np.int16)
    positions[rows[:, None], order] = np.arange(1, n_teams + 1, dtype=np.int16)

    distributions = pd.DataFrame(
        {
            "simulacao": np.repeat(np.arange(1, n_simulations + 1), n_teams),
            "team_id": np.tile(ids, n_simulations),
            "clube": np.tile(teams["team"].to_numpy(), n_simulations),
            "pontos": points.ravel(),
            "vitorias": wins.ravel(),
            "saldo_gols": goal_difference.ravel(),
            "gols_pro": goals_for.ravel(),
            "posicao": positions.ravel(),
        }
    )
    threshold = n_teams - relegated_slots + 1
    summary = (
        distributions.assign(rebaixado=lambda frame: frame["posicao"] >= threshold)
        .groupby(["team_id", "clube"], as_index=False)
        .agg(
            prob_rebaixamento=("rebaixado", "mean"),
            pontos_mediana=("pontos", "median"),
            pontos_p05=("pontos", lambda value: value.quantile(0.05)),
            pontos_p95=("pontos", lambda value: value.quantile(0.95)),
            posicao_mediana=("posicao", "median"),
        )
        .sort_values(["prob_rebaixamento", "pontos_mediana"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary["prob_rebaixamento"] *= 100.0
    return SimulationResult(distributions, summary, n_simulations, relegated_slots)
