from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass(frozen=True)
class DavidsonModel:
    """Bradley–Terry–Davidson com vantagem de mando geral e por clube.

    Para forças ``pi_i = exp(theta_i)``, mando médio ``exp(h)`` e desvio de
    mando ``delta_i``, o denominador é ``a + b + nu * sqrt(a*b)``, onde
    ``a=pi_home*exp(h + delta_home)`` e ``b=pi_away``.
    """

    team_ids: tuple[str, ...]
    strengths: np.ndarray
    home_advantage: float
    draw_parameter: float
    converged: bool
    n_matches: int
    objective: float
    team_home_advantages: np.ndarray | None = None

    def _team_home_advantages(self) -> np.ndarray:
        """Retorna desvios por clube, incluindo modelos antigos sem o campo."""

        if self.team_home_advantages is None:
            return np.zeros(len(self.team_ids), dtype=float)
        values = np.asarray(self.team_home_advantages, dtype=float)
        if values.shape != (len(self.team_ids),):
            raise ValueError("team_home_advantages deve ter um valor por clube.")
        return values

    def probabilities(self, home_id: str, away_id: str) -> tuple[float, float, float]:
        index = {team_id: i for i, team_id in enumerate(self.team_ids)}
        try:
            home_theta = float(self.strengths[index[str(home_id)]])
            away_theta = float(self.strengths[index[str(away_id)]])
        except KeyError as exc:
            raise KeyError(f"Clube desconhecido no modelo: {exc.args[0]}") from exc

        home_delta = float(self._team_home_advantages()[index[str(home_id)]])
        log_home = home_theta + self.home_advantage + home_delta
        log_away = away_theta
        log_draw = np.log(self.draw_parameter) + 0.5 * (log_home + log_away)
        log_denominator = logsumexp([log_home, log_draw, log_away])
        return (
            float(np.exp(log_home - log_denominator)),
            float(np.exp(log_draw - log_denominator)),
            float(np.exp(log_away - log_denominator)),
        )

    def strength_table(self, teams: pd.DataFrame) -> pd.DataFrame:
        lookup = dict(zip(self.team_ids, self.strengths, strict=True))
        result = teams.copy()
        result["forca_log"] = result["team_id"].astype(str).map(lookup)
        result["forca_relativa"] = np.exp(result["forca_log"])
        result["forca_relativa"] /= result["forca_relativa"].mean()
        home_lookup = dict(zip(self.team_ids, self._team_home_advantages(), strict=True))
        result["desvio_mando_log"] = result["team_id"].astype(str).map(home_lookup)
        result["multiplicador_mando_clube"] = np.exp(
            self.home_advantage + result["desvio_mando_log"]
        )
        return result.sort_values("forca_relativa", ascending=False).reset_index(drop=True)


def fit_davidson(
    completed: pd.DataFrame,
    team_ids: list[str] | tuple[str, ...],
    regularization: float = 0.25,
    home_regularization: float = 1.0,
) -> DavidsonModel:
    """Ajusta a máxima verossimilhança penalizada sem usar jogos futuros.

    As somas das forças e dos desvios de mando por clube são fixadas em zero
    para identificabilidade. Penalizações L2 separadas estabilizam as forças e
    encolhem o mando específico nas rodadas iniciais.
    """

    ids = tuple(str(item) for item in team_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("team_ids precisa conter ao menos dois IDs únicos.")
    if regularization < 0:
        raise ValueError("regularization não pode ser negativa.")
    if home_regularization < 0:
        raise ValueError("home_regularization não pode ser negativa.")

    index = {team_id: i for i, team_id in enumerate(ids)}
    rows = completed.loc[:, ["home_id", "away_id", "home_goals", "away_goals"]]
    unknown = (set(rows["home_id"].astype(str)) | set(rows["away_id"].astype(str))).difference(ids)
    if unknown:
        raise ValueError(f"Partidas contêm clubes fora do catálogo: {sorted(unknown)}")

    if rows.empty:
        return DavidsonModel(
            ids,
            np.zeros(len(ids)),
            0.0,
            0.65,
            True,
            0,
            0.0,
            np.zeros(len(ids)),
        )

    home = rows["home_id"].astype(str).map(index).to_numpy(dtype=int)
    away = rows["away_id"].astype(str).map(index).to_numpy(dtype=int)
    home_goals = rows["home_goals"].to_numpy(dtype=float)
    away_goals = rows["away_goals"].to_numpy(dtype=float)
    outcome = np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))

    draw_rate = float(np.mean(outcome == 1))
    initial_nu = np.clip(2.0 * draw_rate / max(1.0 - draw_rate, 1e-6), 0.15, 3.0)
    x0 = np.zeros(2 * len(ids))
    x0[-2] = np.log(initial_nu)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        theta_free = parameters[: len(ids) - 1]
        theta = np.append(theta_free, -theta_free.sum())
        delta_start = len(ids) - 1
        delta_free = parameters[delta_start : 2 * delta_start]
        team_home_advantages = np.append(delta_free, -delta_free.sum())
        log_nu = float(parameters[-2])
        home_advantage = float(parameters[-1])
        return theta, team_home_advantages, log_nu, home_advantage

    def objective(parameters: np.ndarray) -> float:
        theta, team_home_advantages, log_nu, home_advantage = unpack(parameters)
        log_home = theta[home] + home_advantage + team_home_advantages[home]
        log_away = theta[away]
        log_draw = log_nu + 0.5 * (log_home + log_away)
        logits = np.column_stack([log_home, log_draw, log_away])
        selected = logits[np.arange(len(outcome)), outcome]
        negative_log_likelihood = np.sum(logsumexp(logits, axis=1) - selected)
        penalty = regularization * (np.dot(theta, theta) + home_advantage**2)
        penalty += home_regularization * np.dot(team_home_advantages, team_home_advantages)
        return float(negative_log_likelihood + penalty)

    bounds = (
        [(-5.0, 5.0)] * (len(ids) - 1)
        + [(-2.0, 2.0)] * (len(ids) - 1)
        + [(-4.0, 4.0), (-2.0, 2.0)]
    )
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    theta, team_home_advantages, log_nu, home_advantage = unpack(result.x)
    return DavidsonModel(
        team_ids=ids,
        strengths=theta,
        home_advantage=home_advantage,
        draw_parameter=float(np.exp(log_nu)),
        converged=bool(result.success),
        n_matches=len(rows),
        objective=float(result.fun),
        team_home_advantages=team_home_advantages,
    )
