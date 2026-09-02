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


@dataclass(frozen=True)
class DavidsonBacktestResult:
    """Seleção de hiperparâmetros obtida por validação temporal expansiva."""

    regularization: float
    home_advantage_regularization: float
    home_regularization: float
    decay_half_life: float | None
    scores: pd.DataFrame
    validation_matchdays: tuple[int, ...]
    n_validation_matches: int


def fit_davidson(
    completed: pd.DataFrame,
    team_ids: list[str] | tuple[str, ...],
    regularization: float = 0.25,
    home_regularization: float = 1.0,
    home_advantage_regularization: float | None = None,
    decay_half_life: float | None = None,
    reference_matchday: int | None = None,
) -> DavidsonModel:
    """Ajusta a máxima verossimilhança penalizada sem usar jogos futuros.

    As somas das forças e dos desvios de mando por clube são fixadas em zero
    para identificabilidade. Penalizações L2 separadas estabilizam forças,
    mando médio e mando específico. Quando ``decay_half_life`` é informado,
    jogos antigos recebem peso exponencial menor, normalizado para média um.
    """

    ids = tuple(str(item) for item in team_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("team_ids precisa conter ao menos dois IDs únicos.")
    if regularization < 0:
        raise ValueError("regularization não pode ser negativa.")
    if home_regularization < 0:
        raise ValueError("home_regularization não pode ser negativa.")
    if home_advantage_regularization is None:
        # Mantém a semântica das chamadas anteriores, nas quais h usava a
        # mesma penalização das forças.
        home_advantage_regularization = regularization
    if home_advantage_regularization < 0:
        raise ValueError("home_advantage_regularization não pode ser negativa.")
    if decay_half_life is not None and (not np.isfinite(decay_half_life) or decay_half_life <= 0):
        raise ValueError("decay_half_life precisa ser positiva e finita.")

    index = {team_id: i for i, team_id in enumerate(ids)}
    columns = ["home_id", "away_id", "home_goals", "away_goals"]
    if decay_half_life is not None:
        if "matchday" not in completed.columns:
            raise ValueError("matchday é obrigatória quando há decaimento temporal.")
        columns.append("matchday")
    rows = completed.loc[:, columns]
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

    weights = np.ones(len(rows), dtype=float)
    if decay_half_life is not None:
        matchdays = pd.to_numeric(rows["matchday"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(matchdays).all():
            raise ValueError("matchday contém valores ausentes ou inválidos.")
        reference = float(np.max(matchdays) if reference_matchday is None else reference_matchday)
        if np.any(matchdays > reference):
            raise ValueError("reference_matchday não pode anteceder jogos usados no ajuste.")
        weights = np.power(0.5, (reference - matchdays) / float(decay_half_life))
        weights /= weights.mean()

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

    def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        theta, team_home_advantages, log_nu, home_advantage = unpack(parameters)
        log_home = theta[home] + home_advantage + team_home_advantages[home]
        log_away = theta[away]
        log_draw = log_nu + 0.5 * (log_home + log_away)
        logits = np.column_stack([log_home, log_draw, log_away])
        selected = logits[np.arange(len(outcome)), outcome]
        log_denominator = logsumexp(logits, axis=1)
        negative_log_likelihood = np.sum(weights * (log_denominator - selected))
        penalty = regularization * np.dot(theta, theta)
        penalty += home_advantage_regularization * home_advantage**2
        penalty += home_regularization * np.dot(team_home_advantages, team_home_advantages)

        probabilities = np.exp(logits - log_denominator[:, None])
        probabilities[np.arange(len(outcome)), outcome] -= 1.0
        residual = weights[:, None] * probabilities
        home_residual = residual[:, 0] + 0.5 * residual[:, 1]
        away_residual = residual[:, 2] + 0.5 * residual[:, 1]

        theta_gradient = np.zeros(len(ids), dtype=float)
        np.add.at(theta_gradient, home, home_residual)
        np.add.at(theta_gradient, away, away_residual)
        theta_gradient += 2.0 * regularization * theta

        delta_gradient = np.zeros(len(ids), dtype=float)
        np.add.at(delta_gradient, home, home_residual)
        delta_gradient += 2.0 * home_regularization * team_home_advantages

        gradient = np.concatenate(
            [
                theta_gradient[:-1] - theta_gradient[-1],
                delta_gradient[:-1] - delta_gradient[-1],
                [float(residual[:, 1].sum())],
                [
                    float(home_residual.sum())
                    + 2.0 * home_advantage_regularization * home_advantage
                ],
            ]
        )
        return float(negative_log_likelihood + penalty), gradient

    bounds = (
        [(-5.0, 5.0)] * (len(ids) - 1)
        + [(-2.0, 2.0)] * (len(ids) - 1)
        + [(-4.0, 4.0), (-2.0, 2.0)]
    )
    result = minimize(objective_and_gradient, x0, method="L-BFGS-B", jac=True, bounds=bounds)
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


def select_davidson_hyperparameters(
    completed: pd.DataFrame,
    team_ids: list[str] | tuple[str, ...],
    regularizations: tuple[float, ...] = (0.25, 1.0, 3.0, 5.0),
    home_advantage_regularizations: tuple[float, ...] = (0.0, 1.0, 5.0),
    home_regularizations: tuple[float, ...] = (1.0, 10.0, 200.0),
    decay_half_lives: tuple[float | None, ...] = (4.0, 12.0, 38.0, None),
    min_training_matchdays: int = 10,
    max_validation_matchdays: int = 8,
) -> DavidsonBacktestResult:
    """Seleciona decaimento e regularizações sem usar rodadas futuras.

    Cada fold treina em todas as rodadas anteriores e prevê a rodada seguinte.
    A ordenação usa log loss médio; Brier score é mantido como diagnóstico.
    """

    required = {"matchday", "home_id", "away_id", "home_goals", "away_goals"}
    missing = required.difference(completed.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no backtest: {sorted(missing)}")
    if min_training_matchdays < 1 or max_validation_matchdays < 1:
        raise ValueError("As quantidades de rodadas do backtest precisam ser positivas.")

    candidates = (
        tuple(float(value) for value in regularizations),
        tuple(float(value) for value in home_advantage_regularizations),
        tuple(float(value) for value in home_regularizations),
        tuple(None if value is None else float(value) for value in decay_half_lives),
    )
    if any(not values for values in candidates):
        raise ValueError("Cada grade de hiperparâmetros precisa ter ao menos um valor.")
    if any(not np.isfinite(value) or value < 0 for values in candidates[:3] for value in values):
        raise ValueError("Regularizações do backtest precisam ser finitas e não negativas.")
    if any(
        not np.isfinite(value) or value <= 0
        for value in candidates[3]
        if value is not None
    ):
        raise ValueError("Meias-vidas do backtest precisam ser positivas e finitas.")

    rows = completed.copy()
    rows["matchday"] = pd.to_numeric(rows["matchday"], errors="coerce")
    if rows["matchday"].isna().any():
        raise ValueError("matchday contém valores ausentes ou inválidos.")
    rows["matchday"] = rows["matchday"].astype(int)
    matchdays = sorted(rows["matchday"].unique().tolist())
    eligible = matchdays[min_training_matchdays:]
    validation_matchdays = tuple(eligible[-max_validation_matchdays:])
    if not validation_matchdays:
        raise ValueError("Não há rodadas suficientes para o backtest temporal.")

    parsed_dates = None
    if "utc_date" in rows.columns:
        parsed_dates = pd.to_datetime(rows["utc_date"], errors="coerce", utc=True)

    folds: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for matchday in validation_matchdays:
        validation = rows.loc[rows["matchday"] == matchday]
        training_mask = rows["matchday"] < matchday
        if parsed_dates is not None and parsed_dates.loc[validation.index].notna().all():
            validation_start = parsed_dates.loc[validation.index].min()
            training_mask &= parsed_dates < validation_start
        training = rows.loc[training_mask]
        if not training.empty and not validation.empty:
            folds.append((matchday, training, validation))
    if not folds:
        raise ValueError("Nenhum fold temporal válido pôde ser construído.")

    score_rows = []
    ids = tuple(str(item) for item in team_ids)
    for regularization in candidates[0]:
        for home_advantage_regularization in candidates[1]:
            for home_regularization in candidates[2]:
                for decay_half_life in candidates[3]:
                    losses: list[float] = []
                    brier_scores: list[float] = []
                    converged = 0
                    for matchday, training, validation in folds:
                        model = fit_davidson(
                            training,
                            ids,
                            regularization=regularization,
                            home_regularization=home_regularization,
                            home_advantage_regularization=home_advantage_regularization,
                            decay_half_life=decay_half_life,
                            reference_matchday=matchday - 1,
                        )
                        converged += int(model.converged)
                        for row in validation.itertuples(index=False):
                            probabilities = np.asarray(
                                model.probabilities(str(row.home_id), str(row.away_id)), dtype=float
                            )
                            outcome = 0 if row.home_goals > row.away_goals else (
                                1 if row.home_goals == row.away_goals else 2
                            )
                            losses.append(float(-np.log(np.clip(probabilities[outcome], 1e-12, 1.0))))
                            expected = np.zeros(3, dtype=float)
                            expected[outcome] = 1.0
                            brier_scores.append(float(np.sum((probabilities - expected) ** 2)))
                    score_rows.append(
                        {
                            "regularization": regularization,
                            "home_advantage_regularization": home_advantage_regularization,
                            "home_regularization": home_regularization,
                            "decay_half_life": decay_half_life,
                            "log_loss": float(np.mean(losses)),
                            "brier_score": float(np.mean(brier_scores)),
                            "log_loss_se": float(np.std(losses, ddof=1) / np.sqrt(len(losses)))
                            if len(losses) > 1
                            else 0.0,
                            "convergence_rate": converged / len(folds),
                            "n_matches": len(losses),
                        }
                    )

    scores = pd.DataFrame(score_rows).sort_values(
        ["log_loss", "brier_score"], ascending=True
    ).reset_index(drop=True)
    converged_scores = scores.loc[np.isclose(scores["convergence_rate"], 1.0)]
    best = converged_scores.iloc[0] if not converged_scores.empty else scores.iloc[0]
    best_half_life = (
        None if pd.isna(best["decay_half_life"]) else float(best["decay_half_life"])
    )
    return DavidsonBacktestResult(
        regularization=float(best["regularization"]),
        home_advantage_regularization=float(best["home_advantage_regularization"]),
        home_regularization=float(best["home_regularization"]),
        decay_half_life=best_half_life,
        scores=scores,
        validation_matchdays=tuple(matchday for matchday, _, _ in folds),
        n_validation_matches=int(best["n_matches"]),
    )
