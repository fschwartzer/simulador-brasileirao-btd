import numpy as np
import pandas as pd

from brasileirao.model import DavidsonModel, fit_davidson


def test_probabilities_sum_to_one_and_home_advantage_helps_home():
    model = DavidsonModel(("a", "b"), np.array([0.0, 0.0]), np.log(1.4), 0.7, True, 0, 0.0)
    p_home, p_draw, p_away = model.probabilities("a", "b")
    assert np.isclose(p_home + p_draw + p_away, 1.0)
    assert p_home > p_away


def test_fit_uses_results_and_returns_centered_strengths():
    results = pd.DataFrame(
        {
            "home_id": ["a", "b", "a", "b", "a", "b"],
            "away_id": ["b", "a", "b", "a", "b", "a"],
            "home_goals": [2, 0, 1, 1, 3, 0],
            "away_goals": [0, 1, 0, 1, 1, 2],
        }
    )
    model = fit_davidson(results, ["a", "b"], regularization=0.5)
    assert model.converged
    assert np.isclose(model.strengths.sum(), 0.0)
    assert np.isclose(model.team_home_advantages.sum(), 0.0)
    assert model.strengths[0] > model.strengths[1]
    assert model.draw_parameter > 0


def test_team_specific_home_advantage_changes_probabilities():
    model = DavidsonModel(
        ("a", "b"),
        np.zeros(2),
        0.0,
        0.7,
        True,
        0,
        0.0,
        np.array([np.log(1.5), -np.log(1.5)]),
    )

    a_at_home = model.probabilities("a", "b")[0]
    b_at_home = model.probabilities("b", "a")[0]

    assert a_at_home > b_at_home


def test_fit_learns_team_specific_home_advantage_and_regularizes_it():
    results = pd.DataFrame(
        {
            "home_id": ["a"] * 8 + ["b"] * 8,
            "away_id": ["b"] * 8 + ["a"] * 8,
            "home_goals": [2] * 8 + [0] * 8,
            "away_goals": [0] * 8 + [2] * 8,
        }
    )

    lightly_regularized = fit_davidson(results, ["a", "b"], home_regularization=0.1)
    strongly_regularized = fit_davidson(results, ["a", "b"], home_regularization=10.0)

    assert lightly_regularized.team_home_advantages[0] > 0
    assert np.isclose(lightly_regularized.team_home_advantages.sum(), 0.0)
    assert np.linalg.norm(strongly_regularized.team_home_advantages) < np.linalg.norm(
        lightly_regularized.team_home_advantages
    )


def test_strength_table_exposes_club_home_advantage():
    model = DavidsonModel(
        ("a", "b"), np.zeros(2), np.log(1.2), 0.7, True, 0, 0.0, np.array([0.1, -0.1])
    )
    teams = pd.DataFrame({"team_id": ["a", "b"], "team": ["A", "B"]})

    table = model.strength_table(teams).set_index("team_id")

    assert np.isclose(table.loc["a", "desvio_mando_log"], 0.1)
    assert np.isclose(table.loc["a", "multiplicador_mando_clube"], np.exp(np.log(1.2) + 0.1))


def test_fit_rejects_negative_home_regularization():
    empty = pd.DataFrame(columns=["home_id", "away_id", "home_goals", "away_goals"])

    with np.testing.assert_raises_regex(ValueError, "home_regularization"):
        fit_davidson(empty, ["a", "b"], home_regularization=-0.1)
