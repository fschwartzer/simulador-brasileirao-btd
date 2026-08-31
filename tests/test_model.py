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
    assert model.strengths[0] > model.strengths[1]
    assert model.draw_parameter > 0

