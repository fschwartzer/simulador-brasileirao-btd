import numpy as np
import pandas as pd

from brasileirao.data import make_demo_matches, split_at_matchday, standings_from_results, team_catalog
from brasileirao.model import DavidsonModel, fit_davidson
from brasileirao.simulation import simulate_season


def test_cutoff_does_not_use_finished_future_matches():
    matches = make_demo_matches(played_matchdays=5)
    observed, remaining = split_at_matchday(matches, 3)
    assert observed["matchday"].max() == 3
    assert len(observed) == 30
    assert len(remaining) == 350


def test_standings_tiebreaks_by_wins_then_goal_difference_then_goals_for():
    teams = pd.DataFrame({"team_id": ["a", "b", "c", "d"], "team": ["A", "B", "C", "D"]})
    # A e B: 3 pts, 1 vitória e SG +1; A fica à frente por GP (2 contra 1).
    results = pd.DataFrame(
        {
            "home_id": ["a", "b"],
            "away_id": ["c", "d"],
            "home_goals": [2, 1],
            "away_goals": [1, 0],
        }
    )
    table = standings_from_results(results, teams)
    assert table.iloc[0]["team_id"] == "a"
    assert table.iloc[1]["team_id"] == "b"


def test_simulation_is_reproducible_and_has_one_row_per_team_and_run():
    matches = make_demo_matches(played_matchdays=2)
    teams = team_catalog(matches)
    observed, remaining = split_at_matchday(matches, 2)
    model = fit_davidson(observed, teams["team_id"].tolist())
    first = simulate_season(observed, remaining, teams, model, n_simulations=25, seed=7)
    second = simulate_season(observed, remaining, teams, model, n_simulations=25, seed=7)
    assert len(first.distributions) == 25 * len(teams)
    pd.testing.assert_frame_equal(first.distributions, second.distributions)
    assert np.isclose(first.summary["prob_rebaixamento"].sum(), 400.0)


def test_ordering_respects_official_keys_in_completed_season():
    matches = make_demo_matches(played_matchdays=38)
    teams = team_catalog(matches)
    observed, remaining = split_at_matchday(matches, 38)
    model = fit_davidson(observed, teams["team_id"].tolist())
    result = simulate_season(observed, remaining, teams, model, n_simulations=2, seed=9)
    simulated = result.distributions.query("simulacao == 1").sort_values("posicao")
    keys = list(zip(simulated["pontos"], simulated["vitorias"], simulated["saldo_gols"], simulated["gols_pro"]))
    assert keys == sorted(keys, reverse=True)
