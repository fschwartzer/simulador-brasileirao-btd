"""Simulador auditável do Campeonato Brasileiro Série A."""

from .model import DavidsonBacktestResult, DavidsonModel, fit_davidson, select_davidson_hyperparameters
from .simulation import SimulationResult, simulate_season

__all__ = [
    "DavidsonBacktestResult",
    "DavidsonModel",
    "SimulationResult",
    "fit_davidson",
    "select_davidson_hyperparameters",
    "simulate_season",
]
