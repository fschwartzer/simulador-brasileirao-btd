"""Simulador auditável do Campeonato Brasileiro Série A."""

from .model import DavidsonModel, fit_davidson
from .simulation import SimulationResult, simulate_season

__all__ = ["DavidsonModel", "SimulationResult", "fit_davidson", "simulate_season"]

