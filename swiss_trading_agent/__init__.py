"""
Swiss Trading Agent - A multi-agent system for trading Swiss Market Index stocks.
"""

__version__ = "0.1.0"

from .agents.base_agent import BaseAgent
from .agents.momentum_agent import MomentumAgent
from .agents.value_agent import ValueAgent
from .agents.risk_agent import RiskAgent
from .coordinator import AgentCoordinator

__all__ = [
    "BaseAgent",
    "MomentumAgent", 
    "ValueAgent",
    "RiskAgent",
    "AgentCoordinator",
]
