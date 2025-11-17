"""Base agent class for all trading agents."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class BaseAgent(ABC):
    """
    Base class for all trading agents.
    
    Each agent analyzes market data and provides trading signals.
    """
    
    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize the base agent.
        
        Args:
            name: Name of the agent
            weight: Weight of this agent's signals in the overall decision (0-1)
        """
        self.name = name
        self.weight = weight
        self.signals = {}
    
    @abstractmethod
    def analyze(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Analyze market data and generate trading signals.
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            
        Returns:
            Dictionary with signal information:
            {
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'confidence': float (0-1),
                'reason': str
            }
        """
        pass
    
    def get_signal(self, symbol: str) -> Optional[Dict]:
        """Get the last signal for a symbol."""
        return self.signals.get(symbol)
    
    def clear_signals(self):
        """Clear all stored signals."""
        self.signals = {}
