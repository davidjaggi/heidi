"""Configuration management for the trading system."""

from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration settings for the Swiss Trading Agent system."""
    
    # Agent configurations
    MOMENTUM_AGENT_WEIGHT = float(os.getenv('MOMENTUM_AGENT_WEIGHT', '1.0'))
    VALUE_AGENT_WEIGHT = float(os.getenv('VALUE_AGENT_WEIGHT', '1.0'))
    RISK_AGENT_WEIGHT = float(os.getenv('RISK_AGENT_WEIGHT', '1.2'))  # Higher weight for risk
    
    # Data configurations
    DEFAULT_PERIOD = os.getenv('DEFAULT_PERIOD', '1y')
    DEFAULT_INTERVAL = os.getenv('DEFAULT_INTERVAL', '1d')
    
    # Trading parameters
    MAX_VOLATILITY = float(os.getenv('MAX_VOLATILITY', '0.30'))  # 30%
    
    # Momentum parameters
    MOMENTUM_SHORT_WINDOW = int(os.getenv('MOMENTUM_SHORT_WINDOW', '20'))
    MOMENTUM_LONG_WINDOW = int(os.getenv('MOMENTUM_LONG_WINDOW', '50'))
    
    # Value parameters
    VALUE_LOOKBACK_PERIOD = int(os.getenv('VALUE_LOOKBACK_PERIOD', '100'))
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'momentum_agent_weight': cls.MOMENTUM_AGENT_WEIGHT,
            'value_agent_weight': cls.VALUE_AGENT_WEIGHT,
            'risk_agent_weight': cls.RISK_AGENT_WEIGHT,
            'default_period': cls.DEFAULT_PERIOD,
            'default_interval': cls.DEFAULT_INTERVAL,
            'max_volatility': cls.MAX_VOLATILITY,
            'momentum_short_window': cls.MOMENTUM_SHORT_WINDOW,
            'momentum_long_window': cls.MOMENTUM_LONG_WINDOW,
            'value_lookback_period': cls.VALUE_LOOKBACK_PERIOD,
        }
