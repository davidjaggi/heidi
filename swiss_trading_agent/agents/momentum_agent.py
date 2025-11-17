"""Momentum-based trading agent."""

from typing import Dict
import pandas as pd
import numpy as np
from .base_agent import BaseAgent


class MomentumAgent(BaseAgent):
    """
    Trading agent that uses momentum indicators to generate signals.
    
    Uses moving averages and rate of change to identify trending stocks.
    """
    
    def __init__(self, weight: float = 1.0, short_window: int = 20, long_window: int = 50):
        """
        Initialize the momentum agent.
        
        Args:
            weight: Weight of this agent's signals
            short_window: Short-term moving average window
            long_window: Long-term moving average window
        """
        super().__init__("MomentumAgent", weight)
        self.short_window = short_window
        self.long_window = long_window
    
    def analyze(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Analyze momentum indicators and generate trading signal.
        
        Args:
            symbol: Stock symbol
            data: Historical price data with 'Close' column
            
        Returns:
            Signal dictionary with BUY/SELL/HOLD recommendation
        """
        if len(data) < self.long_window:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'Insufficient data for momentum analysis'
            }
        
        # Calculate moving averages
        data['SMA_short'] = data['Close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['Close'].rolling(window=self.long_window).mean()
        
        # Calculate rate of change
        data['ROC'] = data['Close'].pct_change(periods=10) * 100
        
        # Get latest values
        current_price = data['Close'].iloc[-1]
        sma_short = data['SMA_short'].iloc[-1]
        sma_long = data['SMA_long'].iloc[-1]
        roc = data['ROC'].iloc[-1]
        
        # Generate signal
        signal = 'HOLD'
        confidence = 0.5
        reason = ''
        
        if pd.notna(sma_short) and pd.notna(sma_long):
            # Bullish signal: short MA crosses above long MA
            if sma_short > sma_long and current_price > sma_short:
                if roc > 0:
                    signal = 'BUY'
                    confidence = min(0.9, 0.5 + (roc / 20))
                    reason = f'Strong upward momentum: Price above SMAs, ROC={roc:.2f}%'
                else:
                    signal = 'HOLD'
                    confidence = 0.6
                    reason = 'Price above MAs but negative ROC'
            
            # Bearish signal: short MA crosses below long MA
            elif sma_short < sma_long and current_price < sma_short:
                if roc < 0:
                    signal = 'SELL'
                    confidence = min(0.9, 0.5 + abs(roc) / 20)
                    reason = f'Strong downward momentum: Price below SMAs, ROC={roc:.2f}%'
                else:
                    signal = 'HOLD'
                    confidence = 0.6
                    reason = 'Price below MAs but positive ROC'
            else:
                reason = 'Mixed momentum signals'
        
        result = {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'indicators': {
                'sma_short': sma_short,
                'sma_long': sma_long,
                'roc': roc
            }
        }
        
        self.signals[symbol] = result
        return result
