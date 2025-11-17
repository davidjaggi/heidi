"""Value-based trading agent."""

from typing import Dict
import pandas as pd
import numpy as np
from .base_agent import BaseAgent


class ValueAgent(BaseAgent):
    """
    Trading agent that uses value metrics to generate signals.
    
    Analyzes price levels relative to historical averages and volatility.
    """
    
    def __init__(self, weight: float = 1.0, lookback_period: int = 100):
        """
        Initialize the value agent.
        
        Args:
            weight: Weight of this agent's signals
            lookback_period: Number of days to look back for value analysis
        """
        super().__init__("ValueAgent", weight)
        self.lookback_period = lookback_period
    
    def analyze(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Analyze value metrics and generate trading signal.
        
        Args:
            symbol: Stock symbol
            data: Historical price data with 'Close', 'High', 'Low' columns
            
        Returns:
            Signal dictionary with BUY/SELL/HOLD recommendation
        """
        if len(data) < self.lookback_period:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'Insufficient data for value analysis'
            }
        
        # Calculate value indicators
        recent_data = data.tail(self.lookback_period)
        
        current_price = data['Close'].iloc[-1]
        mean_price = recent_data['Close'].mean()
        std_price = recent_data['Close'].std()
        
        # Calculate Z-score (how many standard deviations from mean)
        z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
        
        # Calculate price percentile
        price_percentile = (recent_data['Close'] < current_price).sum() / len(recent_data) * 100
        
        # Calculate Average True Range for volatility
        data['H-L'] = data['High'] - data['Low']
        data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
        data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
        data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        atr = data['TR'].tail(14).mean()
        
        # Generate signal based on value metrics
        signal = 'HOLD'
        confidence = 0.5
        reason = ''
        
        # Undervalued (good buy opportunity)
        if z_score < -1.5:
            signal = 'BUY'
            confidence = min(0.9, 0.5 + abs(z_score) / 4)
            reason = f'Undervalued: Price {abs(z_score):.2f} std devs below mean (percentile: {price_percentile:.0f}%)'
        
        # Overvalued (good sell opportunity)
        elif z_score > 1.5:
            signal = 'SELL'
            confidence = min(0.9, 0.5 + z_score / 4)
            reason = f'Overvalued: Price {z_score:.2f} std devs above mean (percentile: {price_percentile:.0f}%)'
        
        # Moderately undervalued
        elif z_score < -0.5:
            signal = 'BUY'
            confidence = 0.6
            reason = f'Moderately undervalued (percentile: {price_percentile:.0f}%)'
        
        # Moderately overvalued
        elif z_score > 0.5:
            signal = 'SELL'
            confidence = 0.6
            reason = f'Moderately overvalued (percentile: {price_percentile:.0f}%)'
        
        else:
            reason = f'Fair value (percentile: {price_percentile:.0f}%)'
        
        result = {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'indicators': {
                'z_score': z_score,
                'price_percentile': price_percentile,
                'mean_price': mean_price,
                'atr': atr
            }
        }
        
        self.signals[symbol] = result
        return result
