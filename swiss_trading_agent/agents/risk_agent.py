"""Risk management agent."""

from typing import Dict
import pandas as pd
import numpy as np
from .base_agent import BaseAgent


class RiskAgent(BaseAgent):
    """
    Trading agent that assesses risk and provides risk-adjusted signals.
    
    Analyzes volatility, drawdowns, and risk metrics.
    """
    
    def __init__(self, weight: float = 1.0, max_volatility: float = 0.30):
        """
        Initialize the risk agent.
        
        Args:
            weight: Weight of this agent's signals
            max_volatility: Maximum acceptable annualized volatility (default 30%)
        """
        super().__init__("RiskAgent", weight)
        self.max_volatility = max_volatility
    
    def analyze(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Analyze risk metrics and generate trading signal.
        
        Args:
            symbol: Stock symbol
            data: Historical price data with 'Close' column
            
        Returns:
            Signal dictionary with risk assessment
        """
        if len(data) < 30:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'Insufficient data for risk analysis'
            }
        
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        
        # Calculate volatility (annualized)
        volatility = data['Returns'].std() * np.sqrt(252)
        
        # Calculate maximum drawdown
        cumulative = (1 + data['Returns']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate recent volatility trend (last 20 days vs last 60 days)
        recent_vol = data['Returns'].tail(20).std() * np.sqrt(252)
        medium_vol = data['Returns'].tail(60).std() * np.sqrt(252)
        
        # Sharpe ratio approximation (assuming risk-free rate of 0)
        avg_return = data['Returns'].mean() * 252
        sharpe = avg_return / volatility if volatility > 0 else 0
        
        # Generate signal based on risk metrics
        signal = 'HOLD'
        confidence = 0.5
        reason = ''
        
        # High risk - suggest selling or avoiding
        if volatility > self.max_volatility:
            signal = 'SELL'
            confidence = 0.7
            reason = f'High risk: Volatility {volatility:.1%} exceeds max {self.max_volatility:.1%}'
        
        # Very high drawdown - suggest selling
        elif max_drawdown < -0.20:
            signal = 'SELL'
            confidence = 0.8
            reason = f'Severe drawdown: {max_drawdown:.1%} from peak'
        
        # Increasing volatility - caution
        elif recent_vol > medium_vol * 1.5:
            signal = 'SELL'
            confidence = 0.6
            reason = 'Rapidly increasing volatility suggests caution'
        
        # Low risk with positive returns - good buy
        elif volatility < self.max_volatility * 0.5 and sharpe > 0.5:
            signal = 'BUY'
            confidence = 0.7
            reason = f'Low risk with positive returns: Vol={volatility:.1%}, Sharpe={sharpe:.2f}'
        
        # Moderate risk with good risk-adjusted returns
        elif sharpe > 1.0:
            signal = 'BUY'
            confidence = 0.6
            reason = f'Good risk-adjusted returns: Sharpe={sharpe:.2f}'
        
        # Low volatility, stable
        elif volatility < self.max_volatility * 0.7:
            signal = 'HOLD'
            confidence = 0.7
            reason = f'Moderate risk level: Vol={volatility:.1%}'
        
        else:
            reason = f'Moderate risk: Vol={volatility:.1%}, Drawdown={max_drawdown:.1%}'
        
        result = {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'indicators': {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'recent_volatility': recent_vol
            }
        }
        
        self.signals[symbol] = result
        return result
