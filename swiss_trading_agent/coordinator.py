"""Agent coordinator for managing multiple trading agents."""

from typing import Dict, List, Optional
import pandas as pd
from .agents.base_agent import BaseAgent
from .data.smi_data import SMIDataFetcher


class AgentCoordinator:
    """
    Coordinates multiple trading agents to make collective decisions.
    
    Aggregates signals from different agents and produces final recommendations.
    """
    
    def __init__(self, agents: Optional[List[BaseAgent]] = None):
        """
        Initialize the coordinator.
        
        Args:
            agents: List of trading agents to coordinate
        """
        self.agents = agents or []
        self.data_fetcher = SMIDataFetcher()
        self.analysis_results = {}
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the coordinator."""
        self.agents.append(agent)
    
    def remove_agent(self, agent_name: str):
        """Remove an agent by name."""
        self.agents = [a for a in self.agents if a.name != agent_name]
    
    def analyze_stock(self, symbol: str, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyze a stock using all agents.
        
        Args:
            symbol: Stock symbol
            data: Historical data (fetched if not provided)
            
        Returns:
            Dictionary with aggregated analysis results
        """
        # Fetch data if not provided
        if data is None:
            data = self.data_fetcher.fetch_data(symbol)
            if data is None:
                return {
                    'symbol': symbol,
                    'recommendation': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Unable to fetch data',
                    'agent_signals': []
                }
        
        # Collect signals from all agents
        agent_signals = []
        for agent in self.agents:
            signal = agent.analyze(symbol, data.copy())
            agent_signals.append({
                'agent': agent.name,
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'reason': signal['reason'],
                'weight': agent.weight
            })
        
        # Aggregate signals
        recommendation = self._aggregate_signals(agent_signals)
        
        result = {
            'symbol': symbol,
            'recommendation': recommendation['signal'],
            'confidence': recommendation['confidence'],
            'reason': recommendation['reason'],
            'agent_signals': agent_signals,
            'current_price': data['Close'].iloc[-1] if not data.empty else None
        }
        
        self.analysis_results[symbol] = result
        return result
    
    def analyze_portfolio(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Analyze multiple stocks.
        
        Args:
            symbols: List of symbols to analyze (default: all SMI stocks)
            
        Returns:
            Dictionary mapping symbols to analysis results
        """
        if symbols is None:
            symbols = list(self.data_fetcher.get_stocks().keys())
        
        results = {}
        for symbol in symbols:
            print(f"Analyzing {symbol}...")
            result = self.analyze_stock(symbol)
            results[symbol] = result
        
        return results
    
    def _aggregate_signals(self, agent_signals: List[Dict]) -> Dict:
        """
        Aggregate signals from multiple agents into a single recommendation.
        
        Uses weighted voting based on agent confidence and weights.
        
        Args:
            agent_signals: List of agent signal dictionaries
            
        Returns:
            Aggregated recommendation
        """
        if not agent_signals:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'No agent signals'
            }
        
        # Calculate weighted scores
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        
        for signal in agent_signals:
            weight = signal['weight'] * signal['confidence']
            total_weight += signal['weight']
            
            if signal['signal'] == 'BUY':
                buy_score += weight
            elif signal['signal'] == 'SELL':
                sell_score += weight
        
        # Normalize scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine final recommendation
        threshold = 0.4  # Minimum score to trigger BUY/SELL
        
        if buy_score > sell_score and buy_score > threshold:
            signal = 'BUY'
            confidence = buy_score
            buy_agents = [s['agent'] for s in agent_signals if s['signal'] == 'BUY']
            reason = f"Majority BUY signal from: {', '.join(buy_agents)}"
        elif sell_score > buy_score and sell_score > threshold:
            signal = 'SELL'
            confidence = sell_score
            sell_agents = [s['agent'] for s in agent_signals if s['signal'] == 'SELL']
            reason = f"Majority SELL signal from: {', '.join(sell_agents)}"
        else:
            signal = 'HOLD'
            confidence = 1.0 - max(buy_score, sell_score)
            reason = "Mixed or weak signals from agents"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason
        }
    
    def get_top_recommendations(self, n: int = 5, signal_type: str = 'BUY') -> List[Dict]:
        """
        Get top N recommendations of a specific type.
        
        Args:
            n: Number of recommendations to return
            signal_type: Type of signal ('BUY' or 'SELL')
            
        Returns:
            List of top recommendations sorted by confidence
        """
        filtered = [
            r for r in self.analysis_results.values()
            if r['recommendation'] == signal_type
        ]
        
        sorted_results = sorted(
            filtered,
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        return sorted_results[:n]
    
    def print_analysis(self, symbol: str):
        """Print detailed analysis for a stock."""
        if symbol not in self.analysis_results:
            print(f"No analysis available for {symbol}")
            return
        
        result = self.analysis_results[symbol]
        company_name = self.data_fetcher.get_stocks().get(symbol, symbol)
        
        print(f"\n{'='*70}")
        print(f"Analysis for {company_name} ({symbol})")
        print(f"{'='*70}")
        print(f"Current Price: ${result['current_price']:.2f}" if result['current_price'] else "Price: N/A")
        print(f"\nFinal Recommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Reason: {result['reason']}")
        
        print(f"\n{'-'*70}")
        print("Agent Signals:")
        print(f"{'-'*70}")
        for signal in result['agent_signals']:
            print(f"\n{signal['agent']}:")
            print(f"  Signal: {signal['signal']}")
            print(f"  Confidence: {signal['confidence']:.1%}")
            print(f"  Reason: {signal['reason']}")
        print(f"{'='*70}\n")
