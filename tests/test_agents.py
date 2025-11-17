"""
Basic tests for the Swiss Trading Agent system.

These tests verify core functionality using synthetic data.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from swiss_trading_agent import (
    MomentumAgent,
    ValueAgent,
    RiskAgent,
    AgentCoordinator
)
from swiss_trading_agent.data import SMIDataFetcher


class TestAgents(unittest.TestCase):
    """Test individual trading agents."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Generate synthetic test data
        np.random.seed(42)
        days = 100
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, days)))
        
        self.test_data = pd.DataFrame(index=dates)
        self.test_data['Close'] = prices
        self.test_data['High'] = prices * 1.02
        self.test_data['Low'] = prices * 0.98
        self.test_data['Open'] = prices
        self.test_data['Volume'] = np.random.randint(1000000, 5000000, days)
    
    def test_momentum_agent_initialization(self):
        """Test MomentumAgent can be initialized."""
        agent = MomentumAgent(weight=1.0)
        self.assertEqual(agent.name, "MomentumAgent")
        self.assertEqual(agent.weight, 1.0)
    
    def test_momentum_agent_analysis(self):
        """Test MomentumAgent produces valid analysis."""
        agent = MomentumAgent()
        result = agent.analyze('TEST', self.test_data.copy())
        
        self.assertIn('signal', result)
        self.assertIn(result['signal'], ['BUY', 'SELL', 'HOLD'])
        self.assertIn('confidence', result)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        self.assertIn('reason', result)
    
    def test_value_agent_initialization(self):
        """Test ValueAgent can be initialized."""
        agent = ValueAgent(weight=1.0)
        self.assertEqual(agent.name, "ValueAgent")
        self.assertEqual(agent.weight, 1.0)
    
    def test_value_agent_analysis(self):
        """Test ValueAgent produces valid analysis."""
        agent = ValueAgent()
        result = agent.analyze('TEST', self.test_data.copy())
        
        self.assertIn('signal', result)
        self.assertIn(result['signal'], ['BUY', 'SELL', 'HOLD'])
        self.assertIn('confidence', result)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_risk_agent_initialization(self):
        """Test RiskAgent can be initialized."""
        agent = RiskAgent(weight=1.0)
        self.assertEqual(agent.name, "RiskAgent")
        self.assertEqual(agent.weight, 1.0)
    
    def test_risk_agent_analysis(self):
        """Test RiskAgent produces valid analysis."""
        agent = RiskAgent()
        result = agent.analyze('TEST', self.test_data.copy())
        
        self.assertIn('signal', result)
        self.assertIn(result['signal'], ['BUY', 'SELL', 'HOLD'])
        self.assertIn('confidence', result)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)


class TestCoordinator(unittest.TestCase):
    """Test agent coordinator."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        days = 100
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, days)))
        
        self.test_data = pd.DataFrame(index=dates)
        self.test_data['Close'] = prices
        self.test_data['High'] = prices * 1.02
        self.test_data['Low'] = prices * 0.98
        self.test_data['Open'] = prices
        self.test_data['Volume'] = np.random.randint(1000000, 5000000, days)
    
    def test_coordinator_initialization(self):
        """Test coordinator can be initialized."""
        agents = [MomentumAgent(), ValueAgent(), RiskAgent()]
        coordinator = AgentCoordinator(agents)
        self.assertEqual(len(coordinator.agents), 3)
    
    def test_add_agent(self):
        """Test adding agents to coordinator."""
        coordinator = AgentCoordinator()
        self.assertEqual(len(coordinator.agents), 0)
        
        coordinator.add_agent(MomentumAgent())
        self.assertEqual(len(coordinator.agents), 1)
        
        coordinator.add_agent(ValueAgent())
        self.assertEqual(len(coordinator.agents), 2)
    
    def test_analyze_stock(self):
        """Test stock analysis with coordinator."""
        coordinator = AgentCoordinator([
            MomentumAgent(),
            ValueAgent(),
            RiskAgent()
        ])
        
        result = coordinator.analyze_stock('TEST', self.test_data.copy())
        
        self.assertIn('symbol', result)
        self.assertEqual(result['symbol'], 'TEST')
        self.assertIn('recommendation', result)
        self.assertIn(result['recommendation'], ['BUY', 'SELL', 'HOLD'])
        self.assertIn('confidence', result)
        self.assertIn('agent_signals', result)
        self.assertEqual(len(result['agent_signals']), 3)
    
    def test_get_top_recommendations(self):
        """Test getting top recommendations."""
        coordinator = AgentCoordinator([
            MomentumAgent(),
            ValueAgent(),
            RiskAgent()
        ])
        
        # Analyze some stocks
        coordinator.analyze_stock('TEST1', self.test_data.copy())
        coordinator.analyze_stock('TEST2', self.test_data.copy())
        
        # Get top buys
        top_buys = coordinator.get_top_recommendations(n=2, signal_type='BUY')
        self.assertIsInstance(top_buys, list)
        self.assertLessEqual(len(top_buys), 2)


class TestSMIDataFetcher(unittest.TestCase):
    """Test SMI data fetcher."""
    
    def test_get_stocks(self):
        """Test getting SMI stock list."""
        fetcher = SMIDataFetcher()
        stocks = fetcher.get_stocks()
        
        self.assertIsInstance(stocks, dict)
        self.assertEqual(len(stocks), 20)  # SMI has 20 constituents
        
        # Check some known stocks
        self.assertIn('NESN.SW', stocks)
        self.assertIn('NOVN.SW', stocks)
        self.assertIn('ROG.SW', stocks)
        self.assertEqual(stocks['NESN.SW'], 'Nestlé')
    
    def test_cache_functionality(self):
        """Test that caching works."""
        fetcher = SMIDataFetcher()
        
        # Initially cache should be empty
        self.assertEqual(len(fetcher.cache), 0)
        
        # Clear cache
        fetcher.clear_cache()
        self.assertEqual(len(fetcher.cache), 0)


if __name__ == '__main__':
    unittest.main()
