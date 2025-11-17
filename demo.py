"""
Demo script for Swiss Trading Agent with synthetic data.

This script demonstrates the multi-agent system using synthetic stock data,
useful for testing without internet connectivity.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from swiss_trading_agent import (
    MomentumAgent,
    ValueAgent,
    RiskAgent,
    AgentCoordinator
)


def generate_synthetic_stock_data(
    days: int = 252,
    start_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0002
) -> pd.DataFrame:
    """
    Generate synthetic stock price data for testing.
    
    Args:
        days: Number of trading days
        start_price: Starting price
        volatility: Daily volatility (standard deviation)
        trend: Daily trend (mean return)
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    # Generate dates
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='D')
    
    # Generate returns with trend
    returns = np.random.normal(trend, volatility, days)
    
    # Generate prices
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    data = pd.DataFrame(index=dates)
    data['Close'] = prices
    
    # High/Low based on close with some noise
    daily_range = prices * 0.02  # 2% daily range
    data['High'] = prices + np.random.uniform(0, 1, days) * daily_range
    data['Low'] = prices - np.random.uniform(0, 1, days) * daily_range
    data['Open'] = prices - np.random.uniform(-0.5, 0.5, days) * daily_range
    
    # Ensure High >= Close, Low <= Close
    data['High'] = data[['High', 'Close', 'Open']].max(axis=1)
    data['Low'] = data[['Low', 'Close', 'Open']].min(axis=1)
    
    # Generate volume
    data['Volume'] = np.random.randint(1000000, 5000000, days)
    
    return data


def demo_basic_agent_analysis():
    """Demonstrate individual agent analysis."""
    print("\n" + "="*70)
    print("DEMO 1: Individual Agent Analysis")
    print("="*70)
    
    # Generate synthetic data for a stock
    data = generate_synthetic_stock_data(
        days=252,
        start_price=100.0,
        volatility=0.015,
        trend=0.001  # Slight upward trend
    )
    
    print(f"\nGenerated {len(data)} days of synthetic stock data")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"Current price: ${data['Close'].iloc[-1]:.2f}")
    
    # Create agents
    momentum_agent = MomentumAgent()
    value_agent = ValueAgent()
    risk_agent = RiskAgent()
    
    # Analyze with each agent
    print("\n--- Momentum Agent Analysis ---")
    momentum_signal = momentum_agent.analyze('DEMO.STOCK', data.copy())
    print(f"Signal: {momentum_signal['signal']}")
    print(f"Confidence: {momentum_signal['confidence']:.1%}")
    print(f"Reason: {momentum_signal['reason']}")
    
    print("\n--- Value Agent Analysis ---")
    value_signal = value_agent.analyze('DEMO.STOCK', data.copy())
    print(f"Signal: {value_signal['signal']}")
    print(f"Confidence: {value_signal['confidence']:.1%}")
    print(f"Reason: {value_signal['reason']}")
    
    print("\n--- Risk Agent Analysis ---")
    risk_signal = risk_agent.analyze('DEMO.STOCK', data.copy())
    print(f"Signal: {risk_signal['signal']}")
    print(f"Confidence: {risk_signal['confidence']:.1%}")
    print(f"Reason: {risk_signal['reason']}")


def demo_coordinator():
    """Demonstrate agent coordination and aggregation."""
    print("\n" + "="*70)
    print("DEMO 2: Multi-Agent Coordination")
    print("="*70)
    
    # Create coordinator
    coordinator = AgentCoordinator([
        MomentumAgent(weight=1.0),
        ValueAgent(weight=1.0),
        RiskAgent(weight=1.2)
    ])
    
    # Generate synthetic data for multiple "stocks"
    stocks = {
        'BULLISH.STOCK': generate_synthetic_stock_data(
            days=252, volatility=0.015, trend=0.002
        ),
        'BEARISH.STOCK': generate_synthetic_stock_data(
            days=252, volatility=0.015, trend=-0.002
        ),
        'VOLATILE.STOCK': generate_synthetic_stock_data(
            days=252, volatility=0.035, trend=0.0
        ),
    }
    
    print(f"\nAnalyzing {len(stocks)} synthetic stocks...\n")
    
    # Analyze each stock
    for symbol, data in stocks.items():
        result = coordinator.analyze_stock(symbol, data)
        
        print(f"\n{symbol}:")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Price: ${result['current_price']:.2f}")
        print(f"  Reason: {result['reason']}")
        
        # Show agent breakdown
        print("  Agent Signals:")
        for signal in result['agent_signals']:
            print(f"    - {signal['agent']}: {signal['signal']} ({signal['confidence']:.0%})")


def demo_different_market_conditions():
    """Demonstrate system behavior under different market conditions."""
    print("\n" + "="*70)
    print("DEMO 3: Different Market Conditions")
    print("="*70)
    
    coordinator = AgentCoordinator([
        MomentumAgent(),
        ValueAgent(),
        RiskAgent()
    ])
    
    conditions = {
        'Strong Uptrend': {
            'volatility': 0.015,
            'trend': 0.003,
            'start_price': 100
        },
        'Strong Downtrend': {
            'volatility': 0.015,
            'trend': -0.003,
            'start_price': 100
        },
        'High Volatility': {
            'volatility': 0.040,
            'trend': 0.0,
            'start_price': 100
        },
        'Stable/Sideways': {
            'volatility': 0.008,
            'trend': 0.0,
            'start_price': 100
        },
    }
    
    print("\nAnalyzing different market conditions:\n")
    
    for condition_name, params in conditions.items():
        data = generate_synthetic_stock_data(days=252, **params)
        result = coordinator.analyze_stock(condition_name, data)
        
        print(f"{condition_name}:")
        print(f"  → {result['recommendation']} (confidence: {result['confidence']:.0%})")
        print(f"  → {result['reason']}")
        print()


def demo_agent_weights():
    """Demonstrate impact of different agent weights."""
    print("\n" + "="*70)
    print("DEMO 4: Impact of Agent Weights")
    print("="*70)
    
    # Generate data with moderate characteristics
    data = generate_synthetic_stock_data(
        days=252,
        volatility=0.025,
        trend=0.001
    )
    
    configurations = [
        {
            'name': 'Balanced Weights',
            'weights': {'momentum': 1.0, 'value': 1.0, 'risk': 1.0}
        },
        {
            'name': 'Risk-Focused',
            'weights': {'momentum': 0.5, 'value': 0.5, 'risk': 2.0}
        },
        {
            'name': 'Momentum-Focused',
            'weights': {'momentum': 2.0, 'value': 0.5, 'risk': 0.5}
        },
    ]
    
    print("\nSame stock, different agent weight configurations:\n")
    
    for config in configurations:
        coordinator = AgentCoordinator([
            MomentumAgent(weight=config['weights']['momentum']),
            ValueAgent(weight=config['weights']['value']),
            RiskAgent(weight=config['weights']['risk'])
        ])
        
        result = coordinator.analyze_stock('TEST.STOCK', data.copy())
        
        print(f"{config['name']}:")
        print(f"  Weights - M:{config['weights']['momentum']}, "
              f"V:{config['weights']['value']}, R:{config['weights']['risk']}")
        print(f"  → {result['recommendation']} (confidence: {result['confidence']:.0%})")
        print()


def demo_statistics():
    """Show statistical analysis of agent performance."""
    print("\n" + "="*70)
    print("DEMO 5: Agent Statistics")
    print("="*70)
    
    coordinator = AgentCoordinator([
        MomentumAgent(),
        ValueAgent(),
        RiskAgent()
    ])
    
    # Generate multiple stocks
    num_stocks = 20
    recommendations = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    
    print(f"\nAnalyzing {num_stocks} random synthetic stocks...\n")
    
    for i in range(num_stocks):
        # Random parameters
        volatility = np.random.uniform(0.01, 0.04)
        trend = np.random.uniform(-0.002, 0.002)
        
        data = generate_synthetic_stock_data(
            days=252,
            volatility=volatility,
            trend=trend
        )
        
        result = coordinator.analyze_stock(f'STOCK_{i}', data)
        recommendations[result['recommendation']] += 1
    
    print("Recommendation Distribution:")
    for rec_type, count in recommendations.items():
        percentage = (count / num_stocks) * 100
        print(f"  {rec_type}: {count} ({percentage:.0f}%)")
    
    print(f"\nTotal analyzed: {num_stocks} stocks")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("Swiss Trading Agent - System Demonstration")
    print("Using Synthetic Data")
    print("="*70)
    
    try:
        demo_basic_agent_analysis()
        demo_coordinator()
        demo_different_market_conditions()
        demo_agent_weights()
        demo_statistics()
        
        print("\n" + "="*70)
        print("All demonstrations completed successfully!")
        print("="*70)
        print("\nThe multi-agent system is working correctly.")
        print("Each agent analyzes data independently and the coordinator")
        print("aggregates their signals to make final recommendations.")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
