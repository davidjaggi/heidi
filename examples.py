"""
Example usage of the Swiss Trading Agent system.

This script demonstrates various ways to use the multi-agent trading system.
"""

from swiss_trading_agent import (
    MomentumAgent,
    ValueAgent,
    RiskAgent,
    AgentCoordinator
)
from swiss_trading_agent.data import SMIDataFetcher


def example_basic_analysis():
    """Example: Basic stock analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Stock Analysis")
    print("="*70)
    
    # Create agents
    momentum = MomentumAgent(weight=1.0)
    value = ValueAgent(weight=1.0)
    risk = RiskAgent(weight=1.2)
    
    # Create coordinator
    coordinator = AgentCoordinator([momentum, value, risk])
    
    # Analyze a single stock
    result = coordinator.analyze_stock('NESN.SW')
    
    print(f"\nStock: {result['symbol']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Reason: {result['reason']}")
    
    # Show individual agent signals
    print("\nAgent Signals:")
    for signal in result['agent_signals']:
        print(f"  {signal['agent']}: {signal['signal']} ({signal['confidence']:.1%})")


def example_portfolio_analysis():
    """Example: Analyze multiple stocks."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Portfolio Analysis")
    print("="*70)
    
    # Create coordinator with default agents
    coordinator = AgentCoordinator([
        MomentumAgent(),
        ValueAgent(),
        RiskAgent()
    ])
    
    # Analyze specific stocks
    symbols = ['NESN.SW', 'NOVN.SW', 'ROG.SW', 'UBSG.SW']
    results = coordinator.analyze_portfolio(symbols)
    
    print(f"\nAnalyzed {len(results)} stocks")
    
    # Show summary
    for symbol, result in results.items():
        print(f"\n{symbol}: {result['recommendation']} ({result['confidence']:.1%})")
        print(f"  {result['reason']}")


def example_top_recommendations():
    """Example: Get top buy and sell recommendations."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Top Recommendations")
    print("="*70)
    
    coordinator = AgentCoordinator([
        MomentumAgent(),
        ValueAgent(),
        RiskAgent()
    ])
    
    # Analyze a subset of stocks for speed
    symbols = ['NESN.SW', 'NOVN.SW', 'ROG.SW', 'UBSG.SW', 'ABBN.SW']
    coordinator.analyze_portfolio(symbols)
    
    # Get top buy recommendations
    top_buys = coordinator.get_top_recommendations(n=3, signal_type='BUY')
    
    print("\nTop 3 BUY Recommendations:")
    for i, rec in enumerate(top_buys, 1):
        print(f"{i}. {rec['symbol']}: {rec['confidence']:.1%}")
    
    # Get top sell recommendations
    top_sells = coordinator.get_top_recommendations(n=3, signal_type='SELL')
    
    print("\nTop 3 SELL Recommendations:")
    for i, rec in enumerate(top_sells, 1):
        print(f"{i}. {rec['symbol']}: {rec['confidence']:.1%}")


def example_custom_agents():
    """Example: Using custom agent configurations."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Agent Configuration")
    print("="*70)
    
    # Create agents with custom parameters
    aggressive_momentum = MomentumAgent(
        weight=1.5,  # Higher weight
        short_window=10,  # Shorter windows for faster signals
        long_window=30
    )
    
    conservative_risk = RiskAgent(
        weight=2.0,  # Much higher weight on risk
        max_volatility=0.20  # Lower volatility threshold
    )
    
    value = ValueAgent(
        weight=1.0,
        lookback_period=200  # Longer lookback for value
    )
    
    # Create coordinator
    coordinator = AgentCoordinator([aggressive_momentum, value, conservative_risk])
    
    # Analyze
    result = coordinator.analyze_stock('NOVN.SW')
    
    print(f"\nWith custom configuration:")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Confidence: {result['confidence']:.1%}")


def example_data_fetcher():
    """Example: Direct use of data fetcher."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Using Data Fetcher Directly")
    print("="*70)
    
    fetcher = SMIDataFetcher()
    
    # Get list of all SMI stocks
    stocks = fetcher.get_stocks()
    print(f"\nTotal SMI stocks: {len(stocks)}")
    print("\nFirst 5 stocks:")
    for symbol, name in list(stocks.items())[:5]:
        print(f"  {symbol}: {name}")
    
    # Fetch data for a single stock
    data = fetcher.fetch_data('NESN.SW', period='1mo')
    if data is not None:
        print(f"\nNestlé data (last 5 days):")
        print(data[['Close', 'Volume']].tail())
    
    # Get current price
    current_price = fetcher.get_current_price('NESN.SW')
    if current_price:
        print(f"\nCurrent Nestlé price: ${current_price:.2f}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Swiss Trading Agent - Example Usage")
    print("="*70)
    
    try:
        example_basic_analysis()
        example_portfolio_analysis()
        example_top_recommendations()
        example_custom_agents()
        example_data_fetcher()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
