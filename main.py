"""Main entry point for the Swiss Trading Agent system."""

import argparse
from swiss_trading_agent import (
    MomentumAgent,
    ValueAgent,
    RiskAgent,
    AgentCoordinator
)
from swiss_trading_agent.data import SMIDataFetcher
from swiss_trading_agent.utils import Config


def main():
    """Run the multi-agent trading system."""
    parser = argparse.ArgumentParser(
        description='Swiss Trading Agent - Multi-agent stock trading system for SMI'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to analyze (default: all SMI stocks)'
    )
    parser.add_argument(
        '--period',
        default=Config.DEFAULT_PERIOD,
        help='Data period (default: 1y)'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=5,
        help='Number of top recommendations to show (default: 5)'
    )
    parser.add_argument(
        '--detail',
        nargs='+',
        help='Show detailed analysis for specific symbols'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Swiss Trading Agent - Multi-Agent Stock Trading System")
    print("=" * 70)
    print("\nInitializing agents...")
    
    # Create agents
    momentum_agent = MomentumAgent(
        weight=Config.MOMENTUM_AGENT_WEIGHT,
        short_window=Config.MOMENTUM_SHORT_WINDOW,
        long_window=Config.MOMENTUM_LONG_WINDOW
    )
    
    value_agent = ValueAgent(
        weight=Config.VALUE_AGENT_WEIGHT,
        lookback_period=Config.VALUE_LOOKBACK_PERIOD
    )
    
    risk_agent = RiskAgent(
        weight=Config.RISK_AGENT_WEIGHT,
        max_volatility=Config.MAX_VOLATILITY
    )
    
    # Create coordinator
    coordinator = AgentCoordinator([momentum_agent, value_agent, risk_agent])
    
    print(f"✓ Momentum Agent (weight: {Config.MOMENTUM_AGENT_WEIGHT})")
    print(f"✓ Value Agent (weight: {Config.VALUE_AGENT_WEIGHT})")
    print(f"✓ Risk Agent (weight: {Config.RISK_AGENT_WEIGHT})")
    
    # Fetch stock list
    data_fetcher = SMIDataFetcher()
    all_stocks = data_fetcher.get_stocks()
    
    # Determine which stocks to analyze
    if args.symbols:
        # Validate symbols
        symbols = []
        for s in args.symbols:
            if s in all_stocks:
                symbols.append(s)
            else:
                print(f"Warning: {s} is not a valid SMI symbol")
        if not symbols:
            print("No valid symbols provided. Using all SMI stocks.")
            symbols = None
    else:
        symbols = None
    
    print(f"\nAnalyzing {len(symbols) if symbols else len(all_stocks)} stocks...")
    print("-" * 70)
    
    # Analyze portfolio
    results = coordinator.analyze_portfolio(symbols)
    
    # Show top recommendations
    print("\n" + "=" * 70)
    print(f"TOP {args.top} BUY RECOMMENDATIONS")
    print("=" * 70)
    
    top_buys = coordinator.get_top_recommendations(n=args.top, signal_type='BUY')
    if top_buys:
        for i, rec in enumerate(top_buys, 1):
            company = all_stocks.get(rec['symbol'], rec['symbol'])
            print(f"\n{i}. {company} ({rec['symbol']})")
            print(f"   Confidence: {rec['confidence']:.1%}")
            print(f"   Price: ${rec['current_price']:.2f}" if rec['current_price'] else "   Price: N/A")
            print(f"   Reason: {rec['reason']}")
    else:
        print("\nNo BUY recommendations at this time.")
    
    print("\n" + "=" * 70)
    print(f"TOP {args.top} SELL RECOMMENDATIONS")
    print("=" * 70)
    
    top_sells = coordinator.get_top_recommendations(n=args.top, signal_type='SELL')
    if top_sells:
        for i, rec in enumerate(top_sells, 1):
            company = all_stocks.get(rec['symbol'], rec['symbol'])
            print(f"\n{i}. {company} ({rec['symbol']})")
            print(f"   Confidence: {rec['confidence']:.1%}")
            print(f"   Price: ${rec['current_price']:.2f}" if rec['current_price'] else "   Price: N/A")
            print(f"   Reason: {rec['reason']}")
    else:
        print("\nNo SELL recommendations at this time.")
    
    # Show detailed analysis if requested
    if args.detail:
        for symbol in args.detail:
            if symbol in results:
                coordinator.print_analysis(symbol)
            else:
                print(f"\nNo analysis available for {symbol}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
