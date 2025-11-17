# SwissTradingAgent

A sophisticated multi-agent system for trading stocks in the Swiss Market Index (SMI). This system uses specialized AI agents that analyze different aspects of the market to provide comprehensive trading recommendations.

## Features

- **Multi-Agent Architecture**: Three specialized agents work together:
  - **Momentum Agent**: Analyzes price trends and momentum indicators
  - **Value Agent**: Evaluates stocks based on price levels and historical value
  - **Risk Agent**: Assesses volatility and risk-adjusted returns
  
- **Swiss Market Index Coverage**: Analyzes all 20 stocks in the SMI including:
  - Nestlé, Novartis, Roche
  - UBS, Zurich Insurance, Swiss Re
  - ABB, Holcim, and more

- **Intelligent Decision Making**: Agents vote on recommendations with weighted confidence scores

- **Flexible Analysis**: Analyze individual stocks or the entire portfolio

## Installation

1. Clone the repository:
```bash
git clone https://github.com/davidjaggi/SwissTradingAgent.git
cd SwissTradingAgent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

## Usage

### Basic Usage

Analyze all SMI stocks:
```bash
python main.py
```

### Analyze Specific Stocks

```bash
python main.py --symbols NESN.SW NOVN.SW ROG.SW
```

### Get Detailed Analysis

```bash
python main.py --detail NESN.SW --top 3
```

### Custom Time Period

```bash
python main.py --period 6mo --top 10
```

## Configuration

You can customize agent behavior by creating a `.env` file:

```bash
# Agent weights (higher = more influence)
MOMENTUM_AGENT_WEIGHT=1.0
VALUE_AGENT_WEIGHT=1.0
RISK_AGENT_WEIGHT=1.2

# Data parameters
DEFAULT_PERIOD=1y
DEFAULT_INTERVAL=1d

# Risk parameters
MAX_VOLATILITY=0.30

# Momentum parameters
MOMENTUM_SHORT_WINDOW=20
MOMENTUM_LONG_WINDOW=50

# Value parameters
VALUE_LOOKBACK_PERIOD=100
```

## Architecture

### Agents

Each agent implements the `BaseAgent` interface and provides:
- Independent analysis of market data
- Trading signals (BUY/SELL/HOLD)
- Confidence scores (0-1)
- Detailed reasoning

### Coordinator

The `AgentCoordinator` manages all agents:
- Fetches market data
- Distributes data to agents
- Aggregates signals using weighted voting
- Generates final recommendations

### Data Fetcher

The `SMIDataFetcher` provides:
- Real-time data from Yahoo Finance
- Historical price data
- Caching for performance
- Support for all SMI constituent stocks

## SMI Constituent Stocks

The system supports all 20 stocks in the Swiss Market Index:

| Symbol | Company |
|--------|---------|
| NESN.SW | Nestlé |
| NOVN.SW | Novartis |
| ROG.SW | Roche |
| ABBN.SW | ABB |
| UBSG.SW | UBS Group |
| ZURN.SW | Zurich Insurance |
| SREN.SW | Swiss Re |
| HOLN.SW | Holcim |
| SIKA.SW | Sika |
| GIVN.SW | Givaudan |
| LONN.SW | Lonza |
| SLHN.SW | Swiss Life |
| CFR.SW | Richemont |
| GEBN.SW | Geberit |
| ALC.SW | Alcon |
| PGHN.SW | Partners Group |
| SCMN.SW | Swisscom |
| SGSN.SW | SGS |
| UHR.SW | Swatch Group |
| STMN.SW | Straumann |

## Example Output

```
======================================================================
Swiss Trading Agent - Multi-Agent Stock Trading System
======================================================================

Initializing agents...
✓ Momentum Agent (weight: 1.0)
✓ Value Agent (weight: 1.0)
✓ Risk Agent (weight: 1.2)

Analyzing 20 stocks...
----------------------------------------------------------------------

======================================================================
TOP 5 BUY RECOMMENDATIONS
======================================================================

1. Nestlé (NESN.SW)
   Confidence: 75.3%
   Price: $102.45
   Reason: Majority BUY signal from: MomentumAgent, ValueAgent

2. Novartis (NOVN.SW)
   Confidence: 68.9%
   Price: $87.32
   Reason: Majority BUY signal from: ValueAgent, RiskAgent
...
```

## Programmatic Usage

You can also use the system programmatically:

```python
from swiss_trading_agent import (
    MomentumAgent, 
    ValueAgent, 
    RiskAgent, 
    AgentCoordinator
)

# Create agents
momentum = MomentumAgent(weight=1.0)
value = ValueAgent(weight=1.0)
risk = RiskAgent(weight=1.2)

# Create coordinator
coordinator = AgentCoordinator([momentum, value, risk])

# Analyze a stock
result = coordinator.analyze_stock('NESN.SW')
print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']:.1%}")

# Analyze portfolio
results = coordinator.analyze_portfolio()

# Get top buys
top_buys = coordinator.get_top_recommendations(n=5, signal_type='BUY')
```

## Requirements

- Python 3.8+
- pandas
- numpy
- yfinance
- requests
- python-dotenv

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This system is for educational and research purposes only. It does not constitute financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.