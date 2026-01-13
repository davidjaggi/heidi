# Heidi

A multi-agent system powered by LLMs (Gemini or Claude) to analyze the Swiss Market Index (SMI) on a weekly basis. The system deploys autonomous analyst agents for each constituent stock and a portfolio manager agent to recommend an optimal portfolio allocation.

## Features

- **Multi-Agent Architecture (LangGraph)**: 
  - **Analyst Agents**: Parallel execution of agents for each SMI stock.
  - **Portfolio Manager**: Aggregates reports and optimizes allocation.
  - **State Management**: Uses `LangGraph` for robust state handling.
- **Data Sources**: 
  - Market data via `yfinance`.
  - News headlines for sentiment.
- **CLI**: Built with `Typer` and `Rich` for a beautiful terminal experience.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/davidjaggi/heidi.git
   cd heidi
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_key_here
   ANTHROPIC_API_KEY=your_claude_key_here
   ```

## Usage

The application is run as a python module.

### Basic Run
Run the full analysis on all SMI stocks (defined in `data/tickers.txt`):
```bash
python -m cli.main
```

### Options
- `--tickers`: Path to ticker file (default: `data/tickers.txt`)
- `--model`: LLM provider, `gemini` or `claude` (default: `gemini`)
- `--model-name`: Specific model version (e.g., `gemini-1.5-flash-001`, `claude-3-5-sonnet-20240620`)
- `--output`: Directory to save JSON reports (default: `reports/`)

**Example**:
```bash
# Run a quick test with a subset of tickers
python -m cli.main --tickers data/test_tickers.txt --model gemini
```

## Output

Reports are generated in the `reports/` folder:
- **`{TICKER}.json`**: Detailed analysis for each stock including recommendation, confidence score, drivers, and risks.
- **`portfolio.json`**: Final portfolio allocation with weights and reasoning.
