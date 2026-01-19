![Cow Banner](assets/assets/Gemini_Generated_Image_qbdawtqbdawtqbda.png)

# Heidi

A multi-agent system powered by LLMs (Gemini, Claude, or OpenAI) to analyze the Swiss market. The system deploys autonomous analyst agents for each constituent stock and a lead portfolio manager agent to recommend an optimal portfolio allocation.

> [!TIP]
> **Centralized Configuration**: All system settings, including model choices, tickers, and providers, can be managed in default_config.py.

## Features

- **Multi-Agent Architecture (LangGraph)**:
  - **Stock Analyst Agents**: Parallel execution of agents for each stock ticker.
  - **Report Reviewer**: Reviews analyst reports for quality, requesting revisions if needed (up to 2 revision cycles).
  - **Portfolio Manager**: Aggregates analyst reports and optimizes allocation (fully invested, no cash position).
  - **State Management**: Orchestrated via `LangGraph` for robust concurrent execution.
- **Enhanced Observability**:
  - **Nuanced Logging**: Custom `HeidiCallbackHandler` for detailed step-by-step terminal output.
- **Data Sources**:
  - Market data, price history, and news headlines via `yfinance`.
- **Beautiful CLI**: Built with `Typer` and `Rich` for a premium terminal experience.

## Architecture

```
START → Stock Analysts (parallel) → Report Reviewer → Portfolio Manager → END
                    ↑                      |
                    └──────────────────────┘ (revision loop if needed)
```

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
   OPENAI_API_KEY=your_openai_key_here
   
   # Optional: LangSmith Tracing
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langchain_key_here
   LANGCHAIN_PROJECT=heidi
   ```

## Usage

The application is run as a python module.

### Basic Run
Run the full analysis using the defaults in `default_config.py`:
```bash
python -m cli.main
```

### Command Line Overrides
While you can edit `default_config.py`, you can also override settings via CLI:
- `--tickers`: Path to ticker file.
- `--model`: LLM provider (`gemini`, `anthropic`, `openai`).
- `--model-name`: Specific model version.
- `--output`: Directory to save reports.

**Example**:
```bash
python -m cli.main --tickers data/test_tickers.txt --model gemini
```

## Output

Each run generates a timestamped subfolder in `reports/` containing:
- **`README.md`**: A comprehensive Markdown summary of the entire run.
- **`{TICKER}.json`**: Raw structured analysis for each stock.
- **`portfolio.json`**: Raw structured portfolio allocation data.
- **`graph.png`**: A visualization of the agent workflow.

## Disclaimer

This project is intended solely for educational and research purposes.

- It is not designed for real trading or investment use.
- No warranties or guarantees are provided.
- The creator bears no responsibility for any financial losses.
- By using this software, you acknowledge and agree that it is for learning purposes only.
