import typer
import asyncio
import json
import logging
import os
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from typing import List

from src.graph import create_graph

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.ERROR) # Suppress noisy logs, rely on Rich
logger = logging.getLogger("SwissTradingAgent")

app = typer.Typer(no_args_is_help=True)
console = Console()

def load_tickers(ticker_path: str) -> List[str]:
    path = Path(ticker_path)
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {ticker_path}")
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def save_output(output_dir: Path, data: dict, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@app.command()
def run(
    tickers_file: str = typer.Option("data/tickers.txt", "--tickers", help="Path to tickers file"),
    output_dir: str = typer.Option("reports", "--output", help="Output directory"),
    model: str = typer.Option("gemini", "--model", help="Model provider (gemini/claude)"),
    model_name: str = typer.Option(None, "--model-name", help="Specific model name (optional)")
):
    """
    Run the Swiss AI Trader multi-agent system.
    """
    try:
        # 1. Load Data
        ticker_list = load_tickers(tickers_file)
        console.print(f"[bold green]Loaded {len(ticker_list)} tickers from {tickers_file}[/bold green]")

        # 2. Initialize Graph
        graph = create_graph()
        
        # 3. Execute Graph
        initial_state = {
            "tickers": ticker_list, 
            "reports": [], 
            "model_provider": model,
            "model_name": model_name
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(description="Running Analyst Agents...", total=None)
            
            final_state = asyncio.run(graph.ainvoke(initial_state))
            
            progress.update(task, completed=True)

        # 4. Results
        reports = final_state.get("reports", [])
        portfolio = final_state.get("portfolio")
        
        console.print(f"[bold blue]Generated {len(reports)} Analyst Reports[/bold blue]")
        
        # Save individual reports
        out_path = Path(output_dir)
        for r in reports:
            save_output(out_path, r.model_dump(), f"{r.ticker}.json")
            
        # Save Portfolio
        if portfolio:
            save_output(out_path, portfolio.model_dump(), "portfolio.json")
            
            # Display Portfolio Table
            table = Table(title=f"Recommended Portfolio")
            table.add_column("Ticker", style="cyan")
            table.add_column("Weight", justify="right")
            table.add_column("Reasoning", style="magenta")
            
            for alloc in portfolio.allocations:
                table.add_row(alloc.ticker, f"{alloc.weight:.2f}", alloc.reasoning)
                
            console.print(table)
            console.print(f"\n[green]All reports saved to {out_path}[/green]")
        else:
            console.print("[red]No portfolio generated.[/red]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    app()
