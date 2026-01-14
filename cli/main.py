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

from heidi.graph import create_graph
from heidi.default_config import DEFAULT_CONFIG

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO) # Suppress noisy logs, rely on Rich
logger = logging.getLogger("Heidi")

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

def generate_markdown_summary(output_dir: Path, reports: list, portfolio, timestamp: str, model_info: str):
    """
    Generates a summary.md for the run.
    """
    summary_path = output_dir / "summary.md"
    
    content = f"# Heidi Investment Run Summary - {timestamp}\n\n"
    content += f"**Model Configuration:** {model_info}\n\n"
    
    content += "## Individual Analyst Reports\n\n"
    content += "| Ticker | Company | Recommendation | Confidence | Key Drivers |\n"
    content += "| :--- | :--- | :--- | :--- | :--- |\n"
    for r in reports:
        drivers = ", ".join(r.key_drivers[:2])
        content += f"| {r.ticker} | {r.company} | **{r.recommendation.value}** | {r.confidence_score:.2f} | {drivers} |\n"
    
    content += "\n## Portfolio Recommendation\n\n"
    if portfolio:
        content += f"**Timestamp:** {portfolio.timestamp}\n\n"
        content += "| Ticker | Weight | Reasoning |\n"
        content += "| :--- | :--- | :--- |\n"
        for alloc in portfolio.allocations:
            content += f"| {alloc.ticker} | {alloc.weight:.2f} | {alloc.reasoning} |\n"
    else:
        content += "*No portfolio generated.*\n"
        
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)

@app.command()
def run(
    tickers_file: str = typer.Option(DEFAULT_CONFIG["tickers"], "--tickers", help="Path to tickers file"),
    output_dir: str = typer.Option("reports", "--output", help="Base output directory"),
    model: str = typer.Option(DEFAULT_CONFIG["llm_provider"], "--model", help="Model provider (gemini/claude/openai)"),
    model_name: str = typer.Option(None, "--model-name", help="Specific model name (optional)")
):
    """
    Run the Heidi multi-agent system.
    """
    from datetime import datetime
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
        
        # Create timestamped subfolder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = Path(output_dir) / timestamp
        
        # Save individual reports
        for r in reports:
            save_output(run_output_dir, r.model_dump(), f"{r.ticker}.json")
            
        # Save Portfolio
        if portfolio:
            save_output(run_output_dir, portfolio.model_dump(), "portfolio.json")
            
            # Generate Markdown Summary
            model_info = f"{model} ({model_name or 'default'})"
            generate_markdown_summary(run_output_dir, reports, portfolio, timestamp, model_info)
            
            # Display Portfolio Table
            table = Table(title=f"Recommended Portfolio")
            table.add_column("Ticker", style="cyan")
            table.add_column("Weight", justify="right")
            table.add_column("Reasoning", style="magenta")
            
            for alloc in portfolio.allocations:
                table.add_row(alloc.ticker, f"{alloc.weight:.2f}", alloc.reasoning)
                
            console.print(table)
            console.print(f"\n[green]All reports saved to {run_output_dir}[/green]")
        else:
            console.print("[red]No portfolio generated.[/red]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    app()
