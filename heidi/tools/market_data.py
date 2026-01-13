import yfinance as yf
from typing import Dict, Any, List
from datetime import datetime, timedelta

def get_ticker_info(ticker_symbol: str) -> Dict[str, Any]:
    """
    Fetches fundamental info for a ticker.
    """
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    
    # Map yfinance keys to our schema needs where possible, catch missing data
    return {
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "peg_ratio": info.get("pegRatio"),
        "dividend_yield": info.get("dividendYield"),
        "eps": info.get("trailingEps"),
        "profit_margin": info.get("profitMargins"),
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow"),
        "current_price": info.get("currentPrice") or info.get("ask") or info.get("bid"), # Fallback
        "currency": info.get("currency", "CHF"),
        "long_name": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry")
    }

def get_price_history(ticker_symbol: str, period: str = "1y") -> str:
    """
    Fetches price history and returns a summary string or CSV-like string for the LLM.
    We'll return a condensed text representation to save context window.
    """
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period)
    
    if hist.empty:
        return "No price history available."
    
    # Resample to weekly to save tokens if it's 1y data
    weekly = hist['Close'].resample('W').last()
    
    # Creating a simple string representation
    # Date: Price
    history_str = "Weekly Closing Prices (Last 1 Year):\n"
    for date, price in weekly.items():
        history_str += f"{date.strftime('%Y-%m-%d')}: {price:.2f}\n"
        
    return history_str

def get_news(ticker_symbol: str, max_items: int = 5) -> List[Dict[str, str]]:
    """
    Fetches latest news for the ticker.
    """
    ticker = yf.Ticker(ticker_symbol)
    news = ticker.news
    
    results = []
    for item in news[:max_items]:
        results.append({
            "title": item.get("title"),
            "publisher": item.get("publisher"),
            "link": item.get("link"),
            "publish_time": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime('%Y-%m-%d')
        })
    return results

def get_full_analysis_data(ticker_symbol: str) -> Dict[str, Any]:
    """
    Aggregates all data for the agent.
    """
    return {
        "info": get_ticker_info(ticker_symbol),
        "history": get_price_history(ticker_symbol),
        "news": get_news(ticker_symbol)
    }
