import yfinance as yf
import pandas as pd
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
        "industry": info.get("industry"),
        "target_low": info.get("targetLowPrice"),
        "target_mean": info.get("targetMeanPrice"),
        "target_high": info.get("targetHighPrice"),
        "recommendation_mean": info.get("recommendationMean"),
        "recommendation_key": info.get("recommendationKey"),
        "number_of_analysts": info.get("numberOfAnalystOpinions")
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



def get_technical_indicators(ticker_symbol: str) -> Dict[str, Any]:
    """
    Fetches technical indicators for the ticker.
    Indicators are rsi_14, macd, macd_signal, sma_50, sma_200, bollinger_bands, atr_14.
    """
    ticker = yf.Ticker(ticker_symbol)
    # We need enough data for SMA 200, so we fetch 2y by default
    hist = ticker.history(period="2y")

    if hist.empty or len(hist) < 20: # BB needs at least 20
        return {"error": "Insufficient price history available for technical indicators."}
    
    close_prices = hist['Close']
    high_prices = hist['High']
    low_prices = hist['Low']

    # RSI 14
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    exp1 = close_prices.ewm(span=12, adjust=False).mean()
    exp2 = close_prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    # SMAs
    sma_50 = close_prices.rolling(window=50).mean()
    sma_200 = close_prices.rolling(window=200).mean()

    # Bollinger Bands (20-day)
    bb_middle = close_prices.rolling(window=20).mean()
    bb_std = close_prices.rolling(window=20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)

    # ATR (14-day)
    prev_close = close_prices.shift(1)
    tr = pd.concat([
        high_prices - low_prices,
        (high_prices - prev_close).abs(),
        (low_prices - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()

    # Get the latest values
    latest_rsi = rsi.iloc[-1]
    latest_macd = macd.iloc[-1]
    latest_macd_signal = macd_signal.iloc[-1]
    latest_sma_50 = sma_50.iloc[-1]
    latest_sma_200 = sma_200.iloc[-1]
    latest_bb_upper = bb_upper.iloc[-1]
    latest_bb_middle = bb_middle.iloc[-1]
    latest_bb_lower = bb_lower.iloc[-1]
    latest_atr = atr.iloc[-1]

    return {
        "rsi_14": round(latest_rsi, 2) if not pd.isna(latest_rsi) else None,
        "macd": round(latest_macd, 2) if not pd.isna(latest_macd) else None,
        "macd_signal": round(latest_macd_signal, 2) if not pd.isna(latest_macd_signal) else None,
        "sma_50": round(latest_sma_50, 2) if not pd.isna(latest_sma_50) else None,
        "sma_200": round(latest_sma_200, 2) if not pd.isna(latest_sma_200) else None,
        "bb_upper": round(latest_bb_upper, 2) if not pd.isna(latest_bb_upper) else None,
        "bb_middle": round(latest_bb_middle, 2) if not pd.isna(latest_bb_middle) else None,
        "bb_lower": round(latest_bb_lower, 2) if not pd.isna(latest_bb_lower) else None,
        "atr_14": round(latest_atr, 2) if not pd.isna(latest_atr) else None
    }



def get_full_analysis_data(ticker_symbol: str) -> Dict[str, Any]:
    """
    Aggregates all data for the agent.
    """
    return {
        "info": get_ticker_info(ticker_symbol),
        "history": get_price_history(ticker_symbol),
        "news": get_news(ticker_symbol),
        "technical_indicators": get_technical_indicators(ticker_symbol)
    }