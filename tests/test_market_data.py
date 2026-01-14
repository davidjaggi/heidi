import pytest
import pandas as pd
from heidi.tools.market_data import (
    get_ticker_info, 
    get_price_history, 
    get_news, 
    get_technical_indicators, 
    get_full_analysis_data
)

@pytest.fixture
def test_ticker():
    return "NESN.SW"

def test_get_ticker_info(test_ticker):
    info = get_ticker_info(test_ticker)
    assert isinstance(info, dict)
    assert info["long_name"] == "Nestlé S.A."
    assert "market_cap" in info
    assert "current_price" in info
    assert info["currency"] == "CHF"
    assert "target_mean" in info

def test_get_price_history(test_ticker):
    history = get_price_history(test_ticker, period="1mo")
    assert isinstance(history, str)
    assert "Weekly Closing Prices (Last 1 Year):" in history or "No price history available." not in history
    # The string starts with a header, check if dates/prices follow
    lines = history.split("\n")
    assert len(lines) > 1

def test_get_news(test_ticker):
    news = get_news(test_ticker, max_items=2)
    
    # Debugging prints
    print(f"\nFetched {len(news)} news items for {test_ticker}")
    if len(news) > 0:
        print(f"Sample Item: {news[0]}")

    assert isinstance(news, list)
    if len(news) > 0:
        assert "title" in news[0]
        assert "summary" in news[0]

def test_get_technical_indicators(test_ticker):
    tech = get_technical_indicators(test_ticker)
    assert isinstance(tech, dict)
    if "error" not in tech:
        assert "rsi_14" in tech
        assert "macd" in tech
        assert "sma_50" in tech
        assert "sma_200" in tech
        assert "bb_upper" in tech
        assert "atr_14" in tech
        
        # Verify they are numbers or None
        for key in ["rsi_14", "macd", "sma_50", "sma_200", "bb_upper", "atr_14"]:
            val = tech[key]
            assert val is None or isinstance(val, (int, float))
    else:
        pytest.skip(f"Insufficient history for {test_ticker}: {tech['error']}")

def test_get_full_analysis_data(test_ticker):
    data = get_full_analysis_data(test_ticker)
    assert isinstance(data, dict)
    assert "info" in data
    assert "history" in data
    assert "news" in data
    assert "technical_indicators" in data
    assert data["info"]["long_name"] == "Nestlé S.A."
