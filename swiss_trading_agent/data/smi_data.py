"""Data fetcher for Swiss Market Index (SMI) stocks."""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class SMIDataFetcher:
    """
    Fetches market data for Swiss Market Index stocks.
    
    The SMI consists of the 20 largest and most liquid stocks listed on the SIX Swiss Exchange.
    """
    
    # SMI constituent stocks (as of 2024)
    SMI_STOCKS = {
        'NESN.SW': 'Nestlé',
        'NOVN.SW': 'Novartis',
        'ROG.SW': 'Roche',
        'ABBN.SW': 'ABB',
        'UBSG.SW': 'UBS Group',
        'ZURN.SW': 'Zurich Insurance',
        'SREN.SW': 'Swiss Re',
        'HOLN.SW': 'Holcim',
        'SIKA.SW': 'Sika',
        'GIVN.SW': 'Givaudan',
        'LONN.SW': 'Lonza',
        'SLHN.SW': 'Swiss Life',
        'CFR.SW': 'Compagnie Financière Richemont',
        'GEBN.SW': 'Geberit',
        'ALC.SW': 'Alcon',
        'PGHN.SW': 'Partners Group',
        'SCMN.SW': 'Swisscom',
        'SGSN.SW': 'SGS',
        'UHR.SW': 'Swatch Group',
        'STMN.SW': 'Straumann'
    }
    
    def __init__(self):
        """Initialize the data fetcher."""
        self.cache = {}
    
    def get_stocks(self) -> Dict[str, str]:
        """
        Get dictionary of SMI stock symbols and names.
        
        Returns:
            Dictionary mapping symbols to company names
        """
        return self.SMI_STOCKS.copy()
    
    def fetch_data(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d",
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a stock.
        
        Args:
            symbol: Stock symbol (e.g., 'NESN.SW')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1d', '1wk', '1mo')
            force_refresh: Force refresh from API instead of using cache
            
        Returns:
            DataFrame with OHLCV data or None if fetch failed
        """
        cache_key = f"{symbol}_{period}_{interval}"
        
        if not force_refresh and cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"Warning: No data returned for {symbol}")
                return None
            
            # Cache the data
            self.cache[cache_key] = data.copy()
            return data
        
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_multiple(
        self,
        symbols: Optional[List[str]] = None,
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols: List of symbols to fetch (default: all SMI stocks)
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if symbols is None:
            symbols = list(self.SMI_STOCKS.keys())
        
        results = {}
        for symbol in symbols:
            data = self.fetch_data(symbol, period, interval)
            if data is not None:
                results[symbol] = data
        
        return results
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None if unavailable
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('currentPrice') or info.get('regularMarketPrice')
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache = {}
