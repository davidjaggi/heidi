"""
Risk calculation utilities for portfolio stress testing.
Uses historical price data to compute VaR, CVaR, volatility, and other risk metrics.
"""
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def get_price_data(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """
    Fetches historical price data for multiple tickers.
    Returns DataFrame with adjusted close prices.
    """
    logger.info(f"Fetching price data for {tickers} over {period}")

    try:
        if len(tickers) == 1:
            # Single ticker - simpler handling
            ticker = yf.Ticker(tickers[0])
            hist = ticker.history(period=period)
            if hist.empty:
                logger.warning(f"No data returned for {tickers[0]}")
                return pd.DataFrame()
            prices = hist[['Close']]
            prices.columns = [tickers[0]]
            return prices

        # Multiple tickers
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False)

        if data.empty:
            logger.warning(f"No data returned for tickers: {tickers}")
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            # Multiple tickers: extract Close prices
            prices = data['Close']
        else:
            # Single column returned
            prices = data[['Close']] if 'Close' in data.columns else data
            if len(tickers) == 1:
                prices.columns = [tickers[0]]

        logger.info(f"Fetched {len(prices)} price records for {list(prices.columns)}")
        return prices

    except Exception as e:
        logger.error(f"Error fetching price data: {e}")
        return pd.DataFrame()


def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from price data."""
    if prices.empty:
        return pd.DataFrame()
    returns = prices.pct_change().dropna()
    # Replace infinite values with NaN, then drop
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def get_portfolio_returns(
    tickers: List[str],
    weights: List[float],
    period: str = "2y"
) -> pd.Series:
    """
    Calculate weighted portfolio returns.

    Args:
        tickers: List of ticker symbols
        weights: Corresponding weights (must sum to 1.0)
        period: Historical period (default 2y)

    Returns:
        Series of daily portfolio returns
    """
    prices = get_price_data(tickers, period)

    if prices.empty:
        logger.warning("No price data available, returning empty series")
        return pd.Series(dtype=float)

    returns = get_returns(prices)

    if returns.empty:
        logger.warning("No returns data calculated, returning empty series")
        return pd.Series(dtype=float)

    logger.info(f"Returns columns: {list(returns.columns)}, Tickers: {tickers}")

    # Align weights with available tickers (handle missing data)
    weight_dict = dict(zip(tickers, weights))
    available_weights = [weight_dict.get(col, 0) for col in returns.columns]

    logger.info(f"Available weights: {available_weights}")

    # Normalize weights to available tickers
    total_weight = sum(available_weights)
    if total_weight > 0:
        available_weights = [w / total_weight for w in available_weights]
    else:
        logger.warning("No matching tickers found in price data")
        return pd.Series(dtype=float)

    # Calculate weighted portfolio returns
    portfolio_returns = (returns * available_weights).sum(axis=1)

    logger.info(f"Portfolio returns: {len(portfolio_returns)} data points, mean={portfolio_returns.mean():.4f}")

    return portfolio_returns


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate parametric Value at Risk.

    Args:
        returns: Series of daily returns
        confidence: Confidence level (default 95%)

    Returns:
        VaR as a positive percentage (e.g., 0.02 = 2% daily VaR)
    """
    if returns.empty or len(returns) < 2:
        logger.warning("Insufficient data for VaR calculation")
        return 0.0

    mean = returns.mean()
    std = returns.std()

    if pd.isna(mean) or pd.isna(std):
        logger.warning("NaN values in VaR calculation")
        return 0.0

    # Z-score for confidence level
    z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z = z_scores.get(confidence, 1.645)

    # VaR = -(mean - z * std)
    var = -(mean - z * std)
    return float(var) if not pd.isna(var) else 0.0


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).

    CVaR is the expected loss given that loss exceeds VaR.

    Args:
        returns: Series of daily returns
        confidence: Confidence level (default 95%)

    Returns:
        CVaR as a positive percentage
    """
    if returns.empty or len(returns) < 2:
        return 0.0

    var = calculate_var(returns, confidence)

    # CVaR is the mean of returns below the VaR threshold
    threshold = -var  # VaR is positive, threshold is negative return
    tail_returns = returns[returns <= threshold]

    if len(tail_returns) == 0:
        # If no returns below VaR, use VaR as CVaR
        return var

    cvar = -tail_returns.mean()
    return float(cvar) if not pd.isna(cvar) else var


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from returns series.

    Maximum drawdown is the largest peak-to-trough decline.

    Args:
        returns: Series of daily returns

    Returns:
        Maximum drawdown as a positive percentage (e.g., 0.25 = 25%)
    """
    if returns.empty or len(returns) < 2:
        return 0.0

    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()

    # Calculate running maximum
    running_max = cum_returns.expanding().max()

    # Calculate drawdown at each point
    drawdowns = (cum_returns - running_max) / running_max

    # Maximum drawdown (most negative value)
    max_dd = drawdowns.min()

    result = float(-max_dd) if not pd.isna(max_dd) else 0.0
    return result


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Calculate portfolio volatility.

    Args:
        returns: Series of daily returns
        annualize: Whether to annualize (default True, assumes 252 trading days)

    Returns:
        Volatility as percentage
    """
    if returns.empty or len(returns) < 2:
        return 0.0

    vol = returns.std()

    if pd.isna(vol):
        return 0.0

    if annualize:
        vol = vol * np.sqrt(252)

    return float(vol)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculate Sharpe Ratio (risk-adjusted return).

    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Annualized Sharpe Ratio
    """
    if returns.empty or len(returns) < 2:
        return 0.0

    # Annualized return
    annual_return = returns.mean() * 252

    # Annualized volatility
    annual_vol = returns.std() * np.sqrt(252)

    if pd.isna(annual_return) or pd.isna(annual_vol) or annual_vol == 0:
        return 0.0

    sharpe = (annual_return - risk_free_rate) / annual_vol

    return float(sharpe) if not pd.isna(sharpe) else 0.0


def calculate_correlation_matrix(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """
    Calculate correlation matrix between assets.

    Args:
        tickers: List of ticker symbols
        period: Historical period

    Returns:
        DataFrame with correlation matrix
    """
    prices = get_price_data(tickers, period)
    returns = get_returns(prices)

    return returns.corr()


def calculate_diversification_score(
    tickers: List[str],
    weights: List[float],
    period: str = "2y"
) -> float:
    """
    Calculate diversification score based on weighted average correlation.

    Score ranges from 0 (perfectly correlated) to 1 (uncorrelated).
    Higher is better (more diversified).

    Args:
        tickers: List of ticker symbols
        weights: Portfolio weights
        period: Historical period

    Returns:
        Diversification score (0-1)
    """
    if len(tickers) < 2:
        return 1.0  # Single asset is maximally "diversified" for its size

    corr_matrix = calculate_correlation_matrix(tickers, period)

    # Calculate weighted average off-diagonal correlation
    weight_dict = dict(zip(tickers, weights))

    total_weighted_corr = 0.0
    total_weight = 0.0

    for i, t1 in enumerate(corr_matrix.columns):
        for j, t2 in enumerate(corr_matrix.columns):
            if i < j:  # Off-diagonal only
                w1 = weight_dict.get(t1, 0)
                w2 = weight_dict.get(t2, 0)
                pair_weight = w1 * w2

                if pair_weight > 0:
                    total_weighted_corr += corr_matrix.loc[t1, t2] * pair_weight
                    total_weight += pair_weight

    if total_weight == 0:
        return 1.0

    avg_corr = total_weighted_corr / total_weight

    # Convert to diversification score (1 - correlation)
    diversification = 1.0 - avg_corr

    # Clamp to 0-1 range
    return float(max(0.0, min(1.0, diversification)))


def run_stress_tests(
    tickers: List[str],
    weights: List[float]
) -> List[Dict[str, Any]]:
    """
    Run historical stress test scenarios on the portfolio.

    Tests portfolio performance during major market events.

    Args:
        tickers: List of ticker symbols
        weights: Portfolio weights

    Returns:
        List of stress test results with scenario name and impact
    """
    scenarios = [
        {
            "name": "COVID-19 Crash (Feb-Mar 2020)",
            "start": "2020-02-19",
            "end": "2020-03-23"
        },
        {
            "name": "2022 Rate Hike Correction",
            "start": "2022-01-03",
            "end": "2022-06-16"
        },
        {
            "name": "2018 Q4 Selloff",
            "start": "2018-10-01",
            "end": "2018-12-24"
        }
    ]

    results = []

    for scenario in scenarios:
        try:
            # Fetch data for the scenario period
            start = datetime.strptime(scenario["start"], "%Y-%m-%d")
            end = datetime.strptime(scenario["end"], "%Y-%m-%d")

            data = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                prices = data[['Close']]
                prices.columns = [tickers[0]]

            # Calculate portfolio value change
            weight_dict = dict(zip(tickers, weights))

            total_impact = 0.0
            total_weight = 0.0

            for col in prices.columns:
                if col in weight_dict and not prices[col].empty:
                    start_price = prices[col].iloc[0]
                    end_price = prices[col].iloc[-1]

                    if start_price > 0:
                        stock_return = (end_price - start_price) / start_price
                        total_impact += stock_return * weight_dict[col]
                        total_weight += weight_dict[col]

            if total_weight > 0:
                # Normalize impact to actual weight
                impact = total_impact / total_weight * total_weight
                results.append({
                    "scenario": scenario["name"],
                    "portfolio_impact": round(float(impact), 4)
                })

        except Exception as e:
            # Skip scenarios with data issues
            continue

    return results


def get_full_risk_assessment(
    tickers: List[str],
    weights: List[float]
) -> Dict[str, Any]:
    """
    Calculate comprehensive risk assessment for a portfolio.

    Args:
        tickers: List of ticker symbols
        weights: Portfolio weights (must sum to 1.0)

    Returns:
        Dictionary with all risk metrics and stress test results
    """
    # Get portfolio returns
    portfolio_returns = get_portfolio_returns(tickers, weights)

    # Calculate all risk metrics
    risk_metrics = {
        "var_95": round(calculate_var(portfolio_returns, 0.95), 4),
        "cvar_95": round(calculate_cvar(portfolio_returns, 0.95), 4),
        "max_drawdown": round(calculate_max_drawdown(portfolio_returns), 4),
        "annualized_volatility": round(calculate_volatility(portfolio_returns), 4),
        "sharpe_ratio": round(calculate_sharpe_ratio(portfolio_returns), 4),
        "diversification_score": round(calculate_diversification_score(tickers, weights), 4)
    }

    # Run stress tests
    stress_tests = run_stress_tests(tickers, weights)

    # Get correlation matrix for reference
    corr_matrix = calculate_correlation_matrix(tickers)

    return {
        "risk_metrics": risk_metrics,
        "stress_tests": stress_tests,
        "correlation_matrix": corr_matrix.to_dict() if not corr_matrix.empty else {},
        "data_period": "2y",
        "calculation_timestamp": datetime.now().isoformat()
    }
