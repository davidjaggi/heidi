from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class Recommendation(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class Metrics(BaseModel):
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps: Optional[float] = None
    profit_margin: Optional[float] = None
    high_52_week: Optional[float] = Field(None, alias="52_week_high")
    low_52_week: Optional[float] = Field(None, alias="52_week_low")
    analyst_target_price: Optional[float] = None
    currency: str = "CHF"

class AnalystReport(BaseModel):
    ticker: str
    company: str
    recommendation: Recommendation
    confidence_score: float = Field(..., ge=0, le=1)
    key_drivers: List[str]
    risks: List[str]
    technical_view: str
    metrics: Metrics

class PortfolioAllocation(BaseModel):
    ticker: str
    weight: float
    reasoning: str

class Portfolio(BaseModel):
    allocations: List[PortfolioAllocation]
    timestamp: str
