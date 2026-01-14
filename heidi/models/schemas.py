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


class ESGRating(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    NOT_AVAILABLE = "N/A"


class ESGMetrics(BaseModel):
    """ESG/Governance risk metrics (1-10 scale, lower is better)."""
    overall_risk: Optional[int] = Field(None, ge=1, le=10, description="Overall governance risk score")
    overall_risk_rating: ESGRating = ESGRating.NOT_AVAILABLE
    board_risk: Optional[int] = Field(None, ge=1, le=10, description="Board composition and independence risk")
    audit_risk: Optional[int] = Field(None, ge=1, le=10, description="Audit committee and practices risk")
    compensation_risk: Optional[int] = Field(None, ge=1, le=10, description="Executive compensation risk")
    shareholder_rights_risk: Optional[int] = Field(None, ge=1, le=10, description="Shareholder rights and protections risk")

class AnalystReport(BaseModel):
    ticker: str
    company: str
    sector: str
    industry: str
    recommendation: Recommendation
    confidence_score: float = Field(..., ge=0, le=1)
    key_drivers: List[str]
    risks: List[str]
    technical_view: str
    esg_assessment: str = Field(default="", description="ESG/Governance quality assessment")
    metrics: Metrics
    esg_metrics: Optional[ESGMetrics] = None

class PortfolioAllocation(BaseModel):
    ticker: str
    weight: float
    reasoning: str

class Portfolio(BaseModel):
    allocations: List[PortfolioAllocation]
    timestamp: str
