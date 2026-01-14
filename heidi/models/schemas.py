from typing import List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

class Recommendation(str, Enum):
    """Investment recommendation. Must be exactly one of these values."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

    @classmethod
    def _missing_(cls, value):
        """Handle case-insensitive and variation matching."""
        if isinstance(value, str):
            # Normalize: uppercase, replace spaces/hyphens with underscore
            normalized = value.upper().replace(" ", "_").replace("-", "_")
            for member in cls:
                if member.value == normalized:
                    return member
        return None

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
    """Structured analyst report for a stock."""
    ticker: str = Field(description="Stock ticker symbol (e.g., NESN.SW)")
    company: str = Field(description="Full company name")
    sector: str = Field(description="Business sector")
    industry: str = Field(description="Specific industry")
    recommendation: Recommendation = Field(
        description="Investment recommendation: STRONG_BUY, BUY, NEUTRAL, SELL, or STRONG_SELL"
    )
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in recommendation (0.0-1.0)")
    key_drivers: List[str] = Field(description="Key positive factors supporting the recommendation")
    risks: List[str] = Field(description="Key risk factors to consider")
    technical_view: str = Field(description="Technical analysis summary")
    esg_assessment: str = Field(default="", description="ESG/Governance quality assessment")
    metrics: Metrics
    esg_metrics: Optional[ESGMetrics] = None

    @model_validator(mode='before')
    @classmethod
    def normalize_recommendation_value(cls, data):
        """Normalize recommendation value before any field validation."""
        if isinstance(data, dict) and 'recommendation' in data:
            v = data['recommendation']
            if isinstance(v, str):
                # Normalize: uppercase, replace spaces/hyphens with underscore
                normalized = v.upper().replace(" ", "_").replace("-", "_")
                # Handle common variations
                if normalized in ["STRONG_BUY", "STRONGBUY"]:
                    data['recommendation'] = "STRONG_BUY"
                elif normalized in ["STRONG_SELL", "STRONGSELL"]:
                    data['recommendation'] = "STRONG_SELL"
                elif normalized in ["BUY"]:
                    data['recommendation'] = "BUY"
                elif normalized in ["SELL"]:
                    data['recommendation'] = "SELL"
                elif normalized in ["NEUTRAL", "HOLD"]:
                    data['recommendation'] = "NEUTRAL"
                else:
                    data['recommendation'] = normalized
        return data

class PortfolioAllocation(BaseModel):
    ticker: str
    weight: float
    reasoning: str


class Portfolio(BaseModel):
    allocations: List[PortfolioAllocation]
    timestamp: str


class ReviewDecision(str, Enum):
    APPROVED = "APPROVED"
    NEEDS_REVISION = "NEEDS_REVISION"


class ReportReview(BaseModel):
    """Review result for an analyst report."""
    ticker: str
    decision: ReviewDecision
    strengths: List[str] = Field(description="What the report did well")
    issues: List[str] = Field(default_factory=list, description="Issues found that need addressing")
    feedback: str = Field(description="Detailed feedback for the analyst")
    confidence_in_review: float = Field(ge=0, le=1, description="Reviewer's confidence in their assessment")
