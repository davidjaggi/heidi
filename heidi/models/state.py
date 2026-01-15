from typing import List, Annotated, TypedDict, Optional, Dict
from heidi.models.schemas import AnalystReport, Portfolio, RiskAssessment
from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    tickers: List[str]
    reports: Annotated[List[AnalystReport], operator.add]
    portfolio: Portfolio
    model_provider: Optional[str]
    model_name_shallow: Optional[str]
    model_name_deep: Optional[str]
    prompts: Annotated[List[Dict[str, str]], operator.add]
    revision_count: int
    review_feedback: Annotated[List[str], operator.add]
    # Risk management fields
    risk_assessment: RiskAssessment
    risk_feedback: Annotated[List[str], operator.add]
    risk_revision_count: int
