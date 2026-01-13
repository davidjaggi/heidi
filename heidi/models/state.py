import operator
from typing import List, Annotated, TypedDict, Optional
from heidi.models.schemas import AnalystReport, Portfolio

class AgentState(TypedDict):
    tickers: List[str]
    reports: Annotated[List[AnalystReport], operator.add]
    portfolio: Portfolio
    model_provider: Optional[str]
    model_name: Optional[str]
