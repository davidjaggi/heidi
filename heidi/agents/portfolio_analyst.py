import logging
from heidi.models.state import AgentState
from cli.utils.llm import get_llm
from heidi.models.schemas import Portfolio
from heidi.default_config import DEFAULT_CONFIG
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from cli.utils.callbacks import HeidiCallbackHandler

logger = logging.getLogger(__name__)

def portfolio_analyst_node(state: AgentState):
    reports = state["reports"]
    model_provider = state.get("model_provider") or DEFAULT_CONFIG["llm_provider"]
    model_name = state.get("model_name") or DEFAULT_CONFIG["deep_think_llm"]
    # 1. Summarize Reports
    summaries = []
    for r in reports:
        summaries.append(
            f"Ticker: {r.ticker}, Rec: {r.recommendation.value}, Score: {r.confidence_score}, "
            f"Drivers: {', '.join(r.key_drivers[:2])}, Risks: {', '.join(r.risks[:2])}, "
            f"MktCap: {r.metrics.market_cap}, PE: {r.metrics.pe_ratio}"
        )
    data_str = "\n".join(summaries)
    
    # 2. Build Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Lead Portfolio Manager optimizing a Swiss Stock Portfolio."),
        ("user", f"""
Aggregated Analyst Summaries:
{data_str}

Create an optimal portfolio allocation (weights summing to ~1.0).
Provide reasoning for each allocation.
""")
    ])
    
    # 3. Call LLM
    llm = get_llm(model_provider, model_name)
    structured_llm = llm.with_structured_output(Portfolio)
    
    chain = prompt | structured_llm
    portfolio = chain.invoke({}, config={"callbacks": [HeidiCallbackHandler()], "metadata": {"agent_name": "PortfolioManager"}})
    
    # Timestamp
    portfolio.timestamp = datetime.now().isoformat()
    
    return {"portfolio": portfolio}
