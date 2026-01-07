import logging
from src.models.state import AgentState
from cli.utils.llm import get_llm
from src.models.schemas import Portfolio
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

logger = logging.getLogger(__name__)

def portfolio_node(state: AgentState):
    reports = state["reports"]
    # We might need to pass model config in state or use defaults/globals.
    # For now assuming defaults or we could add config to state.
    # Let's assume we can pass config via RunnableConfig if needed, but for simplicity:
    model_provider = "gemini" 
    
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
    llm = get_llm(model_provider)
    structured_llm = llm.with_structured_output(Portfolio)
    
    chain = prompt | structured_llm
    portfolio = chain.invoke({})
    
    # Timestamp
    portfolio.timestamp = datetime.now().isoformat()
    
    return {"portfolio": portfolio}
