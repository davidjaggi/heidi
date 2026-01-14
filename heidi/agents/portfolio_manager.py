import logging
from heidi.models.state import AgentState
from cli.utils.llm import get_llm
from heidi.models.schemas import Portfolio
from heidi.default_config import DEFAULT_CONFIG
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from cli.utils.callbacks import HeidiCallbackHandler

logger = logging.getLogger(__name__)

def portfolio_manager_node(state: AgentState):
    reports = state["reports"]
    model_provider = state.get("model_provider") or DEFAULT_CONFIG["llm_provider"]
    model_name_shallow = state.get("model_name_shallow") or state.get("model_name") or DEFAULT_CONFIG["shallow_think_llm"]
    model_name_deep = state.get("model_name_deep") or state.get("model_name") or DEFAULT_CONFIG["deep_think_llm"]
    # 1. Summarize Reports
    summaries = []
    for r in reports:
        # Build ESG summary if available
        esg_summary = ""
        if r.esg_metrics and r.esg_metrics.overall_risk:
            esg_summary = f", ESG Risk: {r.esg_metrics.overall_risk}/10 ({r.esg_metrics.overall_risk_rating.value})"
        elif r.esg_assessment:
            esg_summary = f", ESG: {r.esg_assessment[:50]}..."

        # Truncate technical view if too long
        tech_view = r.technical_view[:100] + "..." if len(r.technical_view) > 100 else r.technical_view

        summaries.append(
            f"Ticker: {r.ticker}, Company: {r.company}, Sector: {r.sector}\n"
            f"  Recommendation: {r.recommendation.value}, Confidence: {r.confidence_score:.2f}\n"
            f"  Key Drivers: {', '.join(r.key_drivers[:3])}\n"
            f"  Risks: {', '.join(r.risks[:3])}\n"
            f"  Metrics: MktCap={r.metrics.market_cap}, PE={r.metrics.pe_ratio}, DivYield={r.metrics.dividend_yield}{esg_summary}\n"
            f"  Technical: {tech_view}"
        )
    data_str = "\n\n".join(summaries)

    # 2. Build Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Lead Portfolio Manager optimizing a Swiss Stock Portfolio. "
                   "Consider fundamentals, technical signals, risk factors, and ESG/governance quality when allocating."),
        ("user", f"""
Aggregated Analyst Summaries:
{data_str}

Create an optimal portfolio allocation.
IMPORTANT: You must be FULLY INVESTED - weights must sum to exactly 1.0 (100%).
Do NOT hold any cash position. Allocate all capital across the given stocks.

Consider:
- Analyst recommendations and confidence scores
- Risk/reward profile of each stock
- Sector diversification
- ESG/governance quality (prefer lower risk scores)

Provide reasoning for each allocation that references key factors.
""")
    ])

    # 3. Call LLM
    llm = get_llm(model_provider, model_name_deep)
    structured_llm = llm.with_structured_output(Portfolio)

    chain = prompt | structured_llm
    portfolio = chain.invoke({}, config={"callbacks": [HeidiCallbackHandler()], "metadata": {"agent_name": "PortfolioManager"}})

    # Capture prompt for logging
    messages = prompt.format_messages()
    prompt_text = "\n".join([f"### {m.type.upper()}\n{m.content}" for m in messages])

    # Timestamp
    portfolio.timestamp = datetime.now().isoformat()

    return {
        "portfolio": portfolio,
        "prompts": [{"agent": "PortfolioManager", "prompt": prompt_text}]
    }
