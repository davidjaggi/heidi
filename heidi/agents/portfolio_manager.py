import logging
from heidi.models.state import AgentState
from cli.utils.llm import get_llm
from heidi.models.schemas import Portfolio, PortfolioAllocation
from heidi.default_config import DEFAULT_CONFIG
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from cli.utils.callbacks import HeidiCallbackHandler

logger = logging.getLogger(__name__)


def _validate_and_filter_portfolio(portfolio: Portfolio, valid_tickers: list) -> Portfolio:
    """
    Validate portfolio allocations and filter out any invalid tickers.
    Re-normalizes weights to ensure they sum to 1.0.

    Args:
        portfolio: The portfolio from LLM
        valid_tickers: List of valid ticker symbols from input

    Returns:
        Validated and filtered Portfolio
    """
    valid_tickers_set = set(valid_tickers)

    # Filter allocations to only include valid tickers
    valid_allocations = []
    removed_tickers = []

    for alloc in portfolio.allocations:
        if alloc.ticker in valid_tickers_set:
            valid_allocations.append(alloc)
        else:
            removed_tickers.append(alloc.ticker)
            logger.warning(f"Removed invalid ticker from portfolio: {alloc.ticker}")

    if removed_tickers:
        logger.warning(f"Portfolio contained invalid tickers that were removed: {removed_tickers}")

    if not valid_allocations:
        logger.error("No valid allocations remaining after filtering!")
        # Return empty portfolio with valid tickers equally weighted as fallback
        equal_weight = 1.0 / len(valid_tickers)
        valid_allocations = [
            PortfolioAllocation(
                ticker=t,
                weight=equal_weight,
                reasoning="Fallback equal-weight allocation due to invalid portfolio"
            )
            for t in valid_tickers
        ]

    # Re-normalize weights to sum to 1.0
    total_weight = sum(a.weight for a in valid_allocations)
    if total_weight > 0 and abs(total_weight - 1.0) > 0.001:
        logger.info(f"Re-normalizing portfolio weights from {total_weight:.4f} to 1.0")
        normalized_allocations = [
            PortfolioAllocation(
                ticker=a.ticker,
                weight=a.weight / total_weight,
                reasoning=a.reasoning
            )
            for a in valid_allocations
        ]
        valid_allocations = normalized_allocations

    return Portfolio(
        allocations=valid_allocations,
        timestamp=portfolio.timestamp
    )


def portfolio_manager_node(state: AgentState):
    reports = state["reports"]
    valid_tickers = state["tickers"]  # The original input tickers
    model_provider = state.get("model_provider") or DEFAULT_CONFIG["llm_provider"]
    model_name_shallow = state.get("model_name_shallow") or state.get("model_name") or DEFAULT_CONFIG["shallow_think_llm"]
    model_name_deep = state.get("model_name_deep") or state.get("model_name") or DEFAULT_CONFIG["deep_think_llm"]
    risk_feedback = state.get("risk_feedback", [])
    risk_revision_count = state.get("risk_revision_count", 0)

    logger.info(f"Portfolio Manager running (risk revision {risk_revision_count})...")

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
    # Include risk feedback context if this is a revision
    risk_context = ""
    if risk_feedback and risk_revision_count > 0:
        risk_context = f"""
IMPORTANT - RISK MANAGER FEEDBACK (Revision #{risk_revision_count}):
The Risk Manager has reviewed the previous portfolio and requires adjustments.
You MUST address these concerns in your revised allocation:

{chr(10).join(risk_feedback)}

Adjust weights to reduce the identified risks while maintaining portfolio quality.
"""

    system_msg = "You are a Lead Portfolio Manager optimizing a Swiss Stock Portfolio. "
    system_msg += "Consider fundamentals, technical signals, risk factors, and ESG/governance quality when allocating."

    if risk_revision_count > 0:
        system_msg += " This is a REVISION based on Risk Manager feedback. Prioritize addressing the risk concerns."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", f"""
Aggregated Analyst Summaries:
{data_str}
{risk_context}
Create an optimal portfolio allocation.
CRITICAL CONSTRAINTS:
- You must be FULLY INVESTED - weights must sum to exactly 1.0 (100%)
- Do NOT hold any cash position - cash is NOT allowed
- Only allocate to the tickers provided above - no other assets

Consider:
- Analyst recommendations and confidence scores
- Risk/reward profile of each stock
- Sector diversification
- ESG/governance quality (prefer lower risk scores)
{("- RISK MANAGER CONCERNS (address these!)" if risk_revision_count > 0 else "")}

Provide reasoning for each allocation that references key factors.
""")
    ])

    # 3. Call LLM
    llm = get_llm(model_provider, model_name_deep)
    structured_llm = llm.with_structured_output(Portfolio)

    chain = prompt | structured_llm
    portfolio = chain.invoke({}, config={"callbacks": [HeidiCallbackHandler()], "metadata": {"agent_name": "PortfolioManager"}})

    # Validate and filter portfolio to only include valid tickers
    portfolio = _validate_and_filter_portfolio(portfolio, valid_tickers)

    # Capture prompt for logging
    messages = prompt.format_messages()
    prompt_text = "\n".join([f"### {m.type.upper()}\n{m.content}" for m in messages])

    # Timestamp
    portfolio.timestamp = datetime.now().isoformat()

    return {
        "portfolio": portfolio,
        "prompts": [{"agent": "PortfolioManager", "prompt": prompt_text}]
    }
