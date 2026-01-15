"""
Risk Manager Agent - Performs stress tests and risk assessment on portfolios.
Communicates with Portfolio Manager until risk criteria are met.
"""
import logging
from datetime import datetime
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from heidi.models.state import AgentState
from heidi.models.schemas import RiskAssessment, RiskDecision, RiskMetrics, StressTestResult, RiskManagerDecision
from heidi.tools.risk_calculations import get_full_risk_assessment
from heidi.default_config import DEFAULT_CONFIG
from cli.utils.llm import get_llm
from cli.utils.callbacks import HeidiCallbackHandler

logger = logging.getLogger(__name__)


def risk_manager_node(state: AgentState) -> Dict[str, Any]:
    """
    Risk Manager node that evaluates portfolio risk.

    1. Extracts portfolio allocations
    2. Calculates risk metrics using historical data
    3. Runs stress tests
    4. LLM evaluates overall risk acceptability
    5. Returns APPROVED or NEEDS_REVISION with feedback

    Args:
        state: Current agent state with portfolio

    Returns:
        Dict with risk_assessment, risk_feedback, risk_revision_count, prompts
    """
    portfolio = state["portfolio"]
    model_provider = state.get("model_provider") or DEFAULT_CONFIG["llm_provider"]
    model_name_deep = state.get("model_name_deep") or DEFAULT_CONFIG["deep_think_llm"]
    risk_revision_count = state.get("risk_revision_count", 0)

    logger.info(f"Risk Manager running (revision {risk_revision_count})...")

    # Extract tickers and weights from portfolio
    tickers = [alloc.ticker for alloc in portfolio.allocations]
    weights = [alloc.weight for alloc in portfolio.allocations]

    logger.info(f"Portfolio tickers: {tickers}")
    logger.info(f"Portfolio weights: {weights}")

    # Calculate all risk metrics
    logger.info(f"Calculating risk metrics for {len(tickers)} positions...")
    risk_data = get_full_risk_assessment(tickers, weights)
    logger.info(f"Calculated risk metrics: {risk_data['risk_metrics']}")

    # Build risk summary for LLM
    risk_summary = _build_risk_summary(portfolio, risk_data)

    # Build prompt
    prompt = _build_risk_prompt(risk_summary, risk_revision_count)

    # Call LLM for risk decision only (not metrics - we calculate those)
    llm = get_llm(model_provider, model_name_deep)
    structured_llm = llm.with_structured_output(RiskManagerDecision)

    chain = prompt | structured_llm
    llm_decision = chain.invoke(
        {},
        config={
            "callbacks": [HeidiCallbackHandler()],
            "metadata": {"agent_name": "RiskManager"}
        }
    )

    # Build RiskMetrics from calculated data
    metrics = risk_data["risk_metrics"]
    risk_metrics = RiskMetrics(
        var_95=metrics["var_95"],
        cvar_95=metrics["cvar_95"],
        max_drawdown=metrics["max_drawdown"],
        annualized_volatility=metrics["annualized_volatility"],
        sharpe_ratio=metrics["sharpe_ratio"],
        diversification_score=metrics["diversification_score"]
    )

    # Build stress test results
    stress_tests = [
        StressTestResult(scenario=st["scenario"], portfolio_impact=st["portfolio_impact"])
        for st in risk_data.get("stress_tests", [])
    ]

    # Assemble full RiskAssessment
    assessment = RiskAssessment(
        decision=llm_decision.decision,
        risk_metrics=risk_metrics,
        stress_tests=stress_tests,
        concerns=llm_decision.concerns,
        feedback=llm_decision.feedback,
        timestamp=datetime.now().isoformat()
    )

    # Capture prompt for logging
    messages = prompt.format_messages()
    prompt_text = "\n".join([f"### {m.type.upper()}\n{m.content}" for m in messages])

    # Add risk manager output to the log
    stress_test_output = "\n".join([
        f"- {st.scenario}: {st.portfolio_impact:+.2%}"
        for st in assessment.stress_tests
    ]) if assessment.stress_tests else "- No stress tests available"

    output_text = f"""
### RISK MANAGER OUTPUT
**Decision:** {assessment.decision.value}

**Risk Metrics:**
- VaR (95%, daily): {assessment.risk_metrics.var_95:.2%}
- CVaR (95%): {assessment.risk_metrics.cvar_95:.2%}
- Max Drawdown: {assessment.risk_metrics.max_drawdown:.2%}
- Volatility (ann.): {assessment.risk_metrics.annualized_volatility:.2%}
- Sharpe Ratio: {assessment.risk_metrics.sharpe_ratio:.2f}
- Diversification: {assessment.risk_metrics.diversification_score:.2f}

**Stress Tests:**
{stress_test_output}

**Concerns:**
{chr(10).join(f'- {c}' for c in assessment.concerns) if assessment.concerns else '- None identified'}

**Feedback:**
{assessment.feedback}
"""

    # Build feedback for portfolio manager if revision needed
    feedback_list = []
    if assessment.decision == RiskDecision.NEEDS_REVISION:
        feedback_list.append(
            f"[RISK REVISION REQUIRED]\n"
            f"Concerns: {', '.join(assessment.concerns)}\n"
            f"Feedback: {assessment.feedback}"
        )
        logger.info(f"Portfolio REJECTED: {len(assessment.concerns)} concerns identified")
    else:
        feedback_list.append(f"[RISK APPROVED] {assessment.feedback[:100]}...")
        logger.info("Portfolio APPROVED by Risk Manager")

    return {
        "risk_assessment": assessment,
        "risk_feedback": feedback_list,
        "risk_revision_count": risk_revision_count + 1,
        "prompts": [{"agent": "RiskManager", "prompt": prompt_text + output_text}]
    }


def _build_risk_summary(portfolio, risk_data: Dict[str, Any]) -> str:
    """Build a formatted risk summary string for the LLM."""
    metrics = risk_data["risk_metrics"]
    stress_tests = risk_data["stress_tests"]

    # Portfolio allocation summary
    alloc_summary = "\n".join([
        f"  {a.ticker}: {a.weight:.1%} - {a.reasoning[:50]}..."
        for a in portfolio.allocations
    ])

    # Risk metrics summary
    metrics_summary = f"""
RISK METRICS:
  - Value at Risk (95%, daily): {metrics['var_95']:.2%}
  - Conditional VaR (95%): {metrics['cvar_95']:.2%}
  - Maximum Drawdown: {metrics['max_drawdown']:.2%}
  - Annualized Volatility: {metrics['annualized_volatility']:.2%}
  - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
  - Diversification Score: {metrics['diversification_score']:.2f} (0-1, higher is better)
"""

    # Stress test summary
    if stress_tests:
        stress_summary = "STRESS TEST RESULTS:\n"
        for test in stress_tests:
            impact = test["portfolio_impact"]
            impact_str = f"{impact:+.2%}"
            stress_summary += f"  - {test['scenario']}: {impact_str}\n"
    else:
        stress_summary = "STRESS TEST RESULTS:\n  No historical stress data available.\n"

    return f"""
PORTFOLIO ALLOCATION:
{alloc_summary}

{metrics_summary}
{stress_summary}
"""


def _build_risk_prompt(risk_summary: str, revision_count: int) -> ChatPromptTemplate:
    """Build the risk assessment prompt."""

    revision_context = ""
    if revision_count > 0:
        revision_context = f"""
NOTE: This is revision #{revision_count}. The Portfolio Manager has adjusted allocations
based on previous risk feedback. Evaluate if the changes adequately address prior concerns.
Be more lenient if significant improvements were made, but maintain risk standards.
"""

    system_msg = f"""You are a Senior Risk Manager at a Swiss wealth management firm.
Your role is to evaluate portfolio risk and ensure it meets prudent investment standards.
{revision_context}
EVALUATION CRITERIA:

1. VALUE AT RISK: Is the daily VaR reasonable? (typically < 2-3% for balanced portfolios)
2. EXPECTED SHORTFALL: Is the tail risk (CVaR) acceptable?
3. DRAWDOWN RISK: Could the portfolio withstand historical max drawdowns?
4. VOLATILITY: Is the annualized volatility appropriate for the strategy?
5. RISK-ADJUSTED RETURNS: Is the Sharpe Ratio acceptable? (typically > 0.5 is decent)
6. DIVERSIFICATION: Is the portfolio sufficiently diversified? (score > 0.3 preferred)
7. STRESS RESILIENCE: How would the portfolio perform in crisis scenarios?

DECISION GUIDELINES:
- APPROVE if risk metrics are within acceptable bounds and well-diversified
- REJECT if:
  * VaR is excessively high (> 4% daily)
  * Max drawdown risk exceeds 40%
  * Diversification score is very low (< 0.2)
  * Stress test results show catastrophic losses
  * Concentration risk in single positions

When rejecting, provide SPECIFIC, ACTIONABLE feedback:
- Which positions contribute most to risk
- Suggested allocation adjustments (rebalance weights among existing stocks)
- Alternative approaches to reduce risk

IMPORTANT CONSTRAINTS:
- The portfolio MUST remain fully invested (100% allocation)
- Cash positions are NOT allowed
- Do NOT suggest adding new tickers - only rebalance weights among existing stocks
- All suggestions must involve rebalancing weights among the given stocks only
- Do NOT suggest holding cash, reducing total exposure, or adding any assets not in the portfolio"""

    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", f"Evaluate the risk profile of the following portfolio:\n{risk_summary}")
    ])


def should_revise_portfolio(state: AgentState) -> bool:
    """
    Determine if the portfolio needs revision based on risk assessment.

    Returns True if risk assessment requires revision and we haven't
    exceeded the maximum number of risk revisions.

    Args:
        state: Current agent state

    Returns:
        True if portfolio should be revised, False if approved or max revisions reached
    """
    risk_assessment = state.get("risk_assessment")
    risk_revision_count = state.get("risk_revision_count", 0)
    max_risk_revisions = DEFAULT_CONFIG.get("max_risk_revisions", 2)

    # No assessment yet (shouldn't happen in normal flow)
    if risk_assessment is None:
        return False

    # Don't exceed max revisions
    if risk_revision_count >= max_risk_revisions:
        logger.info(f"Max risk revisions ({max_risk_revisions}) reached, proceeding with current portfolio.")
        return False

    # Check decision
    needs_revision = risk_assessment.decision == RiskDecision.NEEDS_REVISION

    if needs_revision:
        logger.info(f"Risk revision required (attempt {risk_revision_count}/{max_risk_revisions})")
    else:
        logger.info("Portfolio approved by Risk Manager.")

    return needs_revision
