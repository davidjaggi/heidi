import logging
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from cli.utils.llm import get_llm
from heidi.default_config import DEFAULT_CONFIG
from heidi.models.schemas import ReportReview, ReviewDecision, AnalystReport
from heidi.models.state import AgentState
from cli.utils.callbacks import HeidiCallbackHandler

logger = logging.getLogger(__name__)


def report_reviewer_node(state: AgentState) -> Dict[str, Any]:
    """
    Reviews all analyst reports and provides feedback.
    Uses the deep thinking model for thorough critique.
    """
    reports = state["reports"]
    model_provider = state.get("model_provider") or DEFAULT_CONFIG["llm_provider"]
    model_name_deep = state.get("model_name_deep") or DEFAULT_CONFIG["deep_think_llm"]
    revision_count = state.get("revision_count", 0)

    logger.info(f"Reviewer Node running (revision {revision_count})...")

    # Get LLM with structured output
    llm = get_llm(model_provider, model_name_deep)
    structured_llm = llm.with_structured_output(ReportReview)

    reviews: List[ReportReview] = []
    feedback_list: List[str] = []
    prompts_log: List[Dict[str, str]] = []

    for report in reports:
        prompt = _build_review_prompt(report, revision_count)

        chain = prompt | structured_llm
        review = chain.invoke(
            {},
            config={
                "callbacks": [HeidiCallbackHandler()],
                "metadata": {"agent_name": f"Reviewer:{report.ticker}"}
            }
        )

        reviews.append(review)

        # Capture prompt and output for logging
        messages = prompt.format_messages()
        prompt_text = "\n".join([f"### {m.type.upper()}\n{m.content}" for m in messages])

        # Add reviewer output to the log
        output_text = f"""
### REVIEWER OUTPUT
**Decision:** {review.decision.value}
**Confidence in Review:** {review.confidence_in_review:.2f}

**Strengths:**
{chr(10).join(f'- {s}' for s in review.strengths) if review.strengths else '- None identified'}

**Issues:**
{chr(10).join(f'- {i}' for i in review.issues) if review.issues else '- None identified'}

**Feedback:**
{review.feedback}
"""
        prompts_log.append({"agent": f"Reviewer:{report.ticker}", "prompt": prompt_text + output_text})

        # Build feedback string
        if review.decision == ReviewDecision.NEEDS_REVISION:
            feedback_list.append(
                f"[{report.ticker}] NEEDS REVISION:\n"
                f"  Issues: {', '.join(review.issues)}\n"
                f"  Feedback: {review.feedback}"
            )
        else:
            feedback_list.append(f"[{report.ticker}] APPROVED: {review.feedback[:100]}...")

    return {
        "review_feedback": feedback_list,
        "revision_count": revision_count + 1,
        "prompts": prompts_log
    }


def _build_review_prompt(report: AnalystReport, revision_count: int) -> ChatPromptTemplate:
    """Build the review prompt for a single analyst report."""

    report_summary = f"""
TICKER: {report.ticker}
COMPANY: {report.company}
SECTOR: {report.sector} | INDUSTRY: {report.industry}

RECOMMENDATION: {report.recommendation.value}
CONFIDENCE SCORE: {report.confidence_score}

KEY DRIVERS:
{chr(10).join(f'- {d}' for d in report.key_drivers)}

RISKS:
{chr(10).join(f'- {r}' for r in report.risks)}

TECHNICAL VIEW:
{report.technical_view}

ESG ASSESSMENT:
{report.esg_assessment or 'Not provided'}

METRICS:
- Market Cap: {report.metrics.market_cap}
- P/E Ratio: {report.metrics.pe_ratio}
- Dividend Yield: {report.metrics.dividend_yield}
"""

    revision_context = ""
    if revision_count > 0:
        revision_context = f"\nThis is revision #{revision_count}. Be more lenient if previous issues were addressed."

    system_msg = f"""You are a Senior Investment Reviewer at a Swiss wealth management firm.
Your role is to critically evaluate analyst reports for quality, accuracy, and actionability.
{revision_context}

Review criteria:
1. LOGICAL CONSISTENCY: Are the key drivers and risks logically consistent with the recommendation?
2. DATA QUALITY: Is the analysis based on current, relevant data? Are metrics properly considered?
3. COMPLETENESS: Does the report cover fundamentals, technicals, and ESG factors adequately?
4. ACTIONABILITY: Is the recommendation clear and well-justified with a reasonable confidence score?
5. RISK AWARENESS: Are key risks properly identified and weighted?

If the report meets professional standards, mark it as APPROVED.
If there are significant issues, mark it as NEEDS_REVISION with specific, actionable feedback.

Be constructive but rigorous. A good report should have:
- Clear reasoning linking data to recommendation
- Balanced view of opportunities and risks
- Appropriate confidence level given the analysis depth
- ESG considerations if governance data is available"""

    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", f"Review the following analyst report:\n{report_summary}")
    ])


def should_revise(state: AgentState) -> bool:
    """
    Determine if reports need revision based on review feedback.
    Returns True if any report needs revision and we haven't exceeded max revisions.
    """
    revision_count = state.get("revision_count", 0)
    feedback = state.get("review_feedback", [])
    max_revisions = DEFAULT_CONFIG.get("max_revisions", 2)

    # Don't exceed max revisions
    if revision_count >= max_revisions:
        logger.info(f"Max revisions ({max_revisions}) reached, proceeding to portfolio.")
        return False

    # Check if any report needs revision
    needs_revision = any("NEEDS REVISION" in f for f in feedback)

    if needs_revision:
        logger.info(f"Some reports need revision (attempt {revision_count + 1}/{max_revisions})")
    else:
        logger.info("All reports approved, proceeding to portfolio.")

    return needs_revision
