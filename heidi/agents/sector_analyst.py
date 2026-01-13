import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from heidi.models.schemas import SectorReport, AnalystReport
from heidi.default_config import DEFAULT_CONFIG
from cli.utils.llm import get_llm
from cli.utils.callbacks import HeidiCallbackHandler

logger = logging.getLogger(__name__)

def sector_analyst_node(inputs: Dict[str, Any]) -> Dict[str, Any]:
    sector = inputs["sector"]
    reports: List[AnalystReport] = inputs["reports"]
    model_provider = inputs.get("model_provider") or DEFAULT_CONFIG["llm_provider"]
    model_name = inputs.get("model_name") or DEFAULT_CONFIG["fast_think_llm"]
    
    logger.info(f"Sector Analyst Node running for {sector}...")
    
    # 1. Prepare Data
    analyst_summaries = []
    for r in reports:
        analyst_summaries.append(
            f"Ticker: {r.ticker}, Recommendation: {r.recommendation.value}, "
            f"Confidence: {r.confidence_score}, Drivers: {', '.join(r.key_drivers[:3])}"
        )
    data_str = "\n".join(analyst_summaries)
    
    # 2. Build Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a specialized Sector Analyst focusing on the {sector} sector in the Swiss Market."),
        ("user", f"""
Analyze the professional summaries provided by stock analysts for companies in the {sector} sector.
Provide a high-level sector outlook, identify the top picks based on the analyst reports, and summarize sector-wide risks.

### Stock Analyst Findings:
{data_str}

Return a structured sector report.
""")
    ])
    
    # 3. Call LLM
    llm = get_llm(model_provider, model_name)
    structured_llm = llm.with_structured_output(SectorReport)
    
    chain = prompt | structured_llm
    sector_report = chain.invoke({}, config={"callbacks": [HeidiCallbackHandler()], "metadata": {"agent_name": f"SectorAnalyst:{sector}"}})
    
    return {"sector_reports": [sector_report]}
