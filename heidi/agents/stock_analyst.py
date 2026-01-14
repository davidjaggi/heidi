import logging
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from heidi.tools.market_data import get_full_analysis_data
from cli.utils.llm import get_llm
from heidi.default_config import DEFAULT_CONFIG
from heidi.models.schemas import AnalystReport
from cli.utils.callbacks import HeidiCallbackHandler

logger = logging.getLogger(__name__)

def stock_analyst_node(inputs: Dict[str, Any]) -> Dict[str, Any]:
    ticker = inputs["ticker"]
    model_provider = inputs.get("model_provider", "gemini")
    model_name_shallow = inputs.get("model_name_shallow") or inputs.get("model_name") or DEFAULT_CONFIG["shallow_think_llm"]
    model_name_deep = inputs.get("model_name_deep") or inputs.get("model_name") or DEFAULT_CONFIG["deep_think_llm"]
    
    logger.info(f"Analyst Node running for {ticker}...")
    
    # 1. Fetch Data
    try:
        data = get_full_analysis_data(ticker)
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        raise

    # 2. Build Prompt
    prompt = _build_prompt(ticker, data)
    
    # 3. Call LLM
    llm = get_llm(model_provider, model_name_shallow)
    structured_llm = llm.with_structured_output(AnalystReport)
    
    chain = prompt | structured_llm
    report = chain.invoke({}, config={"callbacks": [HeidiCallbackHandler()], "metadata": {"agent_name": f"Analyst:{ticker}"}})
    
    # Capture prompt for logging
    messages = prompt.format_messages()
    prompt_text = "\n".join([f"### {m.type.upper()}\n{m.content}" for m in messages])

    # Ensure sector and industry are populated (fallback if LLM misses it)
    if "info" in data:
        if not report.sector:
            report.sector = data["info"].get("sector", "Unknown Sector")
        if not report.industry:
            report.industry = data["info"].get("industry", "Unknown Industry")
    
    # Return as list to match 'reports' state annotation (operator.add)
    return {
        "reports": [report],
        "prompts": [{"agent": f"Analyst:{ticker}", "prompt": prompt_text}]
    }

def _build_prompt(ticker: str, data: Dict[str, Any]) -> ChatPromptTemplate:
    info = data.get("info", {})
    history = data.get("history", "")
    news = data.get("news", [])
    tech = data.get("technical_indicators", {})
    esg = data.get("esg", {})
    
    news_str = "\n".join([f"- {n['title']}, {n['summary']})" for n in news])
    
    tech_str = f"""
- RSI (14): {tech.get('rsi_14', 'N/A')}
- MACD: {tech.get('macd', 'N/A')} (Signal: {tech.get('macd_signal', 'N/A')})
- SMA 50: {tech.get('sma_50', 'N/A')}
- SMA 200: {tech.get('sma_200', 'N/A')}
- Bollinger Bands: Upper {tech.get('bb_upper', 'N/A')}, Middle {tech.get('bb_middle', 'N/A')}, Lower {tech.get('bb_lower', 'N/A')}
- ATR (14): {tech.get('atr_14', 'N/A')}
    """.strip()

    esg_str = f"""
- Overall Risk: {esg.get('overall_risk', 'N/A')} ({esg.get('overall_risk_rating', 'N/A')})
- Board Risk: {esg.get('board_risk', 'N/A')} ({esg.get('board_risk_rating', 'N/A')})
- Audit Risk: {esg.get('audit_risk', 'N/A')} ({esg.get('audit_risk_rating', 'N/A')})
- Compensation Risk: {esg.get('compensation_risk', 'N/A')} ({esg.get('compensation_risk_rating', 'N/A')})
- Shareholder Rights Risk: {esg.get('shareholder_rights_risk', 'N/A')} ({esg.get('shareholder_rights_risk_rating', 'N/A')})
- Average Governance Risk: {esg.get('avg_governance_risk', 'N/A')} ({esg.get('avg_governance_risk_rating', 'N/A')})
    """.strip()

    system_msg = f"""
You are a senior financial analyst specializing in the Swiss Market. 
Analyze the following data for {ticker} and produce a structured investment report.

### Company Info
Name: {info.get("long_name")}
Sector: {info.get("sector")}
Industry: {info.get("industry")}
Market Cap: {info.get("market_cap")}
PE Ratio: {info.get("pe_ratio")}
Dividend Yield: {info.get("dividend_yield")}

### Analyst Price Targets
- Target Low: {info.get("target_low", "N/A")}
- Target Mean: {info.get("target_mean", "N/A")}
- Target High: {info.get("target_high", "N/A")}
- Rating: {info.get("recommendation_key", "N/A")} (Mean Score: {info.get("recommendation_mean", "N/A")})
- Coverage: Based on {info.get("number_of_analysts", "N/A")} analysts

### Market Data
Current Price: {info.get('current_price')} {info.get('currency')}
52 Week High: {info.get('52_week_high')}
52 Week Low: {info.get('52_week_low')}

### Technical Indicators
{tech_str}

### ESG / Governance Risk (1-10 scale, lower is better)
{esg_str}

### Price History (Weekly Close, Last 1 Year)
{history}

### Recent News
{news_str}

### Instructions
1. Analyze the fundamentals, technical trend, sentiment, and ESG/governance factors.
2. Determine a recommendation: STRONG_BUY, BUY, NEUTRAL, SELL, or STRONG_SELL.
3. Assign a confidence score (0.0 to 1.0).
4. List key drivers and risks (include any ESG concerns if relevant).
5. Provide a technical view.
6. Provide an ESG assessment summarizing governance quality.
"""
    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", "Generate the analyst report.")
    ])
