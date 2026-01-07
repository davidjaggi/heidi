import logging
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from src.data.market_data import get_full_analysis_data
from cli.utils.llm import get_llm
from src.models.schemas import AnalystReport

logger = logging.getLogger(__name__)

def analyst_node(inputs: Dict[str, Any]) -> Dict[str, Any]:
    ticker = inputs["ticker"]
    model_provider = inputs.get("model_provider", "gemini")
    model_name = inputs.get("model_name")
    
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
    llm = get_llm(model_provider, model_name)
    structured_llm = llm.with_structured_output(AnalystReport)
    
    chain = prompt | structured_llm
    report = chain.invoke({})
    
    # Return as list to match 'reports' state annotation (operator.add)
    return {"reports": [report]}

def _build_prompt(ticker: str, data: Dict[str, Any]) -> ChatPromptTemplate:
    info = data.get("info", {})
    history = data.get("history", "")
    news = data.get("news", [])
    
    news_str = "\n".join([f"- {n['title']} ({n['publisher']}, {n['publish_time']})" for n in news])
    
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

### Market Data
Current Price: {info.get('current_price')} {info.get('currency')}
52 Week High: {info.get('52_week_high')}
52 Week Low: {info.get('52_week_low')}

### Price History (Weekly Close, Last 1 Year)
{history}

### Recent News
{news_str}

### Instructions
1. Analyze the fundamentals, technical trend, and sentiment.
2. Determine a recommendation: STRONG_BUY, BUY, NEUTRAL, SELL, or STRONG_SELL.
3. Assign a confidence score (0.0 to 1.0).
4. List key drivers and risks.
5. Provide a technical view.
"""
    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", "Generate the analyst report.")
    ])
