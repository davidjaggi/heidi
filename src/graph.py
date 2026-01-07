from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from typing import List, Dict

from src.models.state import AgentState
from src.agents.analyst import analyst_node
from src.agents.portfolio import portfolio_node

def map_tickers(state: AgentState):
    """
    Maps the list of tickers to individual Analyst Node executions.
    """
    return [
        Send("analyst_node", {
            "ticker": ticker,
            "model_provider": state.get("model_provider"),
            "model_name": state.get("model_name")
        }) 
        for ticker in state["tickers"]
    ]

def create_graph():
    builder = StateGraph(AgentState)
    
    # Nodes
    builder.add_node("analyst_node", analyst_node)
    builder.add_node("portfolio_node", portfolio_node)
    
    # Edges
    # Start -> Map over tickers -> Analyst Nodes
    builder.add_conditional_edges(START, map_tickers, ["analyst_node"])
    
    # Analyst Nodes -> Portfolio Manager
    # Since 'analyst_node' is mapped, we wait for all to finish before moving to portfolio
    builder.add_edge("analyst_node", "portfolio_node")
    
    builder.add_edge("portfolio_node", END)
    
    return builder.compile()
