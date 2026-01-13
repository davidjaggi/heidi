from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from typing import List, Dict

from heidi.models.state import AgentState
from heidi.agents.stock_analyst import stock_analyst_node
from heidi.agents.portfolio_analyst import portfolio_analyst_node

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
    builder.add_node("analyst_node", stock_analyst_node)
    builder.add_node("portfolio_node", portfolio_analyst_node)
    
    # Edges
    # Start -> Map over tickers -> Analyst Nodes
    builder.add_conditional_edges(START, map_tickers, ["analyst_node"])
    
    # Analyst Nodes -> Portfolio Manager
    builder.add_edge("analyst_node", "portfolio_node")
    
    builder.add_edge("portfolio_node", END)

    graph = builder.compile()
    
    # save graph as png
    graph.get_graph().draw_mermaid_png(
        output_file_path="heidi/graph.png"
    )
    return graph
