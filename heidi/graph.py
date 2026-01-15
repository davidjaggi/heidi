from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.constants import Send
from langchain_core.runnables.graph import MermaidDrawMethod

from heidi.models.state import AgentState
from heidi.agents.stock_analyst import stock_analyst_node
from heidi.agents.portfolio_manager import portfolio_manager_node
from heidi.agents.report_reviewer import report_reviewer_node, should_revise
from heidi.agents.risk_manager import risk_manager_node, should_revise_portfolio


def save_graph_as_png(app: CompiledStateGraph, output_file_path: str = "") -> None:
    """Save the compiled graph as a PNG image using Mermaid API."""
    png_image = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    file_path = output_file_path if output_file_path else "graph.png"
    with open(file_path, "wb") as f:
        f.write(png_image)


def map_tickers(state: AgentState):
    """
    Maps the list of tickers to individual Analyst Node executions.
    """
    return [
        Send("analyst_node", {
            "ticker": ticker,
            "model_provider": state.get("model_provider"),
            "model_name_shallow": state.get("model_name_shallow"),
            "model_name_deep": state.get("model_name_deep"),
            "review_feedback": state.get("review_feedback", [])
        })
        for ticker in state["tickers"]
    ]


def route_after_review(state: AgentState):
    """
    Route after review: revise reports (via Send) or proceed to portfolio.
    Returns list of Send objects for revision, or node name string for portfolio.
    """
    if should_revise(state):
        # Use Send to fan out to analyst nodes again with proper inputs
        return [
            Send("analyst_node", {
                "ticker": ticker,
                "model_provider": state.get("model_provider"),
                "model_name_shallow": state.get("model_name_shallow"),
                "model_name_deep": state.get("model_name_deep"),
                "review_feedback": state.get("review_feedback", [])
            })
            for ticker in state["tickers"]
        ]
    return "portfolio_node"  # Must match actual node name


def route_after_risk_review(state: AgentState):
    """
    Route after risk review: revise portfolio or finish.
    Returns node name string for portfolio revision, or END to finish.
    """
    if should_revise_portfolio(state):
        return "portfolio_node"
    return END


def create_graph(save_png: bool = True, png_path: str = "heidi/graph.png") -> CompiledStateGraph:
    """
    Creates and compiles the Heidi multi-agent graph.

    Graph flow:
    START -> analyst_nodes -> reviewer_node -> (conditional) -> portfolio_node -> risk_manager_node -> (conditional) -> END
                                    ^                |                                    ^                    |
                                    |________________|                                    |____________________|
                              (if needs revision, via Send)                         (if risk needs revision)

    Args:
        save_png: Whether to save the graph visualization as PNG
        png_path: Path to save the graph PNG

    Returns:
        Compiled LangGraph state graph
    """
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("analyst_node", stock_analyst_node)
    builder.add_node("reviewer_node", report_reviewer_node)
    builder.add_node("portfolio_node", portfolio_manager_node)
    builder.add_node("risk_node", risk_manager_node)

    # Edges
    # Start -> Map over tickers -> Analyst Nodes
    builder.add_conditional_edges(START, map_tickers, ["analyst_node"])

    # Analyst Nodes -> Reviewer
    builder.add_edge("analyst_node", "reviewer_node")

    # Reviewer -> Conditional routing (Send for revision, string for proceed)
    builder.add_conditional_edges(
        "reviewer_node",
        route_after_review,
        ["analyst_node", "portfolio_node"]  # Possible destinations
    )

    # Portfolio -> Risk Manager
    builder.add_edge("portfolio_node", "risk_node")

    # Risk Manager -> Conditional routing (back to portfolio or END)
    builder.add_conditional_edges(
        "risk_node",
        route_after_risk_review,
        ["portfolio_node", END]  # Possible destinations
    )

    graph = builder.compile()

    # Save graph visualization
    if save_png:
        save_graph_as_png(graph, png_path)

    return graph
