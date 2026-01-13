import logging
from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger("Heidi")

class HeidiCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for nuanced logging in the Heidi multi-agent system.
    """
    def on_chat_model_start(
        self,
        serialized: Optional[Dict[str, Any]],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        # metadata might contain 'model_name' or 'agent_name' if we pass them
        agent_name = metadata.get("agent_name", "Unknown Agent") if metadata else "Unknown Agent"
        logger.info(f"[{agent_name}] Starting Chat Session...")

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        name = serialized.get("name", "Chain") if serialized else "Chain"
        logger.info(f"Starting execution: {name}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logger.info("LLM Generation completed successfully.")

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        name = serialized.get("name", "Tool") if serialized else "Tool"
        logger.info(f"Tool execution started: {name}")
