import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(model_provider: str = "gemini", model_name: str = None) -> BaseChatModel:
    """
    Returns a LangChain Chat Model instance.
    """
    if model_provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0
        )
    elif model_provider in ["anthropic", "claude"]:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        return ChatAnthropic(
            model=model_name or "claude-haiku-4-5",
            api_key=api_key,
            temperature=0
        )
    elif model_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        return ChatOpenAI(
            model=model_name or "gpt-4o",
            api_key=api_key,
            temperature=0
        )
    else:
        raise ValueError(f"Unknown provider: {model_provider}")
