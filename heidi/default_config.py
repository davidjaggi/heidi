import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "llm_provider": "anthropic", # "gemini",
    "deep_think_llm": "claude-haiku-4-5",
    "shallow_think_llm": "claude-haiku-4-5",
    "tickers": "data/test_tickers.txt",
    "max_revisions": 2,
    "max_risk_revisions": 2
}