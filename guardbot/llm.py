"""Main chat LLM factory: Groq / OpenRouter / mock."""
from __future__ import annotations

from . import config

_MOCK_REPLY = (
    "🤖 (mock mode — no API key found) Your message passed both injection "
    "detectors and reached the main LLM node. Add GROQ_API_KEY or "
    "OPENROUTER_API_KEY to `.env` to get real answers here.\n\n"
    "You asked: “{q}”"
)


def get_chat_llm():
    """Returns a LangChain chat model, or None in mock mode."""
    prov = config.provider()
    if prov == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=config.GROQ_CHAT_MODEL, temperature=0.7, streaming=True)
    if prov == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.OPENROUTER_CHAT_MODEL, temperature=0.7,
                          streaming=True, base_url=config.OPENROUTER_BASE_URL,
                          api_key=config.OPENROUTER_API_KEY)
    return None


def mock_reply(query: str) -> str:
    return _MOCK_REPLY.format(q=query[:300])
