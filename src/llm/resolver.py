"""Factory for resolving Phase 2 LLM providers.

This file contains get_llm_provider. The resolver centralizes provider selection
so agents and graph code never branch on concrete provider names.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

from src.llm.base import BaseLLMProvider
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider

logger = logging.getLogger(__name__)


def get_llm_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> BaseLLMProvider:
    """Resolve the configured provider from explicit args or environment."""

    resolved_provider = (provider_name or os.getenv("LLM_PROVIDER", "auto")).lower().strip()
    if resolved_provider == "auto":
        resolved_provider = _auto_detect_provider()
    if resolved_provider == "groq":
        model = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return GroqProvider(model_name=model)
    if resolved_provider == "gemini":
        model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return GeminiProvider(model_name=model)
    if resolved_provider == "ollama":
        from src.llm.ollama_provider import OllamaProvider

        model = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        return OllamaProvider(
            model_name=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
        )
    raise ValueError(
        f"Unknown LLM provider: '{resolved_provider}'. "
        "Supported: auto, ollama, groq, gemini. Set LLM_PROVIDER accordingly."
    )


def _auto_detect_provider() -> str:
    """Prefer local Ollama, then configured cloud providers."""

    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if response.status_code == 200:
            return "ollama"
    except Exception:
        pass
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    raise ValueError(
        "No LLM provider is available. Start Ollama with 'ollama serve', "
        "set GROQ_API_KEY, or set GEMINI_API_KEY."
    )


def get_fallback_provider(primary: BaseLLMProvider) -> Optional[BaseLLMProvider]:
    """Return the alternate configured provider for failover when credentials exist."""

    if primary.provider_name == "groq" and os.getenv("GEMINI_API_KEY"):
        return get_llm_provider("gemini")
    if primary.provider_name == "gemini" and os.getenv("GROQ_API_KEY"):
        return get_llm_provider("groq")
    return None
