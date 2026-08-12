"""Groq LLM provider for Phase 2 extraction agents.

This file contains GroqProvider. Groq is the default Phase 2 provider because
the 8B Llama model is fast enough for three small structured extraction passes.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_core.language_models import BaseChatModel

from src.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class GroqProvider(BaseLLMProvider):
    """GroqProvider lazily constructs a deterministic ChatGroq model."""

    def __init__(self, model_name: str = "llama-3.1-8b-instant") -> None:
        """Store model choice so resolver can switch models through env vars."""

        self._model_name = model_name
        self._llm: Optional[BaseChatModel] = None

    @property
    def provider_name(self) -> str:
        """Return Groq's provider identifier."""

        return "groq"

    @property
    def model_name(self) -> str:
        """Return the configured Groq model name."""

        return self._model_name

    def get_llm(self) -> BaseChatModel:
        """Return a cached ChatGroq client configured for deterministic extraction."""

        if self._llm is None:
            from langchain_groq import ChatGroq

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set")
            self._llm = ChatGroq(
                model=self._model_name,
                temperature=0.0,
                groq_api_key=api_key,
                max_retries=2,
            )
        return self._llm

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return zero because this project treats Groq free-tier usage as free."""

        return 0.0
