"""Gemini fallback provider for Phase 2 extraction agents.

This file contains GeminiProvider. It exists as a fallback when Groq rate limits
or fails, and defaults to Gemini 2.5 Flash because Gemini 1.5 Flash is no longer
available in the current API environment.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_core.language_models import BaseChatModel

from src.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """GeminiProvider lazily constructs a deterministic Gemini chat model."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        """Store model choice so resolver can switch models through env vars."""

        self._model_name = model_name
        self._llm: Optional[BaseChatModel] = None

    @property
    def provider_name(self) -> str:
        """Return Gemini's provider identifier."""

        return "gemini"

    @property
    def model_name(self) -> str:
        """Return the configured Gemini model name."""

        return self._model_name

    def get_llm(self) -> BaseChatModel:
        """Return a cached Gemini client configured for deterministic extraction."""

        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            self._llm = ChatGoogleGenerativeAI(
                model=self._model_name,
                temperature=0.0,
                google_api_key=api_key,
                convert_system_message_to_human=True,
            )
        return self._llm

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate Gemini 2.5 Flash cost using conservative flash-class pricing."""

        return (input_tokens / 1000 * 0.000075) + (output_tokens / 1000 * 0.000300)
