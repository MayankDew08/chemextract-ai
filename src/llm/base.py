"""Abstract LLM provider contract for extraction agents.

This file contains BaseLLMProvider. Agents request a LangChain chat model from
this abstraction so adding a later provider such as Ollama requires one new
provider class and resolver wiring rather than agent rewrites.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """BaseLLMProvider is the stable interface every extraction agent depends on."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a unique provider identifier such as groq or gemini."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the concrete model identifier used for extraction."""

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """Return a lazily initialized LangChain chat model ready for structured output."""

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost for observability without coupling agents to pricing."""

    def __repr__(self) -> str:
        """Represent providers with the active model for debugging."""

        return f"{self.__class__.__name__}(model={self.model_name})"
