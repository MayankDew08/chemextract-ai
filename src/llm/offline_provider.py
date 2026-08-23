"""Deterministic provider for demos and tests that must not use the network."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.llm.base import BaseLLMProvider


class OfflineProvider(BaseLLMProvider):
    """Force extraction agents onto their deterministic fallback paths."""

    @property
    def provider_name(self) -> str:
        """Return the provider identifier used in pipeline metadata."""

        return "offline"

    @property
    def model_name(self) -> str:
        """Return a stable model label for observability."""

        return "deterministic-fallback"

    def get_llm(self) -> BaseChatModel:
        """Fail immediately so callers use their local deterministic extractor."""

        raise RuntimeError("Offline provider intentionally disables LLM inference")

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return zero because deterministic extraction performs no inference."""

        return 0.0
