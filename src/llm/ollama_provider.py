"""Local Ollama provider integration for ChemExtract AI."""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from langchain_core.language_models import BaseChatModel

from src.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Expose a locally running Ollama model through the provider contract."""

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 1024,
        timeout: int = 120,
    ) -> None:
        """Store local model settings without connecting until first use."""

        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._timeout = timeout
        self._llm: Optional[BaseChatModel] = None

    @property
    def provider_name(self) -> str:
        """Return the provider identifier used by agent dispatch."""

        return "ollama"

    @property
    def model_name(self) -> str:
        """Return the configured Ollama model tag."""

        return self._model_name

    def get_llm(self) -> BaseChatModel:
        """Verify Ollama and the model, then return a cached ChatOllama client."""

        if self._llm is not None:
            return self._llm

        models = self._get_models()
        available = [str(model.get("name", "")) for model in models]
        if not any(name == self._model_name for name in available):
            raise RuntimeError(
                f"Model '{self._model_name}' not pulled.\n"
                f"Pull with: ollama pull {self._model_name}\n"
                f"Available: {', '.join(available) or 'none'}"
            )

        from langchain_ollama import ChatOllama

        self._llm = ChatOllama(
                model=self._model_name,
                base_url=self._base_url,
                temperature=0.0,
                num_ctx=4096,       # was 8192
                num_predict=512,    # was 1024
                format="json",
                top_k=10,
                top_p=0.9,
                repeat_penalty=1.1,
                timeout=60,         # was 180
            )
        return self._llm

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return zero because local inference has no API token charge."""

        return 0.0

    def is_available(self) -> bool:
        """Return whether the Ollama tags endpoint responds successfully."""

        try:
            response = httpx.get(f"{self._base_url}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def _get_models(self) -> list[dict]:
        """Fetch installed model metadata and provide actionable errors."""

        try:
            response = httpx.get(f"{self._base_url}/api/tags", timeout=3.0)
            response.raise_for_status()
            payload = response.json()
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise RuntimeError(
                "Ollama server not running.\n"
                "Start with: ollama serve\n"
                "Check status: systemctl status ollama"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Could not query Ollama at {self._base_url}: {exc}") from exc

        models = payload.get("models", [])
        return models if isinstance(models, list) else []
