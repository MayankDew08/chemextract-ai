"""Entity identification agent for Phase 2 extraction.

This file contains EntityIdentificationAgent. It uses the provider abstraction
for structured LLM extraction and falls back to deterministic chemistry regexes
when provider calls fail so the graph can degrade gracefully.
"""

from __future__ import annotations

import logging
import re
import time

from src.agents.prompts import ENTITY_IDENTIFICATION_PROMPT
from src.agents.json_utils import parse_entity_list_resilient
from src.llm.base import BaseLLMProvider
from src.llm.resolver import get_fallback_provider
from src.schemas.chemical import ChemicalEntity, ChemicalRole, EntityList

logger = logging.getLogger(__name__)


class EntityIdentificationAgent:
    """EntityIdentificationAgent extracts chemical names and roles from raw text."""

    def __init__(self, provider: BaseLLMProvider) -> None:
        """Store the selected provider while keeping failover provider-independent."""

        self._provider = provider

    async def extract(self, text: str) -> tuple[EntityList, dict[str, float | int | str]]:
        """Extract entities and return observability metadata for the graph state."""

        started = time.perf_counter()
        provider_used = self._provider
        extraction_method = "primary_llm"
        try:
            result = await self._invoke_provider(provider_used, text)
            if not result.entities:
                logger.warning("Entity agent returned no entities; using deterministic extraction")
                result = self._deterministic_extract(text)
                extraction_method = "deterministic_fallback"
        except Exception as exc:
            logger.warning("Entity agent primary provider failed: %s", exc)
            if _is_empty_entity_schema_error(exc):
                result = self._deterministic_extract(text)
                extraction_method = "deterministic_fallback"
            else:
                fallback = get_fallback_provider(provider_used)
                if fallback is not None:
                    try:
                        provider_used = fallback
                        result = await self._invoke_provider(provider_used, text)
                        extraction_method = "fallback_llm"
                    except Exception as fallback_exc:
                        logger.warning("Entity agent fallback provider failed: %s", fallback_exc)
                        result = self._deterministic_extract(text)
                        extraction_method = "deterministic_fallback"
                else:
                    result = self._deterministic_extract(text)
                    extraction_method = "deterministic_fallback"
        tokens = _estimate_tokens(text, result.model_dump_json())
        return result, {
            "latency": time.perf_counter() - started,
            "tokens": tokens,
            "provider": provider_used.provider_name,
            "model": provider_used.model_name,
            "extraction_method": extraction_method,
        }

    async def _invoke_provider(self, provider: BaseLLMProvider, text: str) -> EntityList:
        """Invoke a provider and parse its raw response into EntityList."""

        llm = provider.get_llm()
        messages = await ENTITY_IDENTIFICATION_PROMPT.ainvoke({"text": text})
        response = await llm.ainvoke(messages)
        response_text = response.content if isinstance(response.content, str) else str(response.content)
        result = parse_entity_list_resilient(response_text, label="entity_agent")
        return result if isinstance(result, EntityList) else EntityList(entities=[])

    def _deterministic_extract(self, text: str) -> EntityList:
        """Extract common synthesis entities with regexes for offline robustness."""

        candidates: list[ChemicalEntity] = []
        solvent_names = ["methanol", "ethanol", "water", "acetone", "dmf", "dmso", "toluene", "hexane"]
        for solvent in solvent_names:
            if re.search(rf"\b{re.escape(solvent)}\b", text, flags=re.IGNORECASE):
                candidates.append(ChemicalEntity(name=solvent.title(), role=ChemicalRole.SOLVENT))
        named_patterns = [
            (r"\bZinc acetate dihydrate\b", ChemicalRole.REACTANT),
            (r"\bZinc acetate\b", ChemicalRole.REACTANT),
            (r"\bKOH\b", ChemicalRole.CATALYST),
            (r"\bpotassium hydroxide\b", ChemicalRole.CATALYST),
            (r"\bZnO\b", ChemicalRole.PRODUCT),
            (r"\bzinc oxide\b", ChemicalRole.PRODUCT),
        ]
        seen = {entity.name.lower() for entity in candidates}
        for pattern, role in named_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match and match.group(0).lower() not in seen:
                candidates.append(ChemicalEntity(name=match.group(0), role=role))
                seen.add(match.group(0).lower())
        return EntityList(entities=candidates, extraction_notes="Deterministic fallback extraction used.")


def _estimate_tokens(input_text: str, output_text: str) -> int:
    """Estimate token count cheaply for observability when provider usage is absent."""

    return max(1, int((len(input_text.split()) + len(output_text.split())) * 1.3))


def _is_empty_entity_schema_error(error: Exception) -> bool:
    """Identify provider errors caused by an empty structured entity list."""

    message = str(error).lower()
    return (
        "tool call validation failed" in message
        and "entities" in message
        and ("minimum 1" in message or "min_length" in message or "at least 1" in message)
    )
