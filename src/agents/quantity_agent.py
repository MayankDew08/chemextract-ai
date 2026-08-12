"""Quantity alignment agent for Phase 2 extraction.

This file contains QuantityAlignmentAgent. It enriches identified entities with
explicit quantities and molar amounts while preserving the entity list produced
by the previous node.
"""

from __future__ import annotations

import logging
import re
import time

from src.agents.prompts import QUANTITY_ALIGNMENT_PROMPT
from src.agents.json_utils import parse_to_pydantic
from src.llm.base import BaseLLMProvider
from src.llm.resolver import get_fallback_provider
from src.schemas.chemical import ChemicalEntity, EntityList, Quantity

logger = logging.getLogger(__name__)


class QuantityAlignmentAgent:
    """QuantityAlignmentAgent aligns explicit measurements to known entities."""

    def __init__(self, provider: BaseLLMProvider) -> None:
        """Store the selected provider while allowing automatic fallback."""

        self._provider = provider

    async def align(
        self,
        text: str,
        entities: list[ChemicalEntity],
    ) -> tuple[EntityList, dict[str, float | int | str]]:
        """Fill quantity fields and return observability metadata."""

        started = time.perf_counter()
        provider_used = self._provider
        input_list = EntityList(entities=entities)
        try:
            result = await self._invoke_provider(provider_used, text, input_list)
        except Exception as exc:
            logger.warning("Quantity agent primary provider failed: %s", exc)
            fallback = get_fallback_provider(provider_used)
            if fallback is not None:
                try:
                    provider_used = fallback
                    result = await self._invoke_provider(provider_used, text, input_list)
                except Exception as fallback_exc:
                    logger.warning("Quantity agent fallback provider failed: %s", fallback_exc)
                    result = self._deterministic_align(text, entities)
            else:
                result = self._deterministic_align(text, entities)
        tokens = _estimate_tokens(text, result.model_dump_json())
        return result, {
            "latency": time.perf_counter() - started,
            "tokens": tokens,
            "provider": provider_used.provider_name,
            "model": provider_used.model_name,
        }

    async def _invoke_provider(self, provider: BaseLLMProvider, text: str, entities: EntityList) -> EntityList:
        """Invoke a provider and parse its raw response into EntityList."""

        llm = provider.get_llm()
        messages = await QUANTITY_ALIGNMENT_PROMPT.ainvoke(
            {"text": text, "entities_json": entities.model_dump_json(indent=2)},
        )
        response = await llm.ainvoke(messages)
        response_text = response.content if isinstance(response.content, str) else str(response.content)
        result = parse_to_pydantic(response_text, EntityList, label="quantity_agent", coerce_numbers=True)
        return result if isinstance(result, EntityList) else self._deterministic_align(text, entities.entities)

    def _deterministic_align(self, text: str, entities: list[ChemicalEntity]) -> EntityList:
        """Align nearby parenthesized quantities for the common demo synthesis style."""

        enriched = []
        for entity in entities:
            quantity = entity.quantity
            moles = entity.moles
            window = _text_window_after_entity(text, entity.name)
            if window:
                measures = _extract_quantities(window)
                for measure in measures:
                    if measure.unit in {"mol", "mmol", "umol", "µmol", "nmol"} and moles is None:
                        moles = measure
                    elif quantity is None:
                        quantity = measure
            enriched.append(entity.model_copy(update={"quantity": quantity, "moles": moles}))
        return EntityList(entities=enriched, extraction_notes="Deterministic fallback quantity alignment used.")


def _text_window_after_entity(text: str, entity_name: str) -> str:
    """Return text near an entity mention so regex quantity alignment stays scoped."""

    match = re.search(re.escape(entity_name), text, flags=re.IGNORECASE)
    if not match:
        return ""
    return text[match.start() : match.end() + 120]


def _extract_quantities(text: str) -> list[Quantity]:
    """Extract schema-compatible quantities from a text window."""

    quantities = []
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(g|mg|kg|mL|L|ml|l|mmol|mol|M|mM|wt%|vol%|%)")
    for value, unit in pattern.findall(text):
        try:
            quantities.append(Quantity(value=float(value), unit=unit))
        except ValueError:
            continue
    return quantities


def _estimate_tokens(input_text: str, output_text: str) -> int:
    """Estimate token count cheaply for observability when provider usage is absent."""

    return max(1, int((len(input_text.split()) + len(output_text.split())) * 1.3))
