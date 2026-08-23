"""Condition extraction agent for Phase 2 extraction.

This file contains ConditionExtractionAgent. It normalizes temperatures,
durations, atmosphere, pressure, technique, and yield into ReactionConditions.
"""

from __future__ import annotations

import logging
import re
import time

from src.agents.prompts import CONDITION_EXTRACTION_PROMPT
from src.agents.json_utils import parse_to_pydantic
from src.llm.base import BaseLLMProvider
from src.llm.resolver import get_fallback_provider
from src.schemas.chemical import EntityList
from src.schemas.recipe import ReactionConditions

logger = logging.getLogger(__name__)


class ConditionExtractionAgent:
    """ConditionExtractionAgent extracts normalized physical reaction settings."""

    def __init__(self, provider: BaseLLMProvider) -> None:
        """Store the selected provider while allowing provider fallback."""

        self._provider = provider

    async def extract(
        self,
        text: str,
        entities: EntityList,
    ) -> tuple[ReactionConditions, dict[str, float | int | str]]:
        """Extract conditions and return observability metadata."""

        started = time.perf_counter()
        provider_used = self._provider
        extraction_method = "primary_llm"
        try:
            result = await self._invoke_provider(provider_used, text, entities)
        except Exception as exc:
            logger.warning("Condition agent primary provider failed: %s", exc)
            fallback = get_fallback_provider(provider_used)
            if fallback is not None:
                try:
                    provider_used = fallback
                    result = await self._invoke_provider(provider_used, text, entities)
                    extraction_method = "fallback_llm"
                except Exception as fallback_exc:
                    logger.warning("Condition agent fallback provider failed: %s", fallback_exc)
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

    async def _invoke_provider(
        self,
        provider: BaseLLMProvider,
        text: str,
        entities: EntityList,
    ) -> ReactionConditions:
        """Invoke a provider and parse its raw response into ReactionConditions."""

        llm = provider.get_llm()
        messages = await CONDITION_EXTRACTION_PROMPT.ainvoke(
            {"text": text, "entities_json": entities.model_dump_json(indent=2)},
        )
        response = await llm.ainvoke(messages)
        response_text = response.content if isinstance(response.content, str) else str(response.content)
        result = parse_to_pydantic(response_text, ReactionConditions, label="condition_agent", coerce_numbers=True)
        if not isinstance(result, ReactionConditions):
            raise ValueError("Condition agent returned no schema-compatible result")
        return result

    def _deterministic_extract(self, text: str) -> ReactionConditions:
        """Extract common condition patterns for offline and provider-failure paths."""

        temperature = None
        temp_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:°C|C\b|celsius)", text, flags=re.IGNORECASE)
        if temp_match:
            temperature = float(temp_match.group(1))
        duration = None
        duration_match = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h\b|min(?:utes?)?|days?)", text, flags=re.IGNORECASE)
        if duration_match:
            value = float(duration_match.group(1))
            unit = duration_match.group(2).lower()
            if unit.startswith("min"):
                duration = value / 60.0
            elif unit.startswith("day"):
                duration = value * 24.0
            else:
                duration = value
        lowered = text.lower()
        technique = None
        for candidate in ["reflux", "hydrothermal", "stirring", "sonication", "calcination", "drying"]:
            if candidate in lowered or f"{candidate}ed" in lowered:
                technique = candidate
                break
        atmosphere = None
        for candidate in ["nitrogen", "argon", "vacuum"]:
            if candidate in lowered:
                atmosphere = candidate
                break
        if atmosphere is None and re.search(r"\b(?:under|in)\s+air\b|\bair\s+atmosphere\b", lowered):
            atmosphere = "air"
        pressure = None
        pressure_match = re.search(r"(\d+(?:\.\d+)?)\s*(atm|bar|kpa|mpa)\b", lowered)
        if pressure_match:
            pressure_value = float(pressure_match.group(1))
            pressure_unit = pressure_match.group(2)
            pressure = {
                "atm": pressure_value,
                "bar": pressure_value * 0.986923,
                "kpa": pressure_value / 101.325,
                "mpa": pressure_value * 9.86923,
            }[pressure_unit]
        elif "ambient pressure" in lowered:
            pressure = 1.0
        return ReactionConditions(
            temperature_celsius=temperature,
            duration_hours=duration,
            pressure_atm=pressure,
            atmosphere=atmosphere,
            technique=technique,
        )


def _estimate_tokens(input_text: str, output_text: str) -> int:
    """Estimate token count cheaply for observability when provider usage is absent."""

    return max(1, int((len(input_text.split()) + len(output_text.split())) * 1.3))
