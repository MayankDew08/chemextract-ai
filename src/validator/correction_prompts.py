"""Correction prompt builders for Phase 3 retry loops.

This file contains one prompt builder per responsible agent. The builders turn
deterministic validation errors into focused retry context so the next agent
call fixes only the failing fields instead of repeating the original prompt.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.schemas.chemical import ChemicalEntity, EntityList
from src.schemas.recipe import ReactionConditions
from src.validator.error_types import ValidationError

logger = logging.getLogger(__name__)


def build_entity_correction_prompt(
    source_text: str,
    previous_entities: list[ChemicalEntity],
    errors: list[ValidationError],
    attempt: int,
    max_attempts: int,
) -> str:
    """Build retry context for entity role/name correction."""

    return _build_prompt(
        heading=f"CORRECTION REQUIRED (Attempt {attempt}/{max_attempts})",
        error_title="Your previous entity extraction had these errors:",
        errors=errors,
        payload_title="Previous extraction:",
        payload=EntityList(entities=previous_entities).model_dump_json(indent=2),
        source_text=source_text,
        instructions=[
            "Fix ONLY the errors listed above",
            "Do not change entities that were correctly identified",
            "Return the complete corrected EntityList",
            f"Pay special attention to: {_field_summary(errors)}",
        ],
    )


def build_quantity_correction_prompt(
    source_text: str,
    current_entities: list[ChemicalEntity],
    errors: list[ValidationError],
    attempt: int,
    max_attempts: int,
) -> str:
    """Build retry context for quantity and unit correction."""

    return _build_prompt(
        heading=f"CORRECTION REQUIRED (Attempt {attempt}/{max_attempts})",
        error_title="Quantity alignment errors found:",
        errors=errors,
        payload_title="Current entities (with incorrect quantities):",
        payload=EntityList(entities=current_entities).model_dump_json(indent=2),
        source_text=source_text,
        instructions=[
            "For each entity listed in the errors, re-read the source text",
            "Find the exact number and unit immediately near that chemical name",
            "Units must be from: g, mg, kg, mL, L, mmol, mol, M, wt%, vol%",
            "If truly no quantity is mentioned, set quantity=None",
            "Do not invent quantities",
        ],
    )


def build_condition_correction_prompt(
    source_text: str,
    previous_conditions: Optional[ReactionConditions],
    errors: list[ValidationError],
    attempt: int,
    max_attempts: int,
) -> str:
    """Build retry context for condition normalization correction."""

    return _build_prompt(
        heading=f"CORRECTION REQUIRED (Attempt {attempt}/{max_attempts})",
        error_title="Condition extraction errors found:",
        errors=errors,
        payload_title="Previous conditions:",
        payload=previous_conditions.model_dump_json(indent=2) if previous_conditions else "null",
        source_text=source_text,
        instructions=[
            "If temperature failed, check whether the source used Kelvin, Celsius, or Fahrenheit",
            "If duration failed, check whether the source used minutes, hours, or days",
            "Return only conditions explicitly supported by the source text",
            "Do not invent missing temperature, duration, pressure, or yield",
        ],
    )


def _build_prompt(
    heading: str,
    error_title: str,
    errors: list[ValidationError],
    payload_title: str,
    payload: str,
    source_text: str,
    instructions: list[str],
) -> str:
    """Assemble a correction prompt consistently across agent types."""

    formatted_errors = "\n".join(
        f"- {error.error_type.value} at {error.field_path}: {error.message} Suggested fix: {error.suggested_fix}"
        for error in errors
    )
    formatted_instructions = "\n".join(f"- {instruction}" for instruction in instructions)
    return (
        f"{heading}\n\n"
        f"{error_title}\n{formatted_errors}\n\n"
        f"{payload_title}\n{payload}\n\n"
        f"Original source text:\n{source_text}\n\n"
        f"Instructions:\n{formatted_instructions}"
    )


def _field_summary(errors: list[ValidationError]) -> str:
    """Summarize failing fields so retry prompts stay targeted."""

    return ", ".join(error.field_path for error in errors) or "none"
