"""Robust JSON parsing helpers for local and cloud LLM responses."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

KNOWN_UNITS = {
    "g", "mg", "kg", "ug", "µg", "ml", "l", "ul", "µl", "mm",
    "mol", "mmol", "umol", "µmol", "nmol", "m", "mm", "um", "µm", "n",
    "wt%", "vol%", "%", "mol%", "eq", "equiv", "v/v", "w/w", "w/v",
}

QUANTITY_PATTERN = re.compile(
    r"^(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*"
    r"(g|mg|kg|ug|µg|mL|L|uL|µL|ml|l|"
    r"mol|mmol|umol|µmol|nmol|M|mM|uM|µM|N|"
    r"wt%|vol%|mol%|%|eq|equiv|v/v|w/w|w/v)$",
    re.IGNORECASE,
)


def _normalize_unit(unit: str) -> str:
    """Normalize model unit spelling to the schema's canonical spelling."""

    canonical = {
        "ml": "mL", "ul": "µL", "ug": "µg", "umol": "µmol", "um": "µM",
        "g": "g", "mg": "mg", "kg": "kg", "l": "L", "mol": "mol",
        "mmol": "mmol", "nmol": "nmol", "m": "M", "mm": "mM", "n": "N",
        "wt%": "wt%", "vol%": "vol%", "mol%": "mol%", "%": "%",
        "eq": "eq", "equiv": "equiv", "v/v": "v/v", "w/w": "w/w", "w/v": "w/v",
    }
    return canonical.get(unit.lower(), unit)


def parse_quantity_string(value: str) -> Optional[dict]:
    """Parse a combined quantity such as ``2.195 g`` into value and unit."""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        value = value.strip()
        match = QUANTITY_PATTERN.match(value)
        if match:
            return {
                "value": float(match.group(1)),
                "unit": _normalize_unit(match.group(2)),
            }

        parts = value.rsplit(None, 1)
        if len(parts) == 2 and parts[1].lower() in {unit.lower() for unit in KNOWN_UNITS}:
            return {"value": float(parts[0]), "unit": _normalize_unit(parts[1])}
    except (ValueError, IndexError, TypeError):
        return None
    return None


def normalize_entity_quantities(data: Any) -> Any:
    """Recursively convert string quantity fields while preserving other data."""

    if isinstance(data, dict):
        return {
            key: parse_quantity_string(value)
            if key in ("quantity", "moles") and isinstance(value, str)
            else normalize_entity_quantities(value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [normalize_entity_quantities(value) for value in data]
    return data


def extract_json_from_text(text: str) -> str:
    """Extract an object or array from preambles, postambles, or code blocks."""

    if not text:
        return text
    text = text.strip()
    for pattern in (
        re.compile(r"```json\s*\n?(.*?)\n?\s*```", re.IGNORECASE | re.DOTALL),
        re.compile(r"```\s*\n?(.*?)\n?\s*```", re.DOTALL),
        re.compile(r"`(.*?)`", re.DOTALL),
    ):
        match = pattern.search(text)
        if match:
            candidate = match.group(1).strip()
            if candidate.startswith(("{", "[")):
                return candidate

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(text[start:], start):
            if escaped:
                escaped = False
                continue
            if char == "\\" and in_string:
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
            if in_string:
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return text


def repair_json(json_str: str) -> str:
    """Repair common JSON mistakes without executing model-generated text."""

    if not json_str or not json_str.strip():
        return json_str
    try:
        result = json_str.strip()
        result = re.sub(r":\s*None\b", ": null", result)
        result = re.sub(r":\s*True\b", ": true", result)
        result = re.sub(r":\s*False\b", ": false", result)
        result = re.sub(r",\s*([}\]])", r"\1", result)
        result = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', result)
        if "'" in result:
            result = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', result)

        stack: list[str] = []
        pairs = {"}": "{", "]": "["}
        for char in result:
            if char in "{[":
                stack.append(char)
            elif char in "}]" and stack and stack[-1] == pairs[char]:
                stack.pop()
        for opener in reversed(stack):
            result += "}" if opener == "{" else "]"
        return result
    except Exception as exc:
        logger.debug("JSON repair failed: %s", exc)
        return json_str


def coerce_numeric_strings(data: Any) -> Any:
    """Recursively convert strings containing only numbers to numeric values."""

    if isinstance(data, dict):
        return {key: coerce_numeric_strings(value) for key, value in data.items()}
    if isinstance(data, list):
        return [coerce_numeric_strings(value) for value in data]
    if isinstance(data, str):
        value = data.strip()
        if not value:
            return data
        try:
            if "." not in value and "e" not in value.lower():
                return int(value)
            return float(value)
        except (ValueError, OverflowError):
            return data
    return data


def parse_to_pydantic(
    text: str,
    model_class: type[T],
    label: str = "",
    coerce_numbers: bool = True,
) -> Optional[T]:
    """Parse model output through direct, extracted, and repaired attempts."""

    if not text or not text.strip():
        logger.warning("[%s] Empty LLM response", label)
        return None
    extracted = extract_json_from_text(text)
    attempts = [text, extracted, repair_json(text), repair_json(extracted)]
    for candidate in attempts:
        try:
            data = json.loads(candidate)
            if coerce_numbers:
                data = coerce_numeric_strings(data)
            data = normalize_entity_quantities(data)
            return model_class.model_validate(data)
        except Exception:
            continue
    logger.warning("Could not parse local model JSON%s", f" ({label})" if label else "")
    return None


def parse_entity_list_resilient(text: str, label: str = "") -> Optional[Any]:
    """Parse EntityList and tolerate direct arrays or alternate list keys."""

    from src.schemas.chemical import ChemicalEntity, EntityList

    result = parse_to_pydantic(text, EntityList, label)
    if result is not None:
        return result
    try:
        data = coerce_numeric_strings(json.loads(repair_json(extract_json_from_text(text))))
        items = data if isinstance(data, list) else None
        if items is None and isinstance(data, dict):
            for key in ("entities", "chemicals", "compounds", "substances", "ingredients", "reactants", "chemical_entities", "results", "items", "materials"):
                if isinstance(data.get(key), list):
                    items = data[key]
                    break
            if items is None and "name" in data and "role" in data:
                items = [data]
        entities = []
        for item in items or []:
            if isinstance(item, dict) and "name" in item:
                try:
                    entity_data = dict(item)
                    # The entity agent is not responsible for measurement
                    # alignment, but small local models often emit bare
                    # numbers here despite the prompt. Treat those fields as
                    # absent so the quantity agent can align them from source.
                    for field_name in ("quantity", "moles"):
                        if entity_data.get(field_name) is not None and not isinstance(entity_data[field_name], dict):
                            entity_data[field_name] = None
                    entities.append(ChemicalEntity.model_validate(entity_data))
                except Exception:
                    continue
        return EntityList(entities=entities) if entities else None
    except Exception as exc:
        logger.debug("[%s] resilient entity parsing failed: %s", label, exc)
        return None
