"""Verbatim-style standalone tests for Ollama JSON parsing; no LLM calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.json_utils import (
    coerce_numeric_strings,
    extract_json_from_text,
    parse_entity_list_resilient,
    parse_quantity_string,
    parse_to_pydantic,
    repair_json,
    normalize_entity_quantities,
)
from src.schemas.chemical import EntityList
from src.schemas.recipe import ReactionConditions

CLEAN_ENTITY_JSON = '{"entities":[{"name":"Zinc acetate dihydrate","formula":"Zn(CH3COO)2","role":"REACTANT","quantity":null,"moles":null,"notes":null},{"name":"methanol","formula":"CH3OH","role":"SOLVENT","quantity":null,"moles":null,"notes":null},{"name":"ZnO","formula":"ZnO","role":"PRODUCT","quantity":null,"moles":null,"notes":null}],"extraction_notes":null}'
CLEAN_CONDITION_JSON = '{"temperature_celsius":65.0,"duration_hours":2.0,"pressure_atm":1.0,"atmosphere":"air","technique":"reflux","yield_percent":null,"additional_conditions":{}}'


def check(condition: bool, message: str) -> None:
    """Raise immediately so the standalone command has a useful failure."""

    if not condition:
        raise AssertionError(message)
    print(f"  ✅ {message}")


def test_ollama_parsing() -> None:
    """Run all Ollama JSON and quantity parsing checks under pytest."""

    print("\n" + "═" * 60)
    print("  JSON UTILS VERBAT TEST")
    print("═" * 60)

    check(extract_json_from_text(CLEAN_ENTITY_JSON).startswith("{"), "Clean JSON passed through")
    md_wrapped = f"```json\n{CLEAN_ENTITY_JSON}\n```"
    check(extract_json_from_text(md_wrapped).startswith("{"), "Markdown json block extracted")
    check(extract_json_from_text(f"Here are entities:\n{CLEAN_ENTITY_JSON}").startswith("{"), "Preamble text removed")
    check(extract_json_from_text(f"{CLEAN_ENTITY_JSON}\nDone").startswith("{"), "Postamble text removed")

    check(",}" not in repair_json('{"entities":[{"name":"ZnO",}]}'), "Trailing commas removed")
    repaired = repair_json('{"value": None, "active": True, "done": False}')
    check('"value": null' in repaired and '"active": true' in repaired and '"done": false' in repaired, "Python literals repaired")

    numeric = coerce_numeric_strings({"value": "2.195", "count": "10", "name": "ZnO"})
    check(numeric["value"] == 2.195 and isinstance(numeric["value"], float), "String float coerced")
    check(numeric["count"] == 10 and numeric["name"] == "ZnO", "Integer and text coercion correct")

    parsed = parse_to_pydantic(CLEAN_ENTITY_JSON, EntityList, "clean")
    check(parsed is not None and len(parsed.entities) == 3, "Clean EntityList parsed")
    check(parse_to_pydantic(md_wrapped, EntityList, "markdown") is not None, "Markdown EntityList parsed")
    string_values = '{"entities":[{"name":"ZnO","formula":"ZnO","role":"PRODUCT","quantity":{"value":"2.195","unit":"g"},"moles":null,"notes":null}],"extraction_notes":null}'
    parsed = parse_to_pydantic(string_values, EntityList, "numbers")
    check(parsed is not None and parsed.entities[0].quantity.value == 2.195, "Nested numeric string parsed")
    check(parse_to_pydantic("not JSON", EntityList, "broken") is None, "Broken input returns None")

    conditions = parse_to_pydantic(CLEAN_CONDITION_JSON, ReactionConditions, "conditions")
    check(conditions is not None and conditions.temperature_celsius == 65.0, "Condition JSON parsed")
    check(conditions is not None and conditions.duration_hours == 2.0, "Duration parsed")
    direct_array = '[{"name":"ZnO","formula":"ZnO","role":"PRODUCT","quantity":null,"moles":null,"notes":null}]'
    check(parse_entity_list_resilient(direct_array, "array") is not None, "Direct entity array wrapped")
    check(parse_entity_list_resilient('{"chemicals":' + direct_array + '}', "alternate") is not None, "Alternate entity key recognized")
    numeric_entity = '{"entities":[{"name":"Titanium(IV) isopropoxide","role":"REACTANT","quantity":2.84,"moles":10}]}'
    numeric_result = parse_entity_list_resilient(numeric_entity, "numeric_entity")
    check(numeric_result is not None and numeric_result.entities[0].name == "Titanium(IV) isopropoxide", "Numeric entity quantities tolerated")

    print("\n── parse_quantity_string ──")
    for value, expected, label in (
        ("2.195 g", {"value": 2.195, "unit": "g"}, "2.195 g parsed"),
        ("10 mmol", {"value": 10.0, "unit": "mmol"}, "10 mmol parsed"),
        ("100 mL", {"value": 100.0, "unit": "mL"}, "100 mL parsed"),
        ("0.5 wt%", {"value": 0.5, "unit": "wt%"}, "0.5 wt% parsed"),
        ("1.2e-3 mol", {"value": 0.0012, "unit": "mol"}, "Scientific notation parsed"),
        ("2.195g", {"value": 2.195, "unit": "g"}, "No-space quantity parsed"),
    ):
        check(parse_quantity_string(value) == expected, label)
    check(parse_quantity_string(None) is None, "None returns None")
    check(parse_quantity_string("some text") is None, "Non-quantity string returns None")

    print("\n── normalize_entity_quantities ──")
    combined = {"entities": [
        {"name": "Zinc acetate dihydrate", "role": "REACTANT", "quantity": "2.195 g", "moles": "10 mmol"},
        {"name": "methanol", "role": "SOLVENT", "quantity": "100 mL", "moles": None},
        {"name": "ZnO", "role": "PRODUCT", "quantity": None, "moles": None},
    ]}
    normalized = normalize_entity_quantities(combined)
    check(normalized["entities"][0]["quantity"] == {"value": 2.195, "unit": "g"}, "Zinc quantity value correct")
    check(normalized["entities"][0]["moles"] == {"value": 10.0, "unit": "mmol"}, "Zinc moles value correct")
    check(normalized["entities"][1]["quantity"] == {"value": 100.0, "unit": "mL"}, "Methanol volume correct")
    check(normalized["entities"][2]["quantity"] is None, "Product quantity stays null")

    print("\n── full parse with combined strings ──")
    combined_json = '{"entities":[{"name":"Zinc acetate dihydrate","formula":null,"role":"REACTANT","quantity":"2.195 g","moles":"10 mmol","notes":null},{"name":"KOH","formula":null,"role":"CATALYST","quantity":"1.12 g","moles":"20 mmol","notes":null}],"extraction_notes":null}'
    combined_result = parse_to_pydantic(combined_json, EntityList, "test_combined")
    check(combined_result is not None, "Combined string JSON parses to EntityList")
    if combined_result is not None:
        check(combined_result.entities[0].quantity.value == 2.195, "Zinc quantity value is 2.195")
        check(combined_result.entities[0].quantity.unit == "g", "Zinc quantity unit is g")
        check(combined_result.entities[0].moles.value == 10.0, "Zinc moles value is 10.0")
        check(combined_result.entities[0].moles.unit == "mmol", "Zinc moles unit is mmol")

    json.loads(CLEAN_ENTITY_JSON)
    json.loads(CLEAN_CONDITION_JSON)
    print("\n  ✅ ALL TESTS PASSED — JSON utils ready for Ollama")


if __name__ == "__main__":
    test_ollama_parsing()
