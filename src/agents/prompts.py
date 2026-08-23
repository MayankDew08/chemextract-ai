"""Prompt templates for Phase 2 extraction agents.

This file contains the entity, quantity, and condition prompt templates. Keeping
prompts centralized makes agent behavior reviewable and avoids prompt drift
between direct tests, demos, and LangGraph node execution.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

OLLAMA_JSON_RULES = """CRITICAL OUTPUT FORMAT RULES — FOLLOW EXACTLY:
1. Return ONLY valid JSON. No other text before or after.
2. Do NOT wrap in markdown code blocks. No backticks.
3. Do NOT add explanations or commentary.
4. Use double quotes only. Never single quotes.
5. No trailing commas before } or ].
6. Numeric values must be numbers not strings.
7. Missing values must be null, not \"null\", \"N/A\", or \"none\".
8. Start with { and end with }. Nothing else."""

ENTITY_FEW_SHOT = ('EXAMPLE: INPUT: Zinc acetate dihydrate (2.195 g, 10 mmol) was dissolved in 100 mL of methanol. KOH (1.12 g, 20 mmol) was added as catalyst and ZnO precipitate formed. OUTPUT: {"entities":[{"name":"Zinc acetate dihydrate","formula":null,"role":"REACTANT","quantity":null,"moles":null,"notes":null},{"name":"methanol","formula":null,"role":"SOLVENT","quantity":null,"moles":null,"notes":null},{"name":"KOH","formula":null,"role":"CATALYST","quantity":null,"moles":null,"notes":null},{"name":"ZnO","formula":"ZnO","role":"PRODUCT","quantity":null,"moles":null,"notes":null}],"extraction_notes":null}\n\nNOW EXTRACT FROM THE ACTUAL TEXT BELOW:\n')
QUANTITY_FEW_SHOT = ('EXAMPLE: INPUT: Zinc acetate (2.195 g, 10 mmol) was dissolved in 100 mL methanol. KOH (1.12 g, 20 mmol) was added. ENTITIES: [{"name":"Zinc acetate","role":"REACTANT"},{"name":"methanol","role":"SOLVENT"},{"name":"KOH","role":"CATALYST"}] OUTPUT: {"entities":[{"name":"Zinc acetate","formula":null,"role":"REACTANT","quantity":{"value":2.195,"unit":"g"},"moles":{"value":10.0,"unit":"mmol"},"notes":null},{"name":"methanol","formula":null,"role":"SOLVENT","quantity":{"value":100.0,"unit":"mL"},"moles":null,"notes":null},{"name":"KOH","formula":null,"role":"CATALYST","quantity":{"value":1.12,"unit":"g"},"moles":{"value":20.0,"unit":"mmol"},"notes":null}],"extraction_notes":null}\n\nNOW ALIGN QUANTITIES FOR THE ACTUAL INPUT BELOW:\n')
CONDITION_FEW_SHOT = ('EXAMPLE: INPUT: The solution was refluxed at 65°C for 2 hours under air at ambient pressure. OUTPUT: {"temperature_celsius":65.0,"duration_hours":2.0,"pressure_atm":1.0,"atmosphere":"air","technique":"reflux","yield_percent":null,"additional_conditions":{}}\n\nUNIT CONVERSION: Kelvin to Celsius subtract 273.15; Fahrenheit to Celsius (F - 32) * 5/9; minutes to hours divide by 60; days to hours multiply by 24.\n\nNOW EXTRACT CONDITIONS FROM THE ACTUAL TEXT BELOW:\n')


ENTITY_IDENTIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            OLLAMA_JSON_RULES.replace("{", "{{").replace("}", "}}") + """\n\nYou are a chemistry expert specializing in extracting chemical
entities from synthesis literature.

Your ONLY job: identify every chemical substance mentioned in the
provided text and assign each one a role.

ROLES:
- REACTANT:  A chemical that is consumed in the reaction to form product
- SOLVENT:   A liquid medium that dissolves other chemicals
- CATALYST:  Speeds up reaction, not consumed, not the main product
- ADDITIVE:  Modifying agent, surfactant, template, dopant
- PRODUCT:   The target compound being synthesized
- UNKNOWN:   Role cannot be determined from context

STRICT RULES:
1. Only extract chemicals explicitly named in the text
2. Do not invent or infer chemicals not mentioned
3. Do not include laboratory equipment
4. Do not include vague terms like solution, mixture, precipitate unless specific
5. If a chemical appears multiple times with different roles, use the primary role
6. Common solvents are SOLVENT
7. The compound being synthesized is PRODUCT
8. Return at minimum 1 entity
9. Extract formula only if it appears explicitly in parentheses

Output the EntityList schema exactly.""",
        ),
        ("human", ENTITY_FEW_SHOT.replace("{", "{{").replace("}", "}}") + "Text chunk:\n{text}"),
    ]
)


QUANTITY_ALIGNMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            OLLAMA_JSON_RULES.replace("{", "{{").replace("}", "}}") + """\n\nYou are a chemistry extraction specialist aligning measurements to entities.

Your ONLY job: for each provided chemical entity, find quantities and molar
amounts explicitly stated in the source text.

STRICT RULES:
1. Preserve every entity name and role from the input list
2. Never invent a quantity, unit, or mole amount
3. If no amount is stated for an entity, set quantity to null
4. If a separate mole amount is stated, set moles separately
5. Units must match the schema allowed units
6. Do not convert mass or volume units; copy the stated unit
7. Return the EntityList schema exactly.

QUANTITY FORMAT — THIS IS MANDATORY:
quantity and moles MUST be nested objects, not strings.
CORRECT: "quantity": {{"value": 2.195, "unit": "g"}}
         "moles": {{"value": 10.0, "unit": "mmol"}}
WRONG:   "quantity": "2.195 g"
         "moles": "10 mmol"
If no quantity or moles is found, return null for that field.""",
        ),
        ("human", QUANTITY_FEW_SHOT.replace("{", "{{").replace("}", "}}") + "Text chunk:\n{text}\n\nEntities from agent 1:\n{entities_json}"),
    ]
)


CONDITION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            OLLAMA_JSON_RULES.replace("{", "{{").replace("}", "}}") + """\n\nYou are a chemistry expert extracting reaction conditions.

Your ONLY job: extract physical reaction conditions from synthesis text.

STRICT RULES:
1. Convert all temperatures to Celsius
2. Convert all durations to hours
3. Set pressure_atm to null when pressure is not explicitly mentioned
4. Set atmosphere to null unless air, nitrogen, argon, vacuum, inert gas, or another atmosphere is explicitly mentioned
5. Extract the synthesis technique if stated or clearly indicated by verbs like refluxed, stirred, sonicated, hydrothermal
6. Extract yield_percent only when a percent yield is explicitly reported
7. Never invent missing temperature, duration, or yield

Output the ReactionConditions schema exactly.""",
        ),
        ("human", CONDITION_FEW_SHOT.replace("{", "{{").replace("}", "}}") + "Text chunk:\n{text}\n\nQuantified entities:\n{entities_json}"),
    ]
)
