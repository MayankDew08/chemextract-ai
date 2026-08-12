"""Standalone deterministic validation rules for Phase 3.

This file contains all 15 rule functions. Each rule accepts one ChemicalRecipe
and returns either one structured ValidationError or None; rules share no mutable
state so the engine is deterministic and free of LLM calls.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.schemas.chemical import ChemicalRole
from src.schemas.recipe import ChemicalRecipe
from src.validator.error_types import ErrorSeverity, ErrorType, ResponsibleAgent, ValidationError

logger = logging.getLogger(__name__)

ALLOWED_UNITS = {
    "g",
    "mg",
    "kg",
    "ug",
    "µg",
    "mL",
    "L",
    "uL",
    "µL",
    "ml",
    "l",
    "mol",
    "mmol",
    "umol",
    "µmol",
    "nmol",
    "M",
    "mM",
    "uM",
    "µM",
    "N",
    "wt%",
    "vol%",
    "%",
    "mol%",
    "eq",
    "equiv",
    "v/v",
    "w/w",
    "w/v",
}


def check_unit_missing(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Block recipes where consumed chemicals lack explicit quantities."""

    for index, entity in enumerate(recipe.entities):
        unit_missing = entity.quantity is None or not str(getattr(entity.quantity, "unit", "")).strip()
        if entity.role in {ChemicalRole.REACTANT, ChemicalRole.CATALYST} and unit_missing:
            return ValidationError(
                error_type=ErrorType.UNIT_MISSING,
                severity=ErrorSeverity.BLOCKING,
                responsible_agent=ResponsibleAgent.QUANTITY_AGENT,
                field_path=f"entities[{index}].quantity",
                entity_name=entity.name,
                actual_value="None" if entity.quantity is None else str(entity.quantity),
                expected="Quantity with unit",
                message=f"Reactant/catalyst '{entity.name}' has no quantity. Find the amount in the source text.",
                suggested_fix=f"Look for a number followed by a unit (g, mg, mL, mmol) near '{entity.name}' in the text.",
            )
        if entity.role == ChemicalRole.SOLVENT and unit_missing:
            return ValidationError(
                error_type=ErrorType.UNIT_MISSING,
                severity=ErrorSeverity.WARNING,
                responsible_agent=ResponsibleAgent.QUANTITY_AGENT,
                field_path=f"entities[{index}].quantity",
                entity_name=entity.name,
                actual_value="None",
                expected="Optional solvent quantity",
                message=f"Solvent '{entity.name}' has no quantity. Abstracts may omit solvent amounts.",
                suggested_fix=f"If available, find the volume near '{entity.name}' in the source text.",
            )
    return None


def check_unit_invalid(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Block recipes containing measurement units outside the allowed set."""

    for index, entity in enumerate(recipe.entities):
        for field_name, quantity in (("quantity", entity.quantity), ("moles", entity.moles)):
            unit = getattr(quantity, "unit", None) if quantity is not None else None
            if quantity is not None and str(unit or "").strip() and unit not in ALLOWED_UNITS:
                return ValidationError(
                    error_type=ErrorType.UNIT_INVALID,
                    severity=ErrorSeverity.BLOCKING,
                    responsible_agent=ResponsibleAgent.QUANTITY_AGENT,
                    field_path=f"entities[{index}].{field_name}.unit",
                    entity_name=entity.name,
                    actual_value=str(unit),
                    expected="Known chemistry unit",
                    message=f"Entity '{entity.name}' has invalid unit '{getattr(quantity, 'unit', None)}'.",
                    suggested_fix="Use only units explicitly present in the source text and supported by the schema.",
                )
    return None


def check_quantity_negative(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Block non-positive quantities because physical inputs must be positive."""

    for index, entity in enumerate(recipe.entities):
        for field_name, quantity in (("quantity", entity.quantity), ("moles", entity.moles)):
            if quantity is not None and getattr(quantity, "value", 0) <= 0:
                return ValidationError(
                    error_type=ErrorType.QUANTITY_NEGATIVE,
                    severity=ErrorSeverity.BLOCKING,
                    responsible_agent=ResponsibleAgent.QUANTITY_AGENT,
                    field_path=f"entities[{index}].{field_name}.value",
                    entity_name=entity.name,
                    actual_value=str(getattr(quantity, "value", None)),
                    expected="Positive numeric value",
                    message=f"Entity '{entity.name}' has non-positive {field_name}.",
                    suggested_fix="Re-read the numeric amount and return a positive value with its unit.",
                )
    return None


def check_no_primary_reactant(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Block recipes with no reactant because synthesis needs a starting material."""

    reactants = [entity for entity in recipe.entities if entity.role == ChemicalRole.REACTANT]
    if len(recipe.entities) == 0:
        return None
    if not reactants:
        return ValidationError(
            error_type=ErrorType.NO_PRIMARY_REACTANT,
            severity=ErrorSeverity.BLOCKING,
            responsible_agent=ResponsibleAgent.ENTITY_AGENT,
            field_path="entities",
            entity_name=None,
            actual_value="0 reactants",
            expected="At least one REACTANT",
            message="No REACTANT found. Every synthesis has at least one primary reactant consumed in the reaction.",
            suggested_fix="Identify which chemical is the main starting material being transformed. Assign it REACTANT role.",
        )
    return None


def check_no_entities(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Block empty entity lists because there is no recipe to trust."""

    if len(recipe.entities) == 0:
        return ValidationError(
            error_type=ErrorType.NO_ENTITIES,
            severity=ErrorSeverity.BLOCKING,
            responsible_agent=ResponsibleAgent.ENTITY_AGENT,
            field_path="entities",
            entity_name=None,
            actual_value="[]",
            expected="At least one chemical entity",
            message="No chemical entities were extracted from the recipe.",
            suggested_fix="Re-read the source text and extract every explicitly named chemical substance.",
        )
    return None


def check_solvent_is_reactant(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Block duplicate role conflicts where one name is solvent and reactant."""

    names_as_solvent = {entity.name.lower() for entity in recipe.entities if entity.role == ChemicalRole.SOLVENT}
    names_as_reactant = {entity.name.lower() for entity in recipe.entities if entity.role == ChemicalRole.REACTANT}
    overlap = names_as_solvent & names_as_reactant
    if overlap:
        name = sorted(overlap)[0]
        return ValidationError(
            error_type=ErrorType.SOLVENT_IS_REACTANT,
            severity=ErrorSeverity.BLOCKING,
            responsible_agent=ResponsibleAgent.ENTITY_AGENT,
            field_path="entities.role",
            entity_name=name,
            actual_value="SOLVENT and REACTANT",
            expected="Single primary role",
            message=f"'{name}' is assigned as both SOLVENT and REACTANT.",
            suggested_fix=f"Choose the primary role for '{name}' and merge duplicate entries.",
        )
    return None


def check_product_quantity_as_input(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Warn when a product has an input quantity because roles may be swapped."""

    for index, entity in enumerate(recipe.entities):
        if entity.role == ChemicalRole.PRODUCT and entity.quantity is not None:
            return ValidationError(
                error_type=ErrorType.PRODUCT_QUANTITY_AS_INPUT,
                severity=ErrorSeverity.WARNING,
                responsible_agent=ResponsibleAgent.ENTITY_AGENT,
                field_path=f"entities[{index}].quantity",
                entity_name=entity.name,
                actual_value=str(entity.quantity),
                expected="Product quantity usually absent in input procedure",
                message=f"Product '{entity.name}' has an input quantity. Products are outputs. This may indicate a role misassignment.",
                suggested_fix=f"Verify '{entity.name}' is truly the product, not a reactant.",
            )
    return None


def check_temperature_range(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Block physically impossible laboratory synthesis temperatures."""

    if recipe.conditions is None:
        return None
    temp = recipe.conditions.temperature_celsius
    if temp is not None and not (-200.0 <= temp <= 3000.0):
        return ValidationError(
            error_type=ErrorType.TEMPERATURE_OUT_OF_RANGE,
            severity=ErrorSeverity.BLOCKING,
            responsible_agent=ResponsibleAgent.CONDITION_AGENT,
            field_path="conditions.temperature_celsius",
            entity_name=None,
            actual_value=str(temp),
            expected="-200.0 <= temperature <= 3000.0",
            message=f"Temperature {temp}°C is outside physically possible range for laboratory synthesis (-200 to 3000°C).",
            suggested_fix="Re-read the temperature. If Kelvin, subtract 273.15. If Fahrenheit, apply (F-32)*5/9.",
        )
    return None


def check_duration_range(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Block reaction durations outside plausible synthesis ranges."""

    if recipe.conditions is None:
        return None
    duration = recipe.conditions.duration_hours
    if duration is not None and not (0.001 <= duration <= 720.0):
        return ValidationError(
            error_type=ErrorType.DURATION_OUT_OF_RANGE,
            severity=ErrorSeverity.BLOCKING,
            responsible_agent=ResponsibleAgent.CONDITION_AGENT,
            field_path="conditions.duration_hours",
            entity_name=None,
            actual_value=str(duration),
            expected="0.001 <= duration_hours <= 720.0",
            message=f"Duration {duration} hours is outside plausible synthesis range.",
            suggested_fix="Re-read the duration and convert minutes or days to hours.",
        )
    return None


def check_pressure_range(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Block pressure values outside physically plausible synthesis ranges."""

    if recipe.conditions is None:
        return None
    pressure = recipe.conditions.pressure_atm
    if not (0.0001 <= pressure <= 1000.0):
        return ValidationError(
            error_type=ErrorType.PRESSURE_OUT_OF_RANGE,
            severity=ErrorSeverity.BLOCKING,
            responsible_agent=ResponsibleAgent.CONDITION_AGENT,
            field_path="conditions.pressure_atm",
            entity_name=None,
            actual_value=str(pressure),
            expected="0.0001 <= pressure_atm <= 1000.0",
            message=f"Pressure {pressure} atm is outside plausible synthesis range.",
            suggested_fix="Re-read pressure units and convert to atm.",
        )
    return None


def check_yield_range(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Block yield percentages outside the only meaningful 0 to 100 range."""

    if recipe.conditions is None:
        return None
    yield_percent = recipe.conditions.yield_percent
    if yield_percent is not None and not (0.0 <= yield_percent <= 100.0):
        return ValidationError(
            error_type=ErrorType.YIELD_OUT_OF_RANGE,
            severity=ErrorSeverity.BLOCKING,
            responsible_agent=ResponsibleAgent.CONDITION_AGENT,
            field_path="conditions.yield_percent",
            entity_name=None,
            actual_value=str(yield_percent),
            expected="0.0 <= yield_percent <= 100.0",
            message=f"Yield {yield_percent}% is outside the valid percentage range.",
            suggested_fix="Only extract yield when the source reports a valid percent yield.",
        )
    return None


def check_no_conditions(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Warn when conditions are absent because abstracts often omit them."""

    if recipe.conditions is None:
        return ValidationError(
            error_type=ErrorType.NO_CONDITIONS,
            severity=ErrorSeverity.WARNING,
            responsible_agent=ResponsibleAgent.CONDITION_AGENT,
            field_path="conditions",
            entity_name=None,
            actual_value="None",
            expected="Reaction conditions if present",
            message="No reaction conditions found. Abstracts may omit exact conditions.",
            suggested_fix="If present, extract temperature, duration, pressure, atmosphere, and technique.",
        )
    has_temp = recipe.conditions.temperature_celsius is not None
    has_duration = recipe.conditions.duration_hours is not None
    if not has_temp and not has_duration:
        return ValidationError(
            error_type=ErrorType.NO_CONDITIONS,
            severity=ErrorSeverity.WARNING,
            responsible_agent=ResponsibleAgent.CONDITION_AGENT,
            field_path="conditions",
            entity_name=None,
            actual_value="No temperature or duration",
            expected="At least temperature or duration when available",
            message="Neither temperature nor duration found. At least one condition should be present for a valid recipe.",
            suggested_fix="Re-read the source text for temperature and duration cues.",
        )
    return None


def check_entity_name_too_short(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Block one-character entity names because they are almost always parse noise."""

    for index, entity in enumerate(recipe.entities):
        if len(entity.name.strip()) < 2:
            return ValidationError(
                error_type=ErrorType.ENTITY_NAME_TOO_SHORT,
                severity=ErrorSeverity.BLOCKING,
                responsible_agent=ResponsibleAgent.ENTITY_AGENT,
                field_path=f"entities[{index}].name",
                entity_name=entity.name,
                actual_value=entity.name,
                expected="At least 2 characters",
                message=f"Entity name '{entity.name}' is too short to be trusted.",
                suggested_fix="Replace parse noise with the full chemical name from the source text or remove it.",
            )
    return None


def check_duplicate_entity_names(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Block repeated entity names because duplicates corrupt graph nodes."""

    seen = set()
    for name in [entity.name.lower().strip() for entity in recipe.entities]:
        if name in seen:
            return ValidationError(
                error_type=ErrorType.DUPLICATE_ENTITY_NAMES,
                severity=ErrorSeverity.BLOCKING,
                responsible_agent=ResponsibleAgent.ENTITY_AGENT,
                field_path="entities",
                entity_name=name,
                actual_value="duplicate",
                expected="Unique entity names",
                message=f"'{name}' appears multiple times. Merge duplicates.",
                suggested_fix=f"Merge duplicate '{name}' entities into one record with the primary role.",
            )
        seen.add(name)
    return None


def check_moles_mass_inconsistent(recipe: ChemicalRecipe) -> Optional[ValidationError]:
    """Warn on impossible implied molecular weights without blocking uncertain chemistry."""

    for index, entity in enumerate(recipe.entities):
        if entity.quantity is None or entity.moles is None:
            continue
        mass_unit = entity.quantity.unit
        mole_unit = entity.moles.unit
        if mass_unit not in {"g", "mg", "kg"} or mole_unit not in {"mol", "mmol"}:
            continue
        grams = entity.quantity.value
        if mass_unit == "mg":
            grams /= 1000.0
        elif mass_unit == "kg":
            grams *= 1000.0
        mol = entity.moles.value
        if mole_unit == "mmol":
            mol /= 1000.0
        if mol <= 0:
            continue
        implied_mw = grams / mol
        if implied_mw < 1.0 or implied_mw > 100000.0:
            return ValidationError(
                error_type=ErrorType.MOLES_MASS_INCONSISTENT,
                severity=ErrorSeverity.WARNING,
                responsible_agent=ResponsibleAgent.QUANTITY_AGENT,
                field_path=f"entities[{index}].quantity",
                entity_name=entity.name,
                actual_value=f"{entity.quantity}, {entity.moles}",
                expected="Plausible mass/mole ratio",
                message=f"Mass/mole ratio for '{entity.name}' implies MW of {implied_mw:.1f} g/mol which is unusual. Verify source text.",
                suggested_fix="Check whether mass and molar amount were aligned to the same entity.",
            )
    return None
