"""Deterministic validation engine for Phase 3.

This file contains ValidationEngine. It runs every pure-Python validation rule
in a fixed order, catches rule exceptions, and returns structured results
without making any LLM calls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from src.schemas.recipe import ChemicalRecipe
from src.validator.error_types import ErrorSeverity, ErrorType, ResponsibleAgent, ValidationError, ValidationResult
from src.validator.rules import (
    check_duplicate_entity_names,
    check_duration_range,
    check_entity_name_too_short,
    check_moles_mass_inconsistent,
    check_no_conditions,
    check_no_entities,
    check_no_primary_reactant,
    check_pressure_range,
    check_product_quantity_as_input,
    check_quantity_negative,
    check_solvent_is_reactant,
    check_temperature_range,
    check_unit_invalid,
    check_unit_missing,
    check_yield_range,
)

logger = logging.getLogger(__name__)


class ValidationEngine:
    """ValidationEngine runs all deterministic rules against a recipe."""

    def __init__(self) -> None:
        """Register rules in dependency-aware order so routing is stable."""

        self._rules: list[Callable[[ChemicalRecipe], list[ValidationError]]] = [
            check_no_entities,
            check_no_primary_reactant,
            check_duplicate_entity_names,
            check_entity_name_too_short,
            check_unit_missing,
            check_unit_invalid,
            check_quantity_negative,
            check_solvent_is_reactant,
            check_product_quantity_as_input,
            check_temperature_range,
            check_duration_range,
            check_pressure_range,
            check_yield_range,
            check_no_conditions,
            check_moles_mass_inconsistent,
        ]

    @property
    def rule_count(self) -> int:
        """Expose rule count so demos can verify all 15 rules are registered."""

        return len(self._rules)

    def validate(self, recipe: ChemicalRecipe) -> ValidationResult:
        """Run all rules and convert rule exceptions into non-blocking warnings."""

        blocking_errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        rules_passed = 0
        for rule_fn in self._rules:
            try:
                findings = rule_fn(recipe)
                if not findings:
                    rules_passed += 1
                else:
                    rule_has_blocking_error = False
                    for finding in findings:
                        if finding.severity == ErrorSeverity.BLOCKING:
                            blocking_errors.append(finding)
                            rule_has_blocking_error = True
                        else:
                            warnings.append(finding)
                    if not rule_has_blocking_error:
                        rules_passed += 1
            except Exception as exc:
                logger.error("Rule %s raised exception: %s", rule_fn.__name__, exc)
                warnings.append(
                    ValidationError(
                        error_type=ErrorType.UNIT_MISSING,
                        severity=ErrorSeverity.WARNING,
                        responsible_agent=ResponsibleAgent.ENTITY_AGENT,
                        field_path="unknown",
                        entity_name=None,
                        actual_value=None,
                        expected="Rule should complete",
                        message=f"Rule {rule_fn.__name__} failed with exception: {exc}",
                        suggested_fix="Check rule implementation",
                    )
                )
        return ValidationResult(
            passed=len(blocking_errors) == 0,
            blocking_errors=blocking_errors,
            warnings=warnings,
            rules_checked=len(self._rules),
            rules_passed=rules_passed,
        )
