"""Shared schemas for deterministic recipe validation.

Validation data is part of the durable recipe contract, so its models live in
the schema layer rather than the validator implementation package.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    """Every deterministic validation failure ChemExtract currently reports."""

    UNIT_MISSING = "UNIT_MISSING"
    UNIT_INVALID = "UNIT_INVALID"
    QUANTITY_NEGATIVE = "QUANTITY_NEGATIVE"
    NO_PRIMARY_REACTANT = "NO_PRIMARY_REACTANT"
    NO_ENTITIES = "NO_ENTITIES"
    SOLVENT_IS_REACTANT = "SOLVENT_IS_REACTANT"
    TEMPERATURE_OUT_OF_RANGE = "TEMPERATURE_OUT_OF_RANGE"
    DURATION_OUT_OF_RANGE = "DURATION_OUT_OF_RANGE"
    PRESSURE_OUT_OF_RANGE = "PRESSURE_OUT_OF_RANGE"
    YIELD_OUT_OF_RANGE = "YIELD_OUT_OF_RANGE"
    NO_CONDITIONS = "NO_CONDITIONS"
    ENTITY_NAME_TOO_SHORT = "ENTITY_NAME_TOO_SHORT"
    MOLES_MASS_INCONSISTENT = "MOLES_MASS_INCONSISTENT"
    DUPLICATE_ENTITY_NAMES = "DUPLICATE_ENTITY_NAMES"
    PRODUCT_QUANTITY_AS_INPUT = "PRODUCT_QUANTITY_AS_INPUT"


class ErrorSeverity(str, Enum):
    """Separate failures that block acceptance from retained warnings."""

    BLOCKING = "BLOCKING"
    WARNING = "WARNING"


class ResponsibleAgent(str, Enum):
    """Map validation findings to the extraction node that can address them."""

    ENTITY_AGENT = "entity_agent"
    QUANTITY_AGENT = "quantity_agent"
    CONDITION_AGENT = "condition_agent"


class ValidationError(BaseModel):
    """A structured validation finding suitable for persistence and routing."""

    error_type: ErrorType
    severity: ErrorSeverity
    responsible_agent: ResponsibleAgent
    field_path: str
    entity_name: Optional[str] = None
    actual_value: Optional[str] = None
    expected: Optional[str] = None
    message: str
    suggested_fix: str


class ValidationResult(BaseModel):
    """The deterministic result of applying every registered validation rule."""

    passed: bool
    blocking_errors: list[ValidationError] = Field(default_factory=list)
    warnings: list[ValidationError] = Field(default_factory=list)
    rules_checked: int = 0
    rules_passed: int = 0
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def has_blocking_errors(self) -> bool:
        """Return whether the recipe must be corrected before acceptance."""

        return bool(self.blocking_errors)

    def errors_for_agent(self, agent: ResponsibleAgent) -> list[ValidationError]:
        """Return all blocking errors assigned to one correction agent."""

        return [error for error in self.blocking_errors if error.responsible_agent == agent]

    def primary_responsible_agent(self) -> Optional[ResponsibleAgent]:
        """Route cascading failures by fixing entities, quantities, then conditions."""

        priority = (
            ResponsibleAgent.ENTITY_AGENT,
            ResponsibleAgent.QUANTITY_AGENT,
            ResponsibleAgent.CONDITION_AGENT,
        )
        return next(
            (agent for agent in priority if any(error.responsible_agent == agent for error in self.blocking_errors)),
            None,
        )
