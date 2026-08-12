"""Validation result schemas for Phase 3 deterministic checks.

This file contains ErrorType, ErrorSeverity, ResponsibleAgent, ValidationError,
and ValidationResult. These schemas keep validation failures structured so the
LangGraph router can send each failure to the responsible correction agent.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """ErrorType enumerates every deterministic validation failure Phase 3 knows."""

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
    """ErrorSeverity separates blocking quality gates from recorded warnings."""

    BLOCKING = "BLOCKING"
    WARNING = "WARNING"


class ResponsibleAgent(str, Enum):
    """ResponsibleAgent maps validation failures back to graph correction nodes."""

    ENTITY_AGENT = "entity_agent"
    QUANTITY_AGENT = "quantity_agent"
    CONDITION_AGENT = "condition_agent"


class ValidationError(BaseModel):
    """ValidationError carries enough context to build targeted retry prompts."""

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
    """ValidationResult is the deterministic output of running all rules."""

    passed: bool
    blocking_errors: list[ValidationError] = Field(default_factory=list)
    warnings: list[ValidationError] = Field(default_factory=list)
    rules_checked: int = 0
    rules_passed: int = 0
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def has_blocking_errors(self) -> bool:
        """Return whether the recipe must be corrected before acceptance."""

        return len(self.blocking_errors) > 0

    def errors_for_agent(self, agent: ResponsibleAgent) -> list[ValidationError]:
        """Return blocking errors assigned to one correction agent."""

        return [error for error in self.blocking_errors if error.responsible_agent == agent]

    def primary_responsible_agent(self) -> Optional[ResponsibleAgent]:
        """Route cascading failures by fixing entities, then quantities, then conditions."""

        if any(error.responsible_agent == ResponsibleAgent.ENTITY_AGENT for error in self.blocking_errors):
            return ResponsibleAgent.ENTITY_AGENT
        if any(error.responsible_agent == ResponsibleAgent.QUANTITY_AGENT for error in self.blocking_errors):
            return ResponsibleAgent.QUANTITY_AGENT
        if any(error.responsible_agent == ResponsibleAgent.CONDITION_AGENT for error in self.blocking_errors):
            return ResponsibleAgent.CONDITION_AGENT
        return None
