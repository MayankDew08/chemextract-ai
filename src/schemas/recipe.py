"""Recipe schemas for Phase 2 structured extraction output.

This file contains ValidationStatus, ReactionConditions, CorrectionRecord, and
ChemicalRecipe. The recipe model is the durable artifact emitted by the graph
and carries observability fields alongside the extracted chemistry.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator

from src.schemas.chemical import ChemicalEntity, ChemicalRole
from src.schemas.paper import AcquisitionMethod, TextCompleteness
from src.schemas.validation import ValidationError

logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    """ValidationStatus reserves Phase 3 correction states without using them yet."""

    PENDING = "PENDING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    CORRECTED = "CORRECTED"


class ExtractionMethod(str, Enum):
    """Identify how an extraction node obtained its accepted output."""

    PRIMARY_LLM = "primary_llm"
    FALLBACK_LLM = "fallback_llm"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"


class ReactionConditions(BaseModel):
    """Physical reaction conditions normalized to Celsius, hours, and atm."""

    temperature_celsius: Optional[float] = Field(default=None, allow_inf_nan=False)
    duration_hours: Optional[float] = Field(default=None, allow_inf_nan=False)
    pressure_atm: Optional[float] = Field(default=None, allow_inf_nan=False)
    atmosphere: Optional[str] = None
    technique: Optional[str] = None
    yield_percent: Optional[float] = Field(default=None, allow_inf_nan=False)
    additional_conditions: Optional[dict[str, str]] = Field(default_factory=dict)

    @field_validator("atmosphere")
    @classmethod
    def normalize_atmosphere(cls, v: Optional[str]) -> Optional[str]:
        """Normalize shorthand gas labels so recipes are queryable."""

        if v is None:
            return None
        value = v.lower().strip()
        mapping = {
            "n2": "nitrogen",
            "ar": "argon",
            "o2": "oxygen",
            "h2": "hydrogen",
            "co2": "carbon dioxide",
            "inert": "argon",
            "ambient": "air",
        }
        return mapping.get(value, value)

    @field_validator("additional_conditions", mode="before")
    @classmethod
    def normalize_additional_conditions(cls, v: object) -> dict[str, str]:
        """Treat provider-emitted null as an empty mapping instead of a hard failure."""

        if v is None:
            return {}
        return v


class CorrectionRecord(BaseModel):
    """Records one future self-correction cycle for auditability."""

    attempt_number: int = Field(..., ge=1)
    error_type: str
    error_field: str
    error_message: str
    agent_routed_to: str = Field(
        validation_alias=AliasChoices("agent_routed_to", "agent_that_fixed"),
        description="Extraction agent selected to address the reported validation error",
    )
    corrected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def agent_that_fixed(self) -> str:
        """Return the legacy audit name for callers during the transition window."""

        return self.agent_routed_to


class ChemicalRecipe(BaseModel):
    """Complete structured output for one extracted synthesis recipe."""

    recipe_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_chunk_id: str
    source_paper_title: Optional[str] = None
    source_paper_doi: Optional[str] = None
    source_paper_url: Optional[str] = None
    source_text_completeness: Optional[TextCompleteness] = None
    source_acquisition_method: Optional[AcquisitionMethod] = None
    title: Optional[str] = None
    entities: list[ChemicalEntity] = Field(default_factory=list)
    conditions: Optional[ReactionConditions] = None
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[ValidationError] = Field(default_factory=list)
    correction_history: list[CorrectionRecord] = Field(default_factory=list)
    total_tokens_used: int = Field(default=0)
    estimated_cost_usd: float = Field(default=0.0)
    total_latency_seconds: float = Field(default=0.0)
    node_latencies: dict[str, float] = Field(default_factory=dict)
    node_tokens: dict[str, int] = Field(default_factory=dict)
    node_extraction_methods: dict[str, ExtractionMethod] = Field(default_factory=dict)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    llm_provider: str = Field(default="unknown")
    llm_model: str = Field(default="unknown")

    def get_reactants(self) -> list[ChemicalEntity]:
        """Return reactants so callers do not duplicate role filtering logic."""

        return [entity for entity in self.entities if entity.role == ChemicalRole.REACTANT]

    def get_solvents(self) -> list[ChemicalEntity]:
        """Return solvents so UI and validation code share the same role rule."""

        return [entity for entity in self.entities if entity.role == ChemicalRole.SOLVENT]

    def get_catalysts(self) -> list[ChemicalEntity]:
        """Return catalysts for recipe summaries and Phase 3 checks."""

        return [entity for entity in self.entities if entity.role == ChemicalRole.CATALYST]

    def get_products(self) -> list[ChemicalEntity]:
        """Return products for downstream graph construction."""

        return [entity for entity in self.entities if entity.role == ChemicalRole.PRODUCT]

    def correction_count(self) -> int:
        """Expose correction count without leaking history internals."""

        return len(self.correction_history)

    def was_corrected(self) -> bool:
        """Return whether Phase 3 or later correction loops modified this recipe."""

        return len(self.correction_history) > 0
