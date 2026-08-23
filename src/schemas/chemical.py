"""Chemical participant schemas for Phase 2 extraction.

This file contains ChemicalRole, Quantity, ChemicalEntity, and EntityList. The
schemas preserve parseable measurements so deterministic validation can report
unsupported or implausible chemistry without discarding the extraction output.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ChemicalRole(str, Enum):
    """ChemicalRole constrains agent labels to reaction-relevant categories."""

    REACTANT = "REACTANT"
    SOLVENT = "SOLVENT"
    CATALYST = "CATALYST"
    ADDITIVE = "ADDITIVE"
    PRODUCT = "PRODUCT"
    UNKNOWN = "UNKNOWN"


class Quantity(BaseModel):
    """A parseable physical quantity whose chemistry semantics are validated later."""

    value: float = Field(..., allow_inf_nan=False, description="Numeric value as extracted from the source")
    unit: str = Field(..., min_length=1, description="Physical unit e.g. g, mg, mL, mmol, M, wt%, vol%")

    @field_validator("unit")
    @classmethod
    def unit_must_be_present(cls, v: str) -> str:
        """Normalize surrounding whitespace while retaining unknown units for validation."""

        unit = v.strip()
        if not unit:
            raise ValueError("Quantity unit cannot be blank")
        return unit

    def __str__(self) -> str:
        """Render quantities in a chemist-readable compact form."""

        return f"{self.value} {self.unit}"


class ChemicalEntity(BaseModel):
    """One chemical participant in a reaction with optional measured amounts."""

    name: str = Field(..., min_length=1, description="Chemical name as it appears in the text")
    formula: Optional[str] = Field(None, description="Chemical formula if determinable e.g. ZnO, H2O")
    role: ChemicalRole = Field(..., description="Role this chemical plays in the reaction")
    quantity: Optional[Quantity] = Field(None, description="Amount used. None if not specified in text.")
    moles: Optional[Quantity] = Field(None, description="Molar amount if separately specified e.g. 10 mmol")
    notes: Optional[str] = Field(None, description="Additional context e.g. anhydrous or commercial grade")

    @field_validator("name")
    @classmethod
    def name_not_unknown(cls, v: str) -> str:
        """Force agents to return a real extracted name instead of placeholders."""

        if v.lower().strip() in {"unknown", "n/a", "na", "none", ""}:
            raise ValueError(f"Chemical name cannot be '{v}'. Extract the actual name from the text.")
        return v.strip()

    @field_validator("formula")
    @classmethod
    def formula_basic_check(cls, v: Optional[str]) -> Optional[str]:
        """Drop malformed formulas without failing the full entity extraction."""

        if v is None:
            return None
        if not re.match(r"^[A-Za-z0-9\(\)\[\]\+\-\.·]+$", v):
            return None
        return v.strip()


class EntityList(BaseModel):
    """Structured wrapper used for reliable entity-agent output parsing."""

    entities: list[ChemicalEntity] = Field(
        default_factory=list,
        description="All chemical entities found in the text; empty when none are explicitly present",
    )
    extraction_notes: Optional[str] = Field(None, description="Ambiguities or assumptions made during extraction")
