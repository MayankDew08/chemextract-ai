"""Query planning schemas for ChemExtract ingestion.

This file defines SearchPlan, SourceSearchQuery, and ChemistryDomain so query
parsers can return one immutable contract that every fetcher understands.
The schema-first design keeps the LLM output, fallback parser, and fetchers
loosely coupled while preserving strict validation at the ingestion boundary.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class ChemistryDomain(str, Enum):
    """ChemistryDomain constrains domain labels so ranking can evolve safely."""

    NANOMATERIALS = "nanomaterials"
    BATTERY_MATERIALS = "battery_materials"
    PHARMACEUTICALS = "pharmaceuticals"
    POLYMERS = "polymers"
    CATALYSIS = "catalysis"
    ORGANIC_SYNTHESIS = "organic_synthesis"
    INORGANIC_SYNTHESIS = "inorganic_synthesis"
    GENERAL = "general"


class SourceSearchQuery(BaseModel):
    """SourceSearchQuery carries source-specific syntax without leaking it upstream."""

    source_name: str
    query_string: str
    max_results: int = Field(default=25, ge=1, le=100)
    filters: dict[str, str] = Field(default_factory=dict)


class SearchPlan(BaseModel):
    """SearchPlan is immutable so downstream ingestion cannot mutate parser intent."""

    raw_query: str
    target_compound: str
    formula: Optional[str] = None
    synonyms: list[str] = Field(default_factory=list)
    source_queries: dict[str, SourceSearchQuery]
    domain: ChemistryDomain = ChemistryDomain.GENERAL
    expected_reactants: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)

    @field_validator("synonyms")
    @classmethod
    def synonyms_not_empty(cls, value: list[str]) -> list[str]:
        """Keep empty synonym lists valid while removing accidental blank entries."""

        return [item.strip() for item in value if item and item.strip()]
