"""LangGraph state schema for Phase 2 extraction.

This file contains PipelineState and make_initial_state. LangGraph uses a
TypedDict state so each node can read the full context and return partial
updates while preserving state between entity, quantity, and condition agents.
"""

from __future__ import annotations

import logging
from typing import Optional

from typing_extensions import TypedDict

from src.schemas.chemical import ChemicalEntity
from src.schemas.paper import AcquisitionMethod, TextCompleteness
from src.schemas.recipe import ChemicalRecipe, CorrectionRecord, ExtractionMethod, ReactionConditions
from src.schemas.validation import ValidationError

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    """State object passed through every LangGraph node in Phase 2."""

    source_text: str
    source_chunk_id: str
    source_paper_title: Optional[str]
    source_paper_doi: Optional[str]
    source_paper_url: Optional[str]
    source_text_completeness: Optional[TextCompleteness]
    source_acquisition_method: Optional[AcquisitionMethod]
    identified_entities: list[ChemicalEntity]
    quantified_entities: list[ChemicalEntity]
    conditions: Optional[ReactionConditions]
    recipe: Optional[ChemicalRecipe]
    retry_count: int
    max_retries: int
    last_validation_errors: list[ValidationError]
    validation_warnings: list[ValidationError]
    correction_history: list[CorrectionRecord]
    route_to: Optional[str]
    total_tokens: int
    total_latency: float
    node_latencies: dict[str, float]
    node_tokens: dict[str, int]
    node_extraction_methods: dict[str, ExtractionMethod]
    llm_provider: str
    llm_model: str


def make_initial_state(
    source_text: str,
    chunk_id: str,
    paper_title: Optional[str] = None,
    paper_doi: Optional[str] = None,
    max_retries: int = 3,
    paper_url: Optional[str] = None,
    text_completeness: Optional[TextCompleteness] = None,
    acquisition_method: Optional[AcquisitionMethod] = None,
) -> PipelineState:
    """Create the initial graph state for one TextChunk extraction."""

    return PipelineState(
        source_text=source_text,
        source_chunk_id=chunk_id,
        source_paper_title=paper_title,
        source_paper_doi=paper_doi,
        source_paper_url=paper_url,
        source_text_completeness=text_completeness,
        source_acquisition_method=acquisition_method,
        identified_entities=[],
        quantified_entities=[],
        conditions=None,
        recipe=None,
        retry_count=0,
        max_retries=max_retries,
        last_validation_errors=[],
        validation_warnings=[],
        correction_history=[],
        route_to=None,
        total_tokens=0,
        total_latency=0.0,
        node_latencies={},
        node_tokens={},
        node_extraction_methods={},
        llm_provider="",
        llm_model="",
    )
