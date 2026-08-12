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
from src.schemas.recipe import ChemicalRecipe, CorrectionRecord, ReactionConditions

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    """State object passed through every LangGraph node in Phase 2."""

    source_text: str
    source_chunk_id: str
    source_paper_title: Optional[str]
    source_paper_doi: Optional[str]
    source_paper_url: Optional[str]
    identified_entities: list[ChemicalEntity]
    quantified_entities: list[ChemicalEntity]
    conditions: Optional[ReactionConditions]
    recipe: Optional[ChemicalRecipe]
    retry_count: int
    max_retries: int
    last_validation_errors: list[str]
    correction_history: list[CorrectionRecord]
    route_to: Optional[str]
    total_tokens: int
    total_latency: float
    node_latencies: dict[str, float]
    node_tokens: dict[str, int]
    llm_provider: str
    llm_model: str


def make_initial_state(
    source_text: str,
    chunk_id: str,
    paper_title: Optional[str] = None,
    paper_doi: Optional[str] = None,
    max_retries: int = 3,
    paper_url: Optional[str] = None,
) -> PipelineState:
    """Create the initial graph state for one TextChunk extraction."""

    return PipelineState(
        source_text=source_text,
        source_chunk_id=chunk_id,
        source_paper_title=paper_title,
        source_paper_doi=paper_doi,
        source_paper_url=paper_url,
        identified_entities=[],
        quantified_entities=[],
        conditions=None,
        recipe=None,
        retry_count=0,
        max_retries=max_retries,
        last_validation_errors=[],
        correction_history=[],
        route_to=None,
        total_tokens=0,
        total_latency=0.0,
        node_latencies={},
        node_tokens={},
        llm_provider="",
        llm_model="",
    )
