"""Pydantic metrics schemas for the Phase 5 dashboard.

This file contains the public observability models returned by the metrics API.
Keeping metrics structured makes dashboard rendering and automated demos use the
same contract.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ErrorFrequency(BaseModel):
    """ErrorFrequency summarizes validation correction errors."""

    error_type: str
    count: int
    percentage: float


class NodeTiming(BaseModel):
    """NodeTiming summarizes per-agent latency and token usage."""

    node_name: str
    avg_latency_seconds: float
    total_calls: int
    total_tokens: int


class PipelineMetrics(BaseModel):
    """PipelineMetrics is the full dashboard snapshot computed from recipe runs."""

    total_recipes: int
    total_chunks_processed: int
    passed_first_try: int
    passed_after_correction: int
    failed_unresolvable: int
    success_rate_percent: float
    self_correction_rate_percent: float
    total_tokens_used: int
    total_cost_usd: float
    avg_latency_seconds: float
    avg_tokens_per_recipe: float
    error_frequencies: list[ErrorFrequency]
    most_common_error: Optional[str]
    total_corrections_made: int
    avg_corrections_per_recipe: float
    node_timings: list[NodeTiming]
    graph_node_count: int
    graph_edge_count: int
    unique_chemicals: int
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
