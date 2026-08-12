"""Metrics API routes for Phase 5 observability."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_metrics_store, get_store
from src.observability.metrics_store import MetricsStore
from src.storage.base import BaseGraphStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics")
async def read_metrics(
    graph_store: Annotated[BaseGraphStore, Depends(get_store)],
    metrics_store: Annotated[MetricsStore, Depends(get_metrics_store)],
):
    """Return the full pipeline metrics snapshot."""

    return await metrics_store.compute_metrics(graph_store)


@router.get("/metrics/errors")
async def read_error_metrics(
    graph_store: Annotated[BaseGraphStore, Depends(get_store)],
    metrics_store: Annotated[MetricsStore, Depends(get_metrics_store)],
):
    """Return validation error frequency metrics."""

    metrics = await metrics_store.compute_metrics(graph_store)
    return metrics.error_frequencies


@router.get("/metrics/nodes")
async def read_node_metrics(
    graph_store: Annotated[BaseGraphStore, Depends(get_store)],
    metrics_store: Annotated[MetricsStore, Depends(get_metrics_store)],
):
    """Return per-node latency and token metrics."""

    metrics = await metrics_store.compute_metrics(graph_store)
    return metrics.node_timings
