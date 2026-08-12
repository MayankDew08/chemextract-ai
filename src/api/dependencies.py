"""FastAPI dependency providers for shared application state.

This file centralizes access to graph storage, metrics storage, and templates so
route modules do not import concrete backends or reach into app.state directly.
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.templating import Jinja2Templates

from src.observability.metrics_store import MetricsStore
from src.storage.base import BaseGraphStore

logger = logging.getLogger(__name__)


async def get_store(request: Request) -> BaseGraphStore:
    """Return the initialized graph store from FastAPI lifespan state."""

    return request.app.state.graph_store


async def get_metrics_store(request: Request) -> MetricsStore:
    """Return the initialized metrics store from FastAPI lifespan state."""

    return request.app.state.metrics_store


async def get_templates(request: Request) -> Jinja2Templates:
    """Return the shared Jinja2 template renderer."""

    return request.app.state.templates
