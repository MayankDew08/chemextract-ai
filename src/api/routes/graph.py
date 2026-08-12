"""Knowledge graph API routes for Phase 5."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Response

from src.api.dependencies import get_store
from src.storage.base import BaseGraphStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/graph/data")
async def graph_data(response: Response, store: Annotated[BaseGraphStore, Depends(get_store)]):
    """Return vis.js-compatible graph JSON."""

    response.headers["Cache-Control"] = "max-age=30"
    return store.export_graph_json().to_visjs_dict()


@router.get("/graph/query/solvents/{compound}")
async def query_solvents(compound: str, store: Annotated[BaseGraphStore, Depends(get_store)]):
    """Return solvents used for reactions producing a compound."""

    return store.query_solvents_for(compound)


@router.get("/graph/query/cooccurrence/{chemical}")
async def query_cooccurrence(chemical: str, store: Annotated[BaseGraphStore, Depends(get_store)]):
    """Return chemicals co-occurring with a query chemical."""

    return store.query_cooccurring_chemicals(chemical, limit=15)


@router.get("/graph/stats")
async def graph_stats(store: Annotated[BaseGraphStore, Depends(get_store)]):
    """Return graph storage stats for the graph sidebar."""

    return {
        "node_count": store.node_count(),
        "edge_count": store.edge_count(),
        "recipe_count": store.recipe_count(),
        "backend": store.backend_name,
    }
