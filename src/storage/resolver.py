"""Graph store resolver for Phase 4 storage backends.

This file contains get_graph_store. It centralizes backend selection so callers
depend only on BaseGraphStore and switch storage by changing GRAPH_BACKEND.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.storage.base import BaseGraphStore

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_pipeline_store: Optional[BaseGraphStore] = None
_pipeline_metrics_store = None


def get_graph_store(backend: Optional[str] = None) -> BaseGraphStore:
    """Return an initialized graph store based on argument or environment."""

    resolved_backend = (backend or os.getenv("GRAPH_BACKEND", "networkx")).lower().strip()
    if resolved_backend == "networkx":
        from src.storage.networkx_store import NetworkXStore

        store = NetworkXStore()
    elif resolved_backend == "obsidian":
        from src.storage.obsidian_store import ObsidianVaultStore

        store = ObsidianVaultStore(vault_path=os.getenv("OBSIDIAN_VAULT_PATH", "./data/obsidian_vault"))
    elif resolved_backend == "neo4j":
        from src.storage.neo4j_store import Neo4jStore

        store = Neo4jStore()
    else:
        raise ValueError(
            f"Unknown graph backend: '{resolved_backend}'. Supported: networkx, obsidian, neo4j. "
            "Set GRAPH_BACKEND env var to one of these."
        )
    store.initialize()
    logger.info("Graph store initialized: %s", resolved_backend)
    return store


def set_pipeline_store(store: BaseGraphStore) -> None:
    """Register the active graph store so the pipeline and API share one instance."""

    global _pipeline_store
    _pipeline_store = store


def get_pipeline_store(backend: Optional[str] = None) -> BaseGraphStore:
    """Return the active pipeline graph store, creating one in standalone mode."""

    global _pipeline_store
    if _pipeline_store is not None:
        return _pipeline_store
    _pipeline_store = get_graph_store(backend)
    return _pipeline_store


def set_pipeline_metrics_store(store) -> None:
    """Register the active metrics store for pipeline observability writes."""

    global _pipeline_metrics_store
    _pipeline_metrics_store = store


def get_pipeline_metrics_store():
    """Return the active pipeline metrics store, if FastAPI registered one."""

    return _pipeline_metrics_store
