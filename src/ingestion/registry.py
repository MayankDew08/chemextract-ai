"""Registry for ingestion fetchers and text acquirers.

This file contains FetcherRegistry, the global registry singleton, and
bootstrap_registry. Concrete fetchers are imported only inside bootstrap_registry
so the orchestrator and tests can depend on abstract interfaces exclusively.
"""

from __future__ import annotations

import logging

from src.ingestion.base import BaseFetcher, BaseTextAcquirer

logger = logging.getLogger(__name__)


class FetcherRegistry:
    """FetcherRegistry stores concrete plugins behind abstract base types."""

    def __init__(self) -> None:
        """Initialize empty maps so bootstrapping can be idempotent."""

        self._fetchers: dict[str, BaseFetcher] = {}
        self._acquirers: dict[str, BaseTextAcquirer] = {}

    def register_fetcher(self, fetcher: BaseFetcher) -> None:
        """Register one fetcher after validating it satisfies BaseFetcher."""

        if not isinstance(fetcher, BaseFetcher):
            raise TypeError("fetcher must implement BaseFetcher")
        self._fetchers[fetcher.source_name] = fetcher
        logger.info("Registered fetcher: %s", fetcher.source_name)

    def register_acquirer(self, acquirer: BaseTextAcquirer) -> None:
        """Register one acquirer after validating it satisfies BaseTextAcquirer."""

        if not isinstance(acquirer, BaseTextAcquirer):
            raise TypeError("acquirer must implement BaseTextAcquirer")
        self._acquirers[acquirer.acquirer_name] = acquirer
        logger.info("Registered acquirer: %s", acquirer.acquirer_name)

    def get_all_fetchers(self) -> list[BaseFetcher]:
        """Return all registered fetchers without exposing internal maps."""

        return list(self._fetchers.values())

    def get_all_acquirers(self) -> list[BaseTextAcquirer]:
        """Return acquirers sorted by priority for deterministic acquisition."""

        return sorted(self._acquirers.values(), key=lambda acquirer: acquirer.priority())

    def get_registered_source_names(self) -> list[str]:
        """Return registered source names for source-aware query parsing."""

        return list(self._fetchers.keys())


registry = FetcherRegistry()


def bootstrap_registry() -> FetcherRegistry:
    """Import and register all concrete Phase 1 classes in exactly one place."""

    from src.ingestion.fetchers.arxiv import ArXivFetcher, ArXivHTMLAcquirer
    from src.ingestion.fetchers.openalex import OpenAlexAbstractAcquirer, OpenAlexFetcher
    from src.ingestion.fetchers.pubmed import PubMedCentralAcquirer, PubMedFetcher
    from src.ingestion.fetchers.semantic_scholar import (
        SemanticScholarAbstractAcquirer,
        SemanticScholarFetcher,
    )

    registry.register_fetcher(PubMedFetcher())
    registry.register_fetcher(ArXivFetcher())
    registry.register_fetcher(OpenAlexFetcher())
    registry.register_fetcher(SemanticScholarFetcher())
    registry.register_acquirer(PubMedCentralAcquirer())
    registry.register_acquirer(ArXivHTMLAcquirer())
    registry.register_acquirer(OpenAlexAbstractAcquirer())
    registry.register_acquirer(SemanticScholarAbstractAcquirer())
    return registry
