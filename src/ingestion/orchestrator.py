"""Async orchestration for Phase 1 literature ingestion.

This file contains FetchOrchestrator. It coordinates query parsing, health
checks, parallel fetching, deduplication, ranking, and text acquisition while
depending only on abstract interfaces and the registry object.
"""

from __future__ import annotations

import asyncio
import logging
import time

from src.ingestion.base import BaseDeduplicator, BaseQueryParser, BaseRanker
from src.ingestion.registry import FetcherRegistry
from src.schemas.paper import TextChunk

logger = logging.getLogger(__name__)


class FetchOrchestrator:
    """FetchOrchestrator wires Phase 1 components without importing concrete sources."""

    def __init__(
        self,
        registry: FetcherRegistry,
        deduplicator: BaseDeduplicator,
        ranker: BaseRanker,
        query_parser: BaseQueryParser,
        top_k_papers: int = 15,
        max_chunks_per_paper: int = 3,
    ) -> None:
        """Store abstract collaborators so the ingestion graph remains swappable."""

        self._registry = registry
        self._deduplicator = deduplicator
        self._ranker = ranker
        self._query_parser = query_parser
        self._top_k_papers = top_k_papers
        self._max_chunks_per_paper = max_chunks_per_paper

    async def run(self, raw_query: str, skip_health_check: bool = False) -> list[TextChunk]:
        """Run the full ingestion pipeline and return unified text chunks."""

        started = time.perf_counter()
        source_names = self._registry.get_registered_source_names()
        try:
            search_plan = await self._query_parser.parse(raw_query, source_names)
        except ValueError as exc:
            logger.error("Query parsing failed: %s", exc)
            return []

        fetchers = self._registry.get_all_fetchers()
        if not skip_health_check:
            health_results = await asyncio.gather(
                *(self._safe_health_check(fetcher) for fetcher in fetchers),
                return_exceptions=False,
            )
            fetchers = [fetcher for fetcher, healthy in zip(fetchers, health_results) if healthy]
            if not fetchers:
                logger.error("No healthy fetchers available")
                return []

        fetch_results = await asyncio.gather(
            *(fetcher.fetch(raw_query, search_plan) for fetcher in fetchers),
            return_exceptions=True,
        )
        all_papers = []
        for result in fetch_results:
            if isinstance(result, Exception):
                logger.warning("Fetcher task raised unexpectedly: %s", result)
                continue
            logger.info(
                "%s fetched %s papers in %.2fs",
                result.source_name,
                len(result.papers),
                result.fetch_duration_seconds,
            )
            if result.success:
                all_papers.extend(result.papers)
        if not all_papers:
            logger.warning("No papers returned for query: %s", raw_query)
            return []

        unique = self._deduplicator.deduplicate(all_papers)
        top_papers = self._ranker.rank(unique, search_plan, self._top_k_papers)

        all_chunks: list[TextChunk] = []
        acquirers = self._registry.get_all_acquirers()
        for paper in top_papers:
            paper_chunks: list[TextChunk] = []
            for acquirer in acquirers:
                try:
                    if not await acquirer.can_acquire(paper):
                        continue
                    chunks = await acquirer.acquire(paper)
                    if chunks:
                        paper_chunks = chunks[: self._max_chunks_per_paper]
                        break
                except Exception as exc:
                    logger.warning("%s failed for %s: %s", acquirer.acquirer_name, paper.title, exc)
            all_chunks.extend(paper_chunks)
            completeness = paper_chunks[0].completeness.value if paper_chunks else "none"
            logger.info("Acquired %s chunks from '%s' (%s)", len(paper_chunks), paper.title, completeness)

        logger.info("Phase 1 ingestion returned %s chunks in %.2fs", len(all_chunks), time.perf_counter() - started)
        return all_chunks

    async def _safe_health_check(self, fetcher: object) -> bool:
        """Run one health check defensively so a broken source cannot abort checks."""

        try:
            return bool(await fetcher.check_health())
        except Exception as exc:
            logger.warning("Health check raised for %s: %s", getattr(fetcher, "source_name", "unknown"), exc)
            return False
