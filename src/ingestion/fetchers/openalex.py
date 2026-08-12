"""OpenAlex fetcher and abstract acquirer.

This file contains OpenAlexFetcher and OpenAlexAbstractAcquirer. OpenAlex offers
broad chemistry metadata and inverted-index abstracts, so this module focuses on
faithfully reconstructing abstracts and leaving full-text retrieval to sources
with structured text APIs.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import httpx

from src.ingestion.base import BaseFetcher, BaseTextAcquirer
from src.ingestion.utils import calculate_chemical_density, detect_experimental_section, generate_chunk_id
from src.schemas.paper import AcquisitionMethod, FetchResult, PaperMetadata, TextChunk, TextCompleteness
from src.schemas.query import SearchPlan

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"


class OpenAlexFetcher(BaseFetcher):
    """OpenAlexFetcher searches Works and reconstructs indexed abstracts."""

    def __init__(self) -> None:
        """Create a fetcher with mailto metadata and lazy HTTP reuse."""

        self._client: Optional[httpx.AsyncClient] = None
        self._email = os.getenv("API_EMAIL", "your.email@example.com")
        self._headers = {"User-Agent": f"ChemExtractAI/1.0 (mailto:{self._email})"}

    @property
    def source_name(self) -> str:
        """Return OpenAlex's registry key."""

        return "openalex"

    async def _get_client(self) -> httpx.AsyncClient:
        """Create or reuse one AsyncClient with the OpenAlex User-Agent."""

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0, headers=self._headers)
        return self._client

    async def fetch(self, query: str, search_plan: SearchPlan) -> FetchResult:
        """Fetch OpenAlex article works and normalize them into PaperMetadata."""

        started = time.perf_counter()
        api_calls = 0
        try:
            source_query = search_plan.source_queries.get(self.source_name)
            query_string = source_query.query_string if source_query else query
            max_results = source_query.max_results if source_query else 25
            client = await self._get_client()
            response = await client.get(
                f"{BASE_URL}/works",
                params={
                    "search": query_string,
                    "per_page": min(max_results, 25),
                    "sort": "relevance_score:desc",
                    "mailto": self._email,
                    "filter": "type:article",
                },
            )
            api_calls += 1
            response.raise_for_status()
            payload = response.json()
            papers = [self._parse_work(work) for work in payload.get("results", []) if isinstance(work, dict)]
            papers = [paper for paper in papers if paper is not None]
            return FetchResult(
                source_name=self.source_name,
                papers=papers,
                total_found=int(payload.get("meta", {}).get("count", len(papers)) or 0),
                fetch_duration_seconds=time.perf_counter() - started,
                api_calls_made=api_calls,
            )
        except Exception as exc:
            logger.error("OpenAlex fetch failed: %s", exc)
            return FetchResult(
                source_name=self.source_name,
                fetch_duration_seconds=time.perf_counter() - started,
                error=str(exc),
                success=False,
                api_calls_made=api_calls,
            )

    def _parse_work(self, work: dict[str, Any]) -> Optional[PaperMetadata]:
        """Normalize one OpenAlex work while skipping records without titles."""

        title = work.get("title") or work.get("display_name") or ""
        if not title:
            return None
        open_access = work.get("open_access") or {}
        abstract = self._reconstruct_abstract(work.get("abstract_inverted_index") or {}) or None
        return PaperMetadata(
            title=title,
            doi=work.get("doi"),
            authors=[
                authorship.get("author", {}).get("display_name", "")
                for authorship in work.get("authorships", [])
                if authorship.get("author", {}).get("display_name")
            ],
            year=work.get("publication_year"),
            abstract=abstract,
            source_db=self.source_name,
            open_access=bool(open_access.get("is_oa")),
            open_access_url=open_access.get("oa_url") or work.get("id"),
            citation_count=int(work.get("cited_by_count") or 0),
            openalex_id=work.get("id"),
            has_experimental_section=detect_experimental_section(abstract or ""),
        )

    def _reconstruct_abstract(self, inverted_index: dict[str, list[int]]) -> str:
        """Reconstruct OpenAlex's inverted-index abstract into readable text."""

        positions: list[tuple[int, str]] = []
        for word, indexes in inverted_index.items():
            for index in indexes:
                positions.append((index, word))
        return " ".join(word for _, word in sorted(positions, key=lambda item: item[0]))

    async def check_health(self) -> bool:
        """Check OpenAlex availability with a one-result chemistry search."""

        try:
            client = await self._get_client()
            response = await client.get(
                f"{BASE_URL}/works",
                params={"search": "chemistry", "per_page": 1, "mailto": self._email},
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception as exc:
            logger.warning("OpenAlex health check failed: %s", exc)
            return False


class OpenAlexAbstractAcquirer(BaseTextAcquirer):
    """OpenAlexAbstractAcquirer emits reconstructed abstracts as TextChunks."""

    @property
    def acquirer_name(self) -> str:
        """Return the acquirer's registry key."""

        return "openalex_abstract"

    def priority(self) -> int:
        """Run after full-text acquirers because abstracts are lower completeness."""

        return 20

    async def can_acquire(self, paper: PaperMetadata) -> bool:
        """Acquire only papers with non-trivial abstracts."""

        return paper.abstract is not None and len(paper.abstract) > 50

    async def acquire(self, paper: PaperMetadata) -> list[TextChunk]:
        """Wrap the OpenAlex abstract in the unified TextChunk schema."""

        if not paper.abstract:
            return []
        return [
            TextChunk(
                chunk_id=generate_chunk_id(paper.dedup_key(), "abstract"),
                paper_metadata=paper,
                text=paper.abstract,
                section_name="abstract",
                completeness=TextCompleteness.ABSTRACT,
                acquisition_method=AcquisitionMethod.OPENALEX_ABSTRACT,
                chemical_density_score=calculate_chemical_density(paper.abstract),
                word_count=len(paper.abstract.split()),
            )
        ]
