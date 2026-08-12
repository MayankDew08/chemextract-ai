"""Semantic Scholar fetcher and abstract acquirer.

This file contains SemanticScholarFetcher and SemanticScholarAbstractAcquirer.
Semantic Scholar contributes citation-rich metadata and abstracts, while
optional API-key support is isolated inside this concrete source class.
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

BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = (
    "paperId,url,title,abstract,year,authors,externalIds,citationCount,"
    "isOpenAccess,openAccessPdf"
)


class SemanticScholarFetcher(BaseFetcher):
    """SemanticScholarFetcher searches the Graph API with optional API-key headers."""

    def __init__(self) -> None:
        """Create a fetcher with lazy client construction and optional auth."""

        self._client: Optional[httpx.AsyncClient] = None
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self._headers = {"x-api-key": api_key} if api_key else {}

    @property
    def source_name(self) -> str:
        """Return Semantic Scholar's registry key."""

        return "semantic_scholar"

    async def _get_client(self) -> httpx.AsyncClient:
        """Create or reuse one AsyncClient for Graph API calls."""

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0, headers=self._headers)
        return self._client

    async def fetch(self, query: str, search_plan: SearchPlan) -> FetchResult:
        """Fetch Semantic Scholar papers and normalize their external IDs."""

        started = time.perf_counter()
        api_calls = 0
        try:
            source_query = search_plan.source_queries.get(self.source_name)
            query_string = source_query.query_string if source_query else query
            max_results = source_query.max_results if source_query else 25
            client = await self._get_client()
            response = await client.get(
                f"{BASE_URL}/paper/search",
                params={"query": query_string, "limit": min(max_results, 100), "fields": FIELDS},
            )
            api_calls += 1
            if response.status_code == 429:
                return FetchResult(
                    source_name=self.source_name,
                    fetch_duration_seconds=time.perf_counter() - started,
                    success=False,
                    error="Semantic Scholar rate limited",
                    api_calls_made=api_calls,
                    rate_limited=True,
                )
            response.raise_for_status()
            payload = response.json()
            papers = [self._parse_paper(item) for item in payload.get("data", []) if isinstance(item, dict)]
            papers = [paper for paper in papers if paper is not None]
            return FetchResult(
                source_name=self.source_name,
                papers=papers,
                total_found=int(payload.get("total", len(papers)) or 0),
                fetch_duration_seconds=time.perf_counter() - started,
                api_calls_made=api_calls,
            )
        except Exception as exc:
            logger.error("Semantic Scholar fetch failed: %s", exc)
            return FetchResult(
                source_name=self.source_name,
                fetch_duration_seconds=time.perf_counter() - started,
                error=str(exc),
                success=False,
                api_calls_made=api_calls,
            )

    def _parse_paper(self, paper: dict[str, Any]) -> Optional[PaperMetadata]:
        """Normalize a Semantic Scholar record while retaining cross-source IDs."""

        title = paper.get("title") or ""
        if not title:
            return None
        external_ids = paper.get("externalIds") or {}
        open_access_pdf = paper.get("openAccessPdf") or {}
        abstract = paper.get("abstract") or None
        paper_id = paper.get("paperId")
        source_url = (
            open_access_pdf.get("url")
            or paper.get("url")
            or (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None)
        )
        return PaperMetadata(
            title=title,
            doi=(external_ids.get("DOI") or "").lower() or None,
            arxiv_id=external_ids.get("ArXiv"),
            pubmed_id=external_ids.get("PubMed"),
            semantic_scholar_id=paper_id,
            abstract=abstract,
            authors=[author.get("name", "") for author in paper.get("authors", []) if author.get("name")],
            year=paper.get("year"),
            citation_count=int(paper.get("citationCount") or 0),
            open_access=bool(paper.get("isOpenAccess")),
            open_access_url=source_url,
            source_db=self.source_name,
            has_experimental_section=detect_experimental_section(abstract or ""),
        )

    async def check_health(self) -> bool:
        """Check Semantic Scholar availability with a tiny title-only search."""

        try:
            client = await self._get_client()
            response = await client.get(
                f"{BASE_URL}/paper/search",
                params={"query": "chemistry", "limit": 1, "fields": "title"},
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception as exc:
            logger.warning("Semantic Scholar health check failed: %s", exc)
            return False


class SemanticScholarAbstractAcquirer(BaseTextAcquirer):
    """SemanticScholarAbstractAcquirer emits API abstracts as TextChunks."""

    @property
    def acquirer_name(self) -> str:
        """Return the acquirer's registry key."""

        return "semantic_scholar_abstract"

    def priority(self) -> int:
        """Run after OpenAlex abstracts because this acquirer is another fallback."""

        return 21

    async def can_acquire(self, paper: PaperMetadata) -> bool:
        """Use Semantic Scholar IDs and abstract length to avoid weak chunks."""

        return (
            paper.semantic_scholar_id is not None
            and paper.abstract is not None
            and len(paper.abstract) > 50
        )

    async def acquire(self, paper: PaperMetadata) -> list[TextChunk]:
        """Wrap the Semantic Scholar abstract in the unified TextChunk schema."""

        if not paper.abstract:
            return []
        return [
            TextChunk(
                chunk_id=generate_chunk_id(paper.dedup_key(), "abstract"),
                paper_metadata=paper,
                text=paper.abstract,
                section_name="abstract",
                completeness=TextCompleteness.ABSTRACT,
                acquisition_method=AcquisitionMethod.SEMANTIC_SCHOLAR_API,
                chemical_density_score=calculate_chemical_density(paper.abstract),
                word_count=len(paper.abstract.split()),
            )
        ]
