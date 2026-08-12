"""ArXiv fetcher and ar5iv HTML acquirer.

This file contains ArXivFetcher and ArXivHTMLAcquirer. ArXiv metadata comes
from the Atom API, and full-text acquisition uses ar5iv HTML sections because
the project deliberately avoids PDF parsing for API-backed sources.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Optional
from xml.etree import ElementTree as ET

import httpx
from bs4 import BeautifulSoup

from src.ingestion.base import BaseFetcher, BaseTextAcquirer
from src.ingestion.utils import calculate_chemical_density, detect_experimental_section, generate_chunk_id
from src.schemas.paper import AcquisitionMethod, FetchResult, PaperMetadata, TextChunk, TextCompleteness
from src.schemas.query import SearchPlan

logger = logging.getLogger(__name__)

BASE_URL = "https://export.arxiv.org/api/query"
AR5IV_BASE = "https://ar5iv.org/abs"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
SECTION_PATTERNS = [
    re.compile(r"experimental\s*(section|methods?|procedures?)?", re.IGNORECASE),
    re.compile(r"materials?\s+and\s+methods?", re.IGNORECASE),
    re.compile(r"synthesis\s+of", re.IGNORECASE),
    re.compile(r"preparation\s+of", re.IGNORECASE),
    re.compile(r"synthetic\s+procedure", re.IGNORECASE),
]


def _node_text(node: Optional[ET.Element]) -> str:
    """Extract XML text safely because Atom entries may omit optional fields."""

    return " ".join((node.text or "").split()) if node is not None else ""


class ArXivFetcher(BaseFetcher):
    """ArXivFetcher searches open preprints and records ar5iv full-text URLs."""

    def __init__(self) -> None:
        """Create a fetcher with a lazily constructed AsyncClient."""

        self._client: Optional[httpx.AsyncClient] = None

    @property
    def source_name(self) -> str:
        """Return ArXiv's registry key."""

        return "arxiv"

    def supports_full_text(self) -> bool:
        """Return true because ar5iv HTML can provide full-text sections."""

        return True

    async def _get_client(self) -> httpx.AsyncClient:
        """Create or reuse one AsyncClient for ArXiv calls."""

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        return self._client

    async def fetch(self, query: str, search_plan: SearchPlan) -> FetchResult:
        """Fetch ArXiv Atom results and normalize entries into PaperMetadata."""

        started = time.perf_counter()
        api_calls = 0
        try:
            source_query = search_plan.source_queries.get(self.source_name)
            query_string = source_query.query_string if source_query else query
            max_results = source_query.max_results if source_query else 25
            await asyncio.sleep(3.0)
            client = await self._get_client()
            response = await client.get(
                BASE_URL,
                params={
                    "search_query": f"all:{query_string}",
                    "max_results": max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
            )
            api_calls += 1
            response.raise_for_status()
            papers = self._parse_atom(response.text)
            return FetchResult(
                source_name=self.source_name,
                papers=papers,
                total_found=len(papers),
                fetch_duration_seconds=time.perf_counter() - started,
                api_calls_made=api_calls,
            )
        except Exception as exc:
            logger.error("ArXiv fetch failed: %s", exc)
            return FetchResult(
                source_name=self.source_name,
                fetch_duration_seconds=time.perf_counter() - started,
                error=str(exc),
                success=False,
                api_calls_made=api_calls,
            )

    def _parse_atom(self, xml_text: str) -> list[PaperMetadata]:
        """Parse Atom XML while preserving identifiers needed by ar5iv."""

        root = ET.fromstring(xml_text)
        papers: list[PaperMetadata] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            entry_id = _node_text(entry.find("atom:id", ATOM_NS))
            match = re.search(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", entry_id)
            arxiv_id = match.group(1) if match else None
            title = _node_text(entry.find("atom:title", ATOM_NS))
            abstract = _node_text(entry.find("atom:summary", ATOM_NS)) or None
            authors = [
                _node_text(author.find("atom:name", ATOM_NS))
                for author in entry.findall("atom:author", ATOM_NS)
                if _node_text(author.find("atom:name", ATOM_NS))
            ]
            published = _node_text(entry.find("atom:published", ATOM_NS))
            year = int(published[:4]) if published[:4].isdigit() else None
            doi = None
            for link in entry.findall("atom:link", ATOM_NS):
                if link.get("title") == "doi":
                    doi = link.get("href")
                    break
            if not title:
                continue
            papers.append(
                PaperMetadata(
                    title=title,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    source_db=self.source_name,
                    open_access=True,
                    open_access_url=f"{AR5IV_BASE}/{arxiv_id}" if arxiv_id else None,
                    has_experimental_section=detect_experimental_section(abstract or ""),
                )
            )
        return papers

    async def check_health(self) -> bool:
        """Check ArXiv reachability with a small category query."""

        try:
            await asyncio.sleep(3.0)
            client = await self._get_client()
            response = await client.get(BASE_URL, params={"search_query": "cat:cond-mat", "max_results": 1})
            return response.status_code == 200
        except Exception as exc:
            logger.warning("ArXiv health check failed: %s", exc)
            return False


class ArXivHTMLAcquirer(BaseTextAcquirer):
    """ArXivHTMLAcquirer extracts experimental sections from ar5iv HTML."""

    def __init__(self) -> None:
        """Create an acquirer with a lazy client for polite ar5iv access."""

        self._client: Optional[httpx.AsyncClient] = None

    @property
    def acquirer_name(self) -> str:
        """Return the acquirer's registry key."""

        return "arxiv_html"

    def priority(self) -> int:
        """Try ar5iv immediately after PMC XML because it can return full text."""

        return 2

    async def _get_client(self) -> httpx.AsyncClient:
        """Create or reuse one AsyncClient for ar5iv requests."""

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        return self._client

    async def can_acquire(self, paper: PaperMetadata) -> bool:
        """Use arxiv_id metadata to determine whether ar5iv can be attempted."""

        return paper.arxiv_id is not None

    async def acquire(self, paper: PaperMetadata) -> list[TextChunk]:
        """Acquire experimental HTML sections or return an abstract fallback."""

        try:
            await asyncio.sleep(1.0)
            client = await self._get_client()
            response = await client.get(f"{AR5IV_BASE}/{paper.arxiv_id}")
            if response.status_code == 404:
                logger.info("ar5iv HTML unavailable for %s", paper.arxiv_id)
                return self._abstract_fallback(paper)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            chunks = self._extract_sections(soup, paper)
            return chunks if chunks else self._abstract_fallback(paper)
        except Exception as exc:
            logger.warning("ar5iv acquisition failed for %s: %s", paper.title, exc)
            return self._abstract_fallback(paper)

    def _extract_sections(self, soup: BeautifulSoup, paper: PaperMetadata) -> list[TextChunk]:
        """Find headers matching experimental patterns and collect section text."""

        chunks: list[TextChunk] = []
        headers = soup.find_all(re.compile(r"^h[1-6]$"))
        for header in headers:
            header_text = header.get_text(separator=" ", strip=True)
            if not any(pattern.search(header_text) for pattern in SECTION_PATTERNS):
                continue
            section_parts = [header_text]
            for sibling in header.find_next_siblings():
                if sibling.name and re.fullmatch(r"h[1-6]", sibling.name):
                    break
                for removable in sibling.find_all(["figure", "table"]):
                    removable.decompose()
                section_parts.append(sibling.get_text(separator=" ", strip=True))
            text = re.sub(r"\s+", " ", " ".join(section_parts)).strip()
            word_count = len(text.split())
            if word_count < 50:
                continue
            chunks.append(
                TextChunk(
                    chunk_id=generate_chunk_id(paper.dedup_key(), header_text),
                    paper_metadata=paper,
                    text=text,
                    section_name=header_text,
                    completeness=TextCompleteness.FULL_TEXT,
                    acquisition_method=AcquisitionMethod.ARXIV_HTML,
                    chemical_density_score=calculate_chemical_density(text),
                    word_count=word_count,
                )
            )
        return chunks

    def _abstract_fallback(self, paper: PaperMetadata) -> list[TextChunk]:
        """Return a consistent abstract chunk when ar5iv full text is unavailable."""

        if not paper.abstract or len(paper.abstract) <= 50:
            return []
        return [
            TextChunk(
                chunk_id=generate_chunk_id(paper.dedup_key(), "abstract"),
                paper_metadata=paper,
                text=paper.abstract,
                section_name="abstract",
                completeness=TextCompleteness.ABSTRACT,
                acquisition_method=AcquisitionMethod.ABSTRACT_ONLY,
                chemical_density_score=calculate_chemical_density(paper.abstract),
                word_count=len(paper.abstract.split()),
            )
        ]
