"""PubMed fetcher and PubMed Central XML acquirer.

This file contains PubMedFetcher and PubMedCentralAcquirer. PubMed searching
uses E-utilities metadata and abstracts, while full-text acquisition is reserved
for PMC XML methods sections to avoid brittle PDF parsing in the API path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional
from xml.etree import ElementTree as ET

import httpx

from src.ingestion.base import BaseFetcher, BaseTextAcquirer
from src.ingestion.utils import calculate_chemical_density, detect_experimental_section, generate_chunk_id
from src.schemas.paper import AcquisitionMethod, FetchResult, PaperMetadata, TextChunk, TextCompleteness
from src.schemas.query import SearchPlan

logger = logging.getLogger(__name__)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EXPERIMENTAL_KEYWORDS = [
    "synthesis",
    "prepared",
    "dissolved",
    "stirred",
    "heated",
    "refluxed",
    "filtered",
    "experimental",
    "procedure",
    "temperature",
    "reaction",
    "mmol",
]
PMC_SECTION_TITLES = {
    "experimental",
    "experimental section",
    "experimental methods",
    "materials and methods",
    "methods",
    "synthesis",
    "preparation",
    "experimental details",
    "synthetic procedures",
}


def _element_text(element: Optional[ET.Element]) -> str:
    """Extract mixed XML text because article titles often contain nested tags."""

    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


class PubMedFetcher(BaseFetcher):
    """PubMedFetcher searches NCBI PubMed while respecting unauthenticated limits."""

    def __init__(self) -> None:
        """Create a fetcher with lazy HTTP client initialization for testability."""

        self._client: Optional[httpx.AsyncClient] = None
        self._email = os.getenv("PUBMED_EMAIL") or os.getenv("API_EMAIL", "your.email@example.com")
        self._api_key = os.getenv("PUBMED_API") or os.getenv("NCBI_API_KEY")

    @property
    def source_name(self) -> str:
        """Return PubMed's registry key."""

        return "pubmed"

    async def _get_client(self) -> httpx.AsyncClient:
        """Create or reuse one AsyncClient to avoid connection churn."""

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _params(self, params: dict[str, str | int]) -> dict[str, str | int]:
        """Attach NCBI contact metadata because E-utilities expects identification."""

        enriched = {**params, "email": self._email}
        if self._api_key:
            enriched["api_key"] = self._api_key
        return enriched

    async def fetch(self, query: str, search_plan: SearchPlan) -> FetchResult:
        """Fetch PubMed abstracts and normalize them into PaperMetadata records."""

        started = time.perf_counter()
        api_calls = 0
        try:
            source_query = search_plan.source_queries.get(self.source_name)
            query_string = source_query.query_string if source_query else query
            max_results = source_query.max_results if source_query else 25
            client = await self._get_client()

            search_response = await client.get(
                f"{BASE_URL}/esearch.fcgi",
                params=self._params(
                    {
                        "db": "pubmed",
                        "term": query_string,
                        "retmax": max_results,
                        "retmode": "json",
                        "sort": "relevance",
                    }
                ),
            )
            api_calls += 1
            search_response.raise_for_status()
            search_payload = search_response.json()
            ids = search_payload.get("esearchresult", {}).get("idlist", [])
            total_found = int(search_payload.get("esearchresult", {}).get("count", 0) or 0)
            if not ids:
                return FetchResult(
                    source_name=self.source_name,
                    total_found=total_found,
                    fetch_duration_seconds=time.perf_counter() - started,
                    api_calls_made=api_calls,
                )

            await asyncio.sleep(0.35)
            fetch_response = await client.get(
                f"{BASE_URL}/efetch.fcgi",
                params=self._params(
                    {
                        "db": "pubmed",
                        "id": ",".join(ids),
                        "retmode": "xml",
                        "rettype": "abstract",
                    }
                ),
            )
            api_calls += 1
            fetch_response.raise_for_status()
            papers = self._parse_pubmed_xml(fetch_response.text)
            return FetchResult(
                source_name=self.source_name,
                papers=papers,
                total_found=total_found,
                fetch_duration_seconds=time.perf_counter() - started,
                api_calls_made=api_calls,
            )
        except Exception as exc:
            logger.error("PubMed fetch failed: %s", exc)
            return FetchResult(
                source_name=self.source_name,
                fetch_duration_seconds=time.perf_counter() - started,
                error=str(exc),
                success=False,
                api_calls_made=api_calls,
            )

    def _parse_pubmed_xml(self, xml_text: str) -> list[PaperMetadata]:
        """Parse PubMed XML into normalized paper records for downstream merging."""

        root = ET.fromstring(xml_text)
        papers: list[PaperMetadata] = []
        for article in root.findall(".//PubmedArticle"):
            title = _element_text(article.find(".//ArticleTitle"))
            if not title:
                continue
            abstract_parts = [_element_text(element) for element in article.findall(".//AbstractText")]
            abstract = " ".join(part for part in abstract_parts if part) or None
            authors = []
            for author in article.findall(".//Author"):
                last_name = _element_text(author.find("LastName"))
                fore_name = _element_text(author.find("ForeName"))
                full_name = " ".join(part for part in (fore_name, last_name) if part)
                if full_name:
                    authors.append(full_name)
            year_text = _element_text(article.find(".//PubDate/Year"))
            year = int(year_text[:4]) if year_text[:4].isdigit() else None
            doi = None
            pmc_id = None
            for article_id in article.findall(".//ArticleId"):
                id_type = article_id.get("IdType")
                if id_type == "doi":
                    doi = _element_text(article_id)
                elif id_type == "pmc":
                    pmc_id = _element_text(article_id)
            experimental_text = abstract or ""
            has_experimental = any(keyword in experimental_text.lower() for keyword in EXPERIMENTAL_KEYWORDS)
            pubmed_id = _element_text(article.find(".//PMID")) or None
            papers.append(
                PaperMetadata(
                    title=title,
                    doi=doi,
                    pubmed_id=pubmed_id,
                    pubmed_central_id=pmc_id,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    source_db=self.source_name,
                    open_access=pmc_id is not None,
                    open_access_url=(
                        f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
                        if pmc_id
                        else f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/" if pubmed_id else None
                    ),
                    has_experimental_section=has_experimental,
                )
            )
        return papers

    async def check_health(self) -> bool:
        """Check PubMed availability with a lightweight EInfo request."""

        try:
            client = await self._get_client()
            response = await client.get(
                f"{BASE_URL}/einfo.fcgi",
                params=self._params({"db": "pubmed", "retmode": "json"}),
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception as exc:
            logger.warning("PubMed health check failed: %s", exc)
            return False


class PubMedCentralAcquirer(BaseTextAcquirer):
    """PubMedCentralAcquirer extracts methods-like sections from PMC XML."""

    def __init__(self) -> None:
        """Create an acquirer with lazy HTTP client initialization for reuse."""

        self._client: Optional[httpx.AsyncClient] = None
        self._email = os.getenv("PUBMED_EMAIL") or os.getenv("API_EMAIL", "your.email@example.com")

    @property
    def acquirer_name(self) -> str:
        """Return the acquirer's registry key."""

        return "pubmed_central_xml"

    def priority(self) -> int:
        """Prefer PMC XML because it can provide structured full text."""

        return 1

    async def _get_client(self) -> httpx.AsyncClient:
        """Create or reuse one AsyncClient for PMC XML requests."""

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def can_acquire(self, paper: PaperMetadata) -> bool:
        """Use only PMC metadata to decide if XML acquisition is possible."""

        return paper.pubmed_central_id is not None

    async def acquire(self, paper: PaperMetadata) -> list[TextChunk]:
        """Acquire methods sections from PMC XML or fall back to the abstract."""

        try:
            client = await self._get_client()
            pmc_id = (paper.pubmed_central_id or "").removeprefix("PMC")
            response = await client.get(
                f"{BASE_URL}/efetch.fcgi",
                params={"db": "pmc", "id": pmc_id, "rettype": "xml", "retmode": "xml", "email": self._email},
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
            chunks = self._extract_sections(root, paper)
            if chunks:
                return chunks
            return self._abstract_fallback(paper)
        except Exception as exc:
            logger.warning("PMC acquisition failed for %s: %s", paper.title, exc)
            return []

    def _extract_sections(self, root: ET.Element, paper: PaperMetadata) -> list[TextChunk]:
        """Extract experimental sections using sec-type and section title signals."""

        chunks: list[TextChunk] = []
        for section in root.findall(".//sec"):
            sec_type = (section.get("sec-type") or "").lower()
            title = _element_text(section.find("title"))
            title_normalized = title.lower().strip()
            is_match = (
                "method" in sec_type
                or "material" in sec_type
                or title_normalized in PMC_SECTION_TITLES
                or "synthesis" in title_normalized
                or "experimental" in title_normalized
            )
            if not is_match:
                continue
            text = " ".join("".join(section.itertext()).split())
            word_count = len(text.split())
            if word_count < 50:
                continue
            section_name = title or "methods"
            chunks.append(
                TextChunk(
                    chunk_id=generate_chunk_id(paper.dedup_key(), section_name),
                    paper_metadata=paper,
                    text=text,
                    section_name=section_name,
                    completeness=TextCompleteness.FULL_TEXT,
                    acquisition_method=AcquisitionMethod.PUBMED_CENTRAL_XML,
                    chemical_density_score=calculate_chemical_density(text),
                    word_count=word_count,
                )
            )
        return chunks

    def _abstract_fallback(self, paper: PaperMetadata) -> list[TextChunk]:
        """Return an abstract chunk when PMC has no usable experimental section."""

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
