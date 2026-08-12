"""Open-content URL fetcher for Phase 6B user-provided inputs."""

from __future__ import annotations

import logging
import re
import time
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from src.ingestion.base import BaseTextInputProcessor, InputSourceType, TextInputResult
from src.ingestion.utils import calculate_chemical_density, detect_experimental_section, generate_chunk_id
from src.schemas.paper import AcquisitionMethod, PaperMetadata, TextChunk, TextCompleteness

logger = logging.getLogger(__name__)


class PaywallDetectionResult(BaseModel):
    """Result of paywall detection on a fetched or candidate page."""

    is_paywalled: bool
    publisher: Optional[str] = None
    open_access_alternatives: list[str] = Field(default_factory=list)
    message: str = ""


class URLFetcher(BaseTextInputProcessor):
    """Fetch open-access article URLs and convert accessible text into TextChunks."""

    processor_name = "url_fetcher"
    supported_input_type = InputSourceType.URL_FETCH

    PAYWALLED_DOMAINS = {
        "pubs.acs.org": "American Chemical Society (ACS)",
        "www.sciencedirect.com": "Elsevier/ScienceDirect",
        "onlinelibrary.wiley.com": "Wiley Online Library",
        "www.nature.com": "Nature/Springer",
        "link.springer.com": "Springer",
        "pubs.rsc.org": "Royal Society of Chemistry",
        "www.tandfonline.com": "Taylor & Francis",
        "www.science.org": "AAAS/Science",
        "chemistry.acs.org": "ACS",
        "www.chemistryworld.com": "Royal Society of Chemistry",
    }
    OPEN_ACCESS_DOMAINS = {
        "arxiv.org": "arxiv",
        "ar5iv.org": "ar5iv",
        "pmc.ncbi.nlm.nih.gov": "pubmed_central",
        "www.ncbi.nlm.nih.gov": "pubmed_central",
        "chemrxiv.org": "chemrxiv",
        "www.biorxiv.org": "biorxiv",
        "www.mdpi.com": "mdpi",
        "journals.plos.org": "plos",
        "www.frontiersin.org": "frontiers",
        "peerj.com": "peerj",
        "f1000research.com": "f1000",
        "www.rsc.org": "rsc_oa",
    }
    CONTENT_SELECTORS = {
        "arxiv.org": [".abstract", "#abs"],
        "ar5iv.org": ["article", ".ltx_document", "body"],
        "pmc.ncbi.nlm.nih.gov": ["#mc_methods", ".jig-ncbiinpagenav-heading", "#methods", ".sec", "article"],
        "chemrxiv.org": [".article-content", "article"],
        "www.mdpi.com": [".html-article-content", "article"],
        "journals.plos.org": ["#artText", "article"],
        "www.frontiersin.org": [".JournalAbstract", "article"],
        "default": ["article", "main", ".article-body", ".paper-content", "#content", "body"],
    }

    def __init__(self, timeout: float = 20.0) -> None:
        """Configure HTTP timeout and lazy client construction."""

        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return a reusable async HTTP client."""

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ChemExtractAI/1.0; academic research tool)",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying async HTTP client if it was opened."""

        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def can_process(self, source: str) -> bool:
        """Return whether source is an HTTP or HTTPS URL."""

        parsed = urlparse(source)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)

    async def process(self, source: str, title_hint: Optional[str] = None) -> TextInputResult:
        """Fetch URL content and convert accessible chemistry text into TextChunks."""

        start_time = time.time()
        result = TextInputResult(source_type=InputSourceType.URL_FETCH, source_identifier=source)
        try:
            if not await self.can_process(source):
                result.success = False
                result.error = "Invalid URL. Use an http:// or https:// URL."
                result.error_type = "INVALID_URL"
                return result

            normalized_url, _url_type = self._normalize_url(source)
            paywall = self._check_paywall_by_domain(normalized_url)
            if paywall.is_paywalled:
                result.success = False
                result.is_paywalled = True
                result.error_type = "PAYWALL_DETECTED"
                result.error = (
                    f"This paper is behind a paywall ({paywall.publisher}). "
                    "ChemExtract AI only accesses open content. See alternatives below."
                )
                result.open_access_alternatives = paywall.open_access_alternatives
                return result

            response = await self._get_client().get(normalized_url)
            response_paywall = self._detect_paywall_in_response(response, normalized_url)
            if response_paywall.is_paywalled:
                result.success = False
                result.is_paywalled = True
                result.error_type = "PAYWALL_DETECTED"
                result.error = "This page requires login or subscription. ChemExtract AI cannot access restricted content."
                result.open_access_alternatives = response_paywall.open_access_alternatives
                return result
            if response.status_code != 200:
                result.success = False
                result.error = f"HTTP {response.status_code} from {normalized_url}"
                result.error_type = "HTTP_ERROR"
                return result

            domain = self._get_domain(normalized_url)
            article_text, page_title = self._extract_article_text(response.text, domain)
            if not article_text or len(article_text.split()) < 100:
                result.success = False
                result.error = "Could not extract sufficient text from this page."
                result.error_type = "EXTRACTION_FAILED"
                return result

            result.total_words_extracted = len(article_text.split())
            sections = self._find_experimental_sections(article_text)
            result.sections_found = [section[0] for section in sections]
            if not sections:
                sections = self._best_paragraphs(article_text)

            paper_meta = PaperMetadata(
                title=title_hint or page_title or domain,
                doi=self._extract_doi_from_url(normalized_url),
                authors=[],
                year=None,
                abstract=None,
                source_db="url_fetch",
                open_access=True,
                open_access_url=source,
            )
            chunks = []
            for section_name, section_text in sections[:5]:
                words = len(section_text.split())
                if words < 50:
                    continue
                chunks.append(
                    TextChunk(
                        chunk_id=generate_chunk_id(f"url::{source}", section_name),
                        paper_metadata=paper_meta,
                        text=section_text,
                        section_name=section_name,
                        completeness=TextCompleteness.FULL_TEXT if result.sections_found else TextCompleteness.PARTIAL,
                        acquisition_method=AcquisitionMethod.USER_UPLOAD,
                        chemical_density_score=calculate_chemical_density(section_text),
                        word_count=words,
                    )
                )
            result.chunks = chunks
            result.words_in_chunks = sum(chunk.word_count for chunk in chunks)
            logger.info("URL fetched: %s -> %s chunks", source, len(chunks))
            return result
        except httpx.TimeoutException:
            result.success = False
            result.error = f"Request timed out after {self.timeout}s"
            result.error_type = "TIMEOUT"
            return result
        except Exception as exc:
            logger.error("URL fetch failed for %s: %s", source, exc)
            result.success = False
            result.error = str(exc)
            result.error_type = "UNEXPECTED_ERROR"
            return result
        finally:
            result.processing_seconds = time.time() - start_time

    def _normalize_url(self, url: str) -> tuple[str, str]:
        """Normalize URLs to an extraction-friendly open HTML endpoint."""

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if "arxiv.org" in domain:
            match = re.search(r"(?:abs|pdf)/(.+?)(?:v\d+)?(?:\.pdf)?$", parsed.path)
            if match:
                return f"https://ar5iv.org/abs/{match.group(1)}", "arxiv"
        return url, "generic"

    def _check_paywall_by_domain(self, url: str) -> PaywallDetectionResult:
        """Detect known paywalled publishers from the URL domain without fetching."""

        domain = self._get_domain(url)
        for paywalled_domain, publisher in self.PAYWALLED_DOMAINS.items():
            if paywalled_domain in domain:
                doi = self._extract_doi_from_url(url)
                return PaywallDetectionResult(
                    is_paywalled=True,
                    publisher=publisher,
                    open_access_alternatives=self._build_alternatives(doi, url),
                    message=f"This URL is from {publisher} which requires a subscription.",
                )
        return PaywallDetectionResult(is_paywalled=False)

    def _detect_paywall_in_response(self, response: httpx.Response, url: str) -> PaywallDetectionResult:
        """Detect login, subscription, and access-denied signals in an HTTP response."""

        final_url = str(response.url)
        login_signals = ["login", "signin", "sign-in", "authenticate", "access-denied", "subscribe", "purchase"]
        if any(signal in final_url.lower() for signal in login_signals):
            doi = self._extract_doi_from_url(url)
            return PaywallDetectionResult(is_paywalled=True, open_access_alternatives=self._build_alternatives(doi, url))
        if response.status_code in (401, 403):
            return PaywallDetectionResult(is_paywalled=True)
        body = response.text.lower()[:5000]
        phrases = [
            "you do not have access",
            "purchase access",
            "subscribe to read",
            "sign in to view",
            "full text available to subscribers",
            "access the full article",
            "get access to the full version",
        ]
        if any(phrase in body for phrase in phrases):
            doi = self._extract_doi_from_url(url)
            return PaywallDetectionResult(is_paywalled=True, open_access_alternatives=self._build_alternatives(doi, url))
        return PaywallDetectionResult(is_paywalled=False)

    def _build_alternatives(self, doi: Optional[str], original_url: str) -> list[str]:
        """Build user-facing open access alternatives for restricted content."""

        alternatives = []
        if doi:
            alternatives.append(f"Check open access version: https://unpaywall.org/{doi}")
            alternatives.append(f"Search Semantic Scholar: https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}")
            alternatives.append(f"Search Google Scholar: https://scholar.google.com/scholar?q={doi}")
        alternatives.append("If you have institutional access: download the PDF and use the Upload PDF tab instead.")
        alternatives.append("Search for the paper on ArXiv: https://arxiv.org/search/")
        return alternatives

    def _extract_article_text(self, html: str, domain: str) -> tuple[str, str]:
        """Extract main article text from HTML with domain selectors and fallbacks."""

        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        page_title = title_tag.get_text(strip=True) if title_tag else ""
        for selector in ["nav", "header", "footer", "aside", "script", "style", "noscript", ".references", "#references"]:
            for element in soup.select(selector):
                element.decompose()
        selectors = self.CONTENT_SELECTORS.get(domain) or self.CONTENT_SELECTORS["default"]
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = self._structured_element_text(element)
                if len(text.split()) > 100:
                    return text, page_title
        paragraphs = soup.find_all("p")
        blocks = [re.sub(r"\s+", " ", paragraph.get_text(" ", strip=True)).strip() for paragraph in paragraphs]
        return "\n\n".join(block for block in blocks if len(block) > 50), page_title

    def _structured_element_text(self, element: object) -> str:
        """Preserve heading and paragraph boundaries from an article element."""

        blocks = []
        for block in element.select("h1, h2, h3, h4, h5, h6, p, tr"):
            text = re.sub(r"\s+", " ", block.get_text(" ", strip=True)).strip()
            if text and (not blocks or text != blocks[-1]):
                blocks.append(text)
        if blocks:
            return "\n\n".join(blocks)
        return re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()

    def _find_experimental_sections(self, text: str) -> list[tuple[str, str]]:
        """Find experimental sections in HTML-derived text."""

        signals = [
            "experimental section",
            "experimental methods",
            "materials and methods",
            "synthesis of",
            "preparation of",
            "synthetic procedure",
            "experimental details",
            "experimental procedures",
        ]
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
        sections: list[tuple[int, float, str, str]] = []
        index = 0
        while index < len(paragraphs):
            paragraph = paragraphs[index].strip()
            lowered = paragraph.lower()
            is_section_heading = len(paragraph.split()) <= 25 and any(signal in lowered for signal in signals)
            has_measurement = bool(
                re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg|µ?L|mL|mmol|mol|mM|M|°C|hours?|hrs?|minutes?)\b", paragraph, re.IGNORECASE)
            )
            has_procedure_verb = bool(
                re.search(r"\b(?:dissolved|added|mixed|stirred|heated|refluxed|washed|dried|centrifuged|calcined|prepared)\b", paragraph, re.IGNORECASE)
            )
            is_procedure = len(paragraph.split()) > 30 and has_measurement and has_procedure_verb
            if is_section_heading or is_procedure:
                parts = [paragraph]
                next_index = index + 1
                word_count = len(paragraph.split())
                while next_index < len(paragraphs) and word_count < 900:
                    next_paragraph = paragraphs[next_index].strip()
                    if len(next_paragraph.split()) <= 25 and re.match(r"^(?:\d+(?:\.\d+)*\.?\s+|[A-Z][A-Za-z ]{2,}:?$)", next_paragraph):
                        break
                    if calculate_chemical_density(next_paragraph) > 0.1 or is_section_heading:
                        parts.append(next_paragraph)
                        word_count += len(next_paragraph.split())
                    next_index += 1
                section_text = " ".join(parts)
                if len(section_text.split()) >= 50:
                    specific_heading = "synthesis of" in lowered or "preparation of" in lowered or "synthetic procedure" in lowered
                    priority = 4 if specific_heading else 3 if is_section_heading else 2
                    density = calculate_chemical_density(section_text)
                    sections.append((priority, density, "experimental_section", section_text))
                index = next_index
            else:
                index += 1
        sections.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if any(priority == 4 for priority, _density, _name, _text in sections):
            sections = [section for section in sections if section[0] == 4]
        return [(name, section_text) for _priority, _density, name, section_text in sections]

    def _best_paragraphs(self, text: str) -> list[tuple[str, str]]:
        """Return highest-density chemistry paragraphs."""

        paragraphs = re.split(r"\n{2,}", text)
        scored = []
        for index, paragraph in enumerate(paragraphs):
            if len(paragraph.split()) < 50:
                continue
            score = calculate_chemical_density(paragraph)
            if score > 0.05:
                scored.append((f"paragraph_{index}", paragraph, score))
        scored.sort(key=lambda item: item[2], reverse=True)
        return [(name, paragraph) for name, paragraph, _score in scored[:3]]

    def _get_domain(self, url: str) -> str:
        """Return lower-cased URL domain."""

        return urlparse(url).netloc.lower()

    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        """Extract a DOI from common DOI URL patterns."""

        patterns = [
            re.compile(r"doi\.org/(10\.\d{4,}/\S+)$"),
            re.compile(r"/doi/(?:abs/|full/|pdf/)?(10\.\d{4,}/\S+)$"),
            re.compile(r"(10\.\d{4,}/\S+)"),
        ]
        for pattern in patterns:
            match = pattern.search(url)
            if match:
                return match.group(1).rstrip(").,;")
        return None
