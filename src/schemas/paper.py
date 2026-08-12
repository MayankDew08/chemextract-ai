"""Paper and text schemas for ChemExtract ingestion.

This file contains PaperMetadata, TextChunk, FetchResult, TextCompleteness,
and AcquisitionMethod. These models create one normalized boundary across
heterogeneous literature APIs so Phase 2 can consume chunks without source
specific conditionals.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class TextCompleteness(str, Enum):
    """TextCompleteness records whether a chunk is full text or a fallback."""

    FULL_TEXT = "full_text"
    ABSTRACT = "abstract"
    PARTIAL = "partial"
    NONE = "none"


class AcquisitionMethod(str, Enum):
    """AcquisitionMethod preserves provenance for ranking and auditing."""

    PUBMED_CENTRAL_XML = "pubmed_central_xml"
    ARXIV_HTML = "arxiv_html"
    OPEN_ACCESS_PDF = "open_access_pdf"
    SEMANTIC_SCHOLAR_API = "semantic_scholar_api"
    OPENALEX_ABSTRACT = "openalex_abstract"
    ABSTRACT_ONLY = "abstract_only"
    USER_UPLOAD = "user_upload"
    UNKNOWN = "unknown"


class PaperMetadata(BaseModel):
    """PaperMetadata normalizes source metadata while retaining source identifiers."""

    title: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pubmed_id: Optional[str] = None
    pubmed_central_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    openalex_id: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = Field(None, ge=1900, le=2100)
    abstract: Optional[str] = None
    source_db: str
    open_access: bool = False
    open_access_url: Optional[str] = None
    citation_count: int = Field(default=0, ge=0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    has_experimental_section: Optional[bool] = None

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: Optional[str]) -> Optional[str]:
        """Normalize DOI variants so exact-match deduplication is reliable."""

        if value is None:
            return None
        normalized = value.strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if normalized.startswith(prefix):
                normalized = normalized.removeprefix(prefix)
        return normalized.strip() or None

    def dedup_key(self) -> str:
        """Return a stable deduplication key for DOI-first, title-second merging."""

        if self.doi:
            return f"doi:{self.doi}"
        title_hash = hashlib.md5(self.title.lower().strip().encode(), usedforsecurity=False).hexdigest()
        return f"title:{title_hash[:12]}"


class TextChunk(BaseModel):
    """TextChunk is the source-independent unit passed into the Phase 2 graph."""

    chunk_id: str
    paper_metadata: PaperMetadata
    text: str = Field(..., min_length=50)
    section_name: Optional[str] = None
    completeness: TextCompleteness = TextCompleteness.ABSTRACT
    acquisition_method: AcquisitionMethod = AcquisitionMethod.UNKNOWN
    chemical_density_score: float = Field(default=0.0, ge=0.0, le=1.0)
    word_count: int = Field(default=0)
    acquired_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("text")
    @classmethod
    def text_not_just_whitespace(cls, value: str) -> str:
        """Reject whitespace-only chunks because they poison downstream extraction."""

        stripped = value.strip()
        if not stripped:
            raise ValueError("text cannot be empty after stripping whitespace")
        return stripped


class FetchResult(BaseModel):
    """FetchResult isolates source failures so orchestration can degrade gracefully."""

    source_name: str
    papers: list[PaperMetadata] = Field(default_factory=list)
    total_found: int = 0
    fetch_duration_seconds: float = 0.0
    error: Optional[str] = None
    success: bool = True
    api_calls_made: int = 0
    rate_limited: bool = False
