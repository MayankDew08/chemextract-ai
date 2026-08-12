"""Abstract contracts for ChemExtract ingestion components.

This file contains BaseFetcher, BaseTextAcquirer, BaseDeduplicator, BaseRanker,
and BaseQueryParser. The orchestrator depends only on these contracts so future
sources such as ChemRxiv can be added by registering one new concrete class.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.schemas.paper import FetchResult, PaperMetadata, TextChunk
from src.schemas.query import SearchPlan

logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    """BaseFetcher defines the searchable source contract for async ingestion."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the stable source identifier used by registry and SearchPlan."""

    @abstractmethod
    async def fetch(self, query: str, search_plan: SearchPlan) -> FetchResult:
        """Fetch papers while containing source-specific failures inside FetchResult."""

    @abstractmethod
    async def check_health(self) -> bool:
        """Check API reachability so orchestration can skip degraded sources."""

    def supports_full_text(self) -> bool:
        """Return whether this source can provide full text through an acquirer."""

        return False


class BaseTextAcquirer(ABC):
    """BaseTextAcquirer defines how ranked papers become source-independent chunks."""

    @property
    @abstractmethod
    def acquirer_name(self) -> str:
        """Return the stable acquirer identifier used for logging and ordering."""

    @abstractmethod
    async def can_acquire(self, paper: PaperMetadata) -> bool:
        """Check only metadata so acquisition ordering does not perform surprise IO."""

    @abstractmethod
    async def acquire(self, paper: PaperMetadata) -> list[TextChunk]:
        """Acquire text chunks for one paper using source-specific retrieval."""

    @abstractmethod
    def priority(self) -> int:
        """Return acquisition priority where lower values are attempted first."""


class BaseDeduplicator(ABC):
    """BaseDeduplicator defines duplicate collapse behind a swappable policy."""

    @abstractmethod
    def deduplicate(self, papers: list[PaperMetadata]) -> list[PaperMetadata]:
        """Return papers with duplicates merged into richer metadata records."""

    @abstractmethod
    def duplicate_count(self) -> int:
        """Return duplicates removed in the most recent deduplication run."""


class BaseRanker(ABC):
    """BaseRanker defines relevance scoring without binding orchestration to a formula."""

    @abstractmethod
    def rank(
        self,
        papers: list[PaperMetadata],
        search_plan: SearchPlan,
        top_k: int,
    ) -> list[PaperMetadata]:
        """Return the highest value papers for text acquisition."""


class BaseQueryParser(ABC):
    """BaseQueryParser turns raw user intent into source-specific search plans."""

    @abstractmethod
    async def parse(self, raw_query: str, registered_sources: list[str]) -> SearchPlan:
        """Parse user text into the immutable SearchPlan contract."""


class InputSourceType(str, Enum):
    """Input source kind for user-provided text processors."""

    PDF_UPLOAD = "pdf_upload"
    URL_FETCH = "url_fetch"
    API_FETCH = "api_fetch"


class TextInputResult(BaseModel):
    """Result of processing one user-provided PDF or URL input."""

    source_type: InputSourceType
    source_identifier: str
    chunks: list[TextChunk] = Field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    error_type: Optional[str] = None
    total_pages: Optional[int] = None
    total_words_extracted: int = 0
    words_in_chunks: int = 0
    sections_found: list[str] = Field(default_factory=list)
    processing_seconds: float = 0.0
    is_paywalled: bool = False
    open_access_alternatives: list[str] = Field(default_factory=list)


class BaseTextInputProcessor(ABC):
    """Abstract interface for processors that turn a specific input into TextChunks."""

    @property
    @abstractmethod
    def processor_name(self) -> str:
        """Return a stable processor identifier."""

    @property
    @abstractmethod
    def supported_input_type(self) -> InputSourceType:
        """Return the input source type handled by this processor."""

    @abstractmethod
    async def process(self, source: str, title_hint: Optional[str] = None) -> TextInputResult:
        """Process one source and return TextChunks without raising."""

    @abstractmethod
    async def can_process(self, source: str) -> bool:
        """Return whether this processor can handle the source without network calls."""
