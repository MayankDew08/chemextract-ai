"""DOI-first and fuzzy-title deduplication for ingestion results.

This file contains DOITitleDeduplicator. The implementation keeps duplicate
collapse behind BaseDeduplicator so future experiments with graph-based or
embedding-based deduplication do not affect fetchers or orchestration.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from rapidfuzz import fuzz

from src.ingestion.base import BaseDeduplicator
from src.ingestion.utils import normalize_title
from src.schemas.paper import PaperMetadata

logger = logging.getLogger(__name__)


class DOITitleDeduplicator(BaseDeduplicator):
    """DOITitleDeduplicator merges exact DOI matches before fuzzy title matches."""

    def __init__(self, title_threshold: int = 88) -> None:
        """Store the title threshold so the policy can be tuned in tests."""

        self._title_threshold = title_threshold
        self._last_duplicate_count = 0

    def deduplicate(self, papers: list[PaperMetadata]) -> list[PaperMetadata]:
        """Deduplicate papers with DOI exact matching and rapidfuzz title matching."""

        before = len(papers)
        doi_groups: dict[str, PaperMetadata] = {}
        no_doi: list[PaperMetadata] = []
        for paper in papers:
            if paper.doi:
                key = f"doi:{paper.doi}"
                doi_groups[key] = self._merge(doi_groups[key], paper) if key in doi_groups else paper
            else:
                no_doi.append(paper)

        unique: list[PaperMetadata] = list(doi_groups.values())
        processed_no_doi: list[PaperMetadata] = []
        for paper in no_doi:
            match_index = self._find_title_match(paper, unique)
            if match_index is not None:
                unique[match_index] = self._merge(unique[match_index], paper)
                continue
            no_doi_match_index = self._find_title_match(paper, processed_no_doi)
            if no_doi_match_index is not None:
                processed_no_doi[no_doi_match_index] = self._merge(
                    processed_no_doi[no_doi_match_index],
                    paper,
                )
                continue
            processed_no_doi.append(paper)

        unique.extend(processed_no_doi)
        self._last_duplicate_count = before - len(unique)
        logger.info("Deduplication: %s -> %s papers (%s removed)", before, len(unique), self._last_duplicate_count)
        return unique

    def duplicate_count(self) -> int:
        """Return the duplicate count from the most recent deduplication pass."""

        return self._last_duplicate_count

    def _find_title_match(self, paper: PaperMetadata, candidates: list[PaperMetadata]) -> Optional[int]:
        """Return the index of a fuzzy title match or None when no match exists."""

        title = normalize_title(paper.title)
        if not title:
            return None
        for index, candidate in enumerate(candidates):
            candidate_title = normalize_title(candidate.title)
            if candidate_title and fuzz.ratio(title, candidate_title) >= self._title_threshold:
                return index
        return None

    def _merge(self, primary: PaperMetadata, secondary: PaperMetadata) -> PaperMetadata:
        """Merge two metadata records, preferring the richer one and filling gaps."""

        richer, poorer = self._order_by_richness(primary, secondary)
        merged: dict[str, Any] = richer.model_dump()
        poorer_data = poorer.model_dump()
        for key, value in poorer_data.items():
            if key == "source_db":
                continue
            if self._is_missing(merged.get(key)) and not self._is_missing(value):
                merged[key] = value
        if primary.source_db != secondary.source_db:
            sources = []
            for source in f"{richer.source_db}+{poorer.source_db}".split("+"):
                if source and source not in sources:
                    sources.append(source)
            merged["source_db"] = "+".join(sources)
        return PaperMetadata(**merged)

    def _order_by_richness(
        self,
        primary: PaperMetadata,
        secondary: PaperMetadata,
    ) -> tuple[PaperMetadata, PaperMetadata]:
        """Return papers ordered by count of populated fields."""

        primary_score = self._richness(primary)
        secondary_score = self._richness(secondary)
        return (secondary, primary) if secondary_score > primary_score else (primary, secondary)

    def _richness(self, paper: PaperMetadata) -> int:
        """Count populated fields so the highest information record survives."""

        return sum(1 for value in paper.model_dump().values() if not self._is_missing(value))

    def _is_missing(self, value: Any) -> bool:
        """Treat None, blank strings, and empty containers as mergeable gaps."""

        return value is None or value == "" or value == [] or value == {}
