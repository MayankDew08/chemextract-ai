"""Chemistry relevance ranking for deduplicated papers.

This file contains ChemistryRelevanceRanker. It encodes the Phase 1 scoring
policy as a swappable BaseRanker so the orchestrator remains independent of
ranking heuristics while recruiters can inspect deterministic scoring logic.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from src.ingestion.base import BaseRanker
from src.schemas.paper import PaperMetadata
from src.schemas.query import SearchPlan

logger = logging.getLogger(__name__)


class ChemistryRelevanceRanker(BaseRanker):
    """ChemistryRelevanceRanker scores papers by extractability and relevance."""

    def rank(
        self,
        papers: list[PaperMetadata],
        search_plan: SearchPlan,
        top_k: int = 15,
    ) -> list[PaperMetadata]:
        """Score and sort papers while returning new Pydantic objects."""

        scored = [
            paper.model_copy(update={"relevance_score": self._score_paper(paper, search_plan)})
            for paper in papers
        ]
        ranked = sorted(scored, key=lambda paper: paper.relevance_score, reverse=True)[:top_k]
        if ranked:
            logger.info(
                "Ranking returned %s papers: top=%.3f bottom=%.3f",
                len(ranked),
                ranked[0].relevance_score,
                ranked[-1].relevance_score,
            )
        return ranked

    def _score_paper(self, paper: PaperMetadata, search_plan: SearchPlan) -> float:
        """Apply the Phase 1 weighted relevance formula exactly once per paper."""

        score = 0.0
        if paper.has_experimental_section is True:
            score += 0.30
        if paper.open_access is True:
            score += 0.20
        score += min(math.log10(paper.citation_count + 1) / 5.0, 0.20)
        if paper.year:
            age = datetime.now(UTC).year - paper.year
            if age <= 2:
                score += 0.15
            elif age <= 5:
                score += 0.10
            elif age <= 10:
                score += 0.05
        title = paper.title.lower()
        target = search_plan.target_compound.lower()
        if target and target in title:
            score += 0.15
        elif any(synonym.lower() in title for synonym in search_plan.synonyms):
            score += 0.08
        return min(score, 1.0)
