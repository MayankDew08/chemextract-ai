"""Top-level ChemExtract AI integration pipeline for Phase 6A.

This module is the single coordinator that connects Phase 1 ingestion,
Phase 2 extraction, Phase 3 validation, Phase 4 graph storage, and Phase 5
observability surfaces.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from src.ingestion.utils import classify_chunk_domain, is_chemistry_text
from src.schemas.paper import TextChunk
from src.schemas.recipe import ChemicalRecipe

logger = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    """Configuration for one ChemExtract pipeline run."""

    query: str = Field(..., min_length=3)
    max_papers: int = Field(default=10, ge=1, le=25)
    max_chunks_per_paper: int = Field(default=3, ge=1, le=10)
    max_retries: int = Field(default=3, ge=1, le=5)
    llm_provider: Optional[str] = None
    graph_backend: Optional[str] = None
    preloaded_chunks: Optional[list[TextChunk]] = None
    verbose: bool = Field(default=True)
    chunk_timeout_seconds: float = Field(default=180.0, ge=8.0, le=600.0)
    inter_chunk_delay_seconds: float = Field(default=2.0, ge=0.0, le=60.0)


class ProgressEvent(BaseModel):
    """One progress update emitted during pipeline execution."""

    stage: str
    message: str
    chunk_index: Optional[int] = None
    chunks_total: Optional[int] = None
    recipe_id: Optional[str] = None
    validation_status: Optional[str] = None
    correction_count: Optional[int] = None
    total_recipes: Optional[int] = None
    passed: Optional[int] = None
    corrected: Optional[int] = None
    failed: Optional[int] = None
    duration_seconds: Optional[float] = None
    total_cost_usd: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_terminal_line(self) -> str:
        """Format this progress event for terminal output."""

        match self.stage:
            case "starting":
                return f"\n{'═' * 60}\n  {self.message}\n{'═' * 60}"
            case "fetching":
                return f"  [Phase 1] {self.message}"
            case "fetched":
                return f"  [Phase 1] ✅ {self.message}\n  {'─' * 56}"
            case "extracting":
                return f"  [Phase 2+3] Chunk {self.chunk_index}/{self.chunks_total}: {self.message}"
            case "stored":
                status_icon = {"PASSED": "✅", "CORRECTED": "⚠️ ", "FAILED": "❌"}.get(
                    self.validation_status,
                    "?",
                )
                corrections = f" ({self.correction_count} corrections)" if self.correction_count else ""
                return f"  [Phase 4] {status_icon} {self.recipe_id}{corrections} → stored"
            case "complete":
                return (
                    f"\n{'═' * 60}\n"
                    "  PIPELINE COMPLETE\n"
                    f"{'─' * 60}\n"
                    f"  Recipes extracted : {self.total_recipes}\n"
                    f"  Passed first try  : {self.passed}\n"
                    f"  Self-corrected    : {self.corrected}\n"
                    f"  Failed            : {self.failed}\n"
                    f"  Total duration    : {(self.duration_seconds or 0.0):.1f}s\n"
                    f"  Estimated cost    : ${(self.total_cost_usd or 0.0):.6f}\n"
                    f"{'═' * 60}"
                )
            case "error":
                return f"  [ERROR] ❌ {self.message}"
            case "skipped":
                return f"  [SKIP] ⏭️  {self.message}"
            case _:
                return f"  [{self.stage}] {self.message}"


class PipelineResult(BaseModel):
    """Final result of one complete ChemExtract pipeline run."""

    query: str
    papers_fetched: int = 0
    chunks_produced: int = 0
    sources_used: list[str] = Field(default_factory=list)
    fetch_duration_seconds: float = 0.0
    recipes_extracted: int = 0
    passed_first_try: int = 0
    self_corrected: int = 0
    failed: int = 0
    skipped_non_chemistry: int = 0
    total_corrections: int = 0
    extraction_duration_seconds: float = 0.0
    nodes_added: int = 0
    edges_added: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    llm_provider: str = ""
    llm_model: str = ""
    recipes: list[ChemicalRecipe] = Field(default_factory=list)
    events: list[ProgressEvent] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def success_rate(self) -> float:
        """Return the percentage of extracted recipes that passed validation."""

        if self.recipes_extracted == 0:
            return 0.0
        return (self.passed_first_try + self.self_corrected) / self.recipes_extracted * 100


class ChemExtractPipeline:
    """Top-level coordinator for the ChemExtract AI pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        """Create a request-scoped pipeline coordinator."""

        self.config = config
        self._result = PipelineResult(query=config.query)
        self._events: asyncio.Queue[ProgressEvent] = asyncio.Queue()
        self._done = False

    async def run(self) -> PipelineResult:
        """Execute the full pipeline and return only the final result."""

        async for _event in self.stream():
            pass
        return self._result

    async def stream(self) -> AsyncIterator[ProgressEvent]:
        """Execute the full pipeline and yield progress events as they occur."""

        pipeline_start = time.time()
        self._result.started_at = datetime.utcnow()
        try:
            yield self._emit(
                ProgressEvent(stage="starting", message=f"ChemExtract AI — Query: '{self.config.query}'"),
            )

            yield self._emit(
                ProgressEvent(
                    stage="fetching",
                    message="Searching PubMed, ArXiv, OpenAlex, Semantic Scholar...",
                ),
            )
            fetch_start = time.time()
            chunks = await self._run_phase1()
            self._result.fetch_duration_seconds = time.time() - fetch_start
            self._record_phase1_stats(chunks)
            yield self._emit(
                ProgressEvent(
                    stage="fetched",
                    message=(
                        f"Found {len(chunks)} text chunks from "
                        f"{len(self._result.sources_used)} sources: {', '.join(self._result.sources_used)}"
                    ),
                ),
            )

            if not chunks:
                self._result.completed_at = datetime.utcnow()
                yield self._emit(
                    ProgressEvent(stage="error", message="Phase 1 returned no chunks. Check API connectivity."),
                )
                return

            store = self._get_store()
            nodes_before = store.node_count()
            edges_before = store.edge_count()
            extraction_start = time.time()
            for index, chunk in enumerate(chunks, 1):
                async for event in self._process_chunk(chunk, index, len(chunks)):
                    yield event
                configured_provider = (self.config.llm_provider or os.getenv("LLM_PROVIDER", "")).lower()
                delay = 0.0 if configured_provider == "ollama" else self.config.inter_chunk_delay_seconds
                if index < len(chunks) and delay > 0:
                    await asyncio.sleep(delay)

            self._result.extraction_duration_seconds = time.time() - extraction_start
            self._result.nodes_added = store.node_count() - nodes_before
            self._result.edges_added = store.edge_count() - edges_before
            self._result.total_duration_seconds = time.time() - pipeline_start
            self._result.completed_at = datetime.utcnow()
            self._done = True
            yield self._emit(
                ProgressEvent(
                    stage="complete",
                    message="Pipeline finished",
                    total_recipes=self._result.recipes_extracted,
                    passed=self._result.passed_first_try,
                    corrected=self._result.self_corrected,
                    failed=self._result.failed,
                    duration_seconds=self._result.total_duration_seconds,
                    total_cost_usd=self._result.total_cost_usd,
                ),
            )
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            self._result.completed_at = datetime.utcnow()
            self._done = True
            yield self._emit(ProgressEvent(stage="error", message=str(exc)))

    async def _run_phase1(self) -> list[TextChunk]:
        """Run Phase 1 fetching, or return preloaded chunks for tests and demos."""

        if self.config.preloaded_chunks is not None:
            logger.info("Using preloaded chunks, skipping Phase 1 fetch")
            return self.config.preloaded_chunks

        from src.ingestion.deduplicator import DOITitleDeduplicator
        from src.ingestion.orchestrator import FetchOrchestrator
        from src.ingestion.query_parser import build_query_parser_from_env
        from src.ingestion.ranker import ChemistryRelevanceRanker
        from src.ingestion.registry import bootstrap_registry

        parser = build_query_parser_from_env()
        orchestrator = FetchOrchestrator(
            registry=bootstrap_registry(),
            deduplicator=DOITitleDeduplicator(),
            ranker=ChemistryRelevanceRanker(),
            query_parser=parser,
            top_k_papers=self.config.max_papers,
            max_chunks_per_paper=self.config.max_chunks_per_paper,
        )
        try:
            return await orchestrator.run(self.config.query, skip_health_check=False)
        except Exception as exc:
            logger.error("Phase 1 failed: %s", exc, exc_info=True)
            return []

    async def _process_chunk(
        self,
        chunk: TextChunk,
        index: int,
        total: int,
    ) -> AsyncIterator[ProgressEvent]:
        """Process one chunk through extraction, validation, storage, and metrics."""

        paper_title = (chunk.paper_metadata.title or "Untitled")[:50]
        if not is_chemistry_text(chunk.text, threshold=0.3):
            domain = classify_chunk_domain(chunk.text)
            logger.info("Chunk %s skipped: domain=%s", index, domain)
            self._result.skipped_non_chemistry += 1
            yield self._emit(
                ProgressEvent(
                    stage="skipped",
                    message=f"Chunk {index}/{total} skipped (not chemistry, domain={domain})",
                    chunk_index=index,
                    chunks_total=total,
                ),
            )
            return

        from src.agents.pipeline import ExtractionPipeline

        yield self._emit(
            ProgressEvent(
                stage="extracting",
                message=f"'{paper_title}...'",
                chunk_index=index,
                chunks_total=total,
            ),
        )
        try:
            provider = self._get_provider()
            extractor = ExtractionPipeline(provider=provider, graph_store=self._get_store(), persist_recipes=False)
            recipe = await asyncio.wait_for(extractor.extract(chunk), timeout=self._chunk_timeout_seconds())
            self._apply_recipe_stats(recipe)
            if recipe.validation_status.value in ("PASSED", "CORRECTED"):
                self._get_store().write_recipe_full(recipe)
            metrics_store = self._get_metrics_store()
            if metrics_store is not None:
                await metrics_store.record_recipe(recipe)
            self._result.recipes.append(recipe)
            yield self._emit(
                ProgressEvent(
                    stage="stored",
                    message="Recipe extracted",
                    chunk_index=index,
                    chunks_total=total,
                    recipe_id=recipe.recipe_id,
                    validation_status=recipe.validation_status.value,
                    correction_count=recipe.correction_count(),
                ),
            )
        except TimeoutError:
            message = f"Chunk {index} timed out after {self._chunk_timeout_seconds():.0f}s"
            logger.warning(message)
            self._result.failed += 1
            yield self._emit(ProgressEvent(stage="error", message=message))
        except Exception as exc:
            logger.error("Chunk %s processing failed: %s", index, exc, exc_info=True)
            self._result.failed += 1
            yield self._emit(ProgressEvent(stage="error", message=f"Chunk {index} failed: {str(exc)[:80]}"))

    def _record_phase1_stats(self, chunks: list[TextChunk]) -> None:
        """Copy Phase 1 source counts into the result object."""

        sources = sorted({chunk.paper_metadata.source_db for chunk in chunks})
        papers = {chunk.paper_metadata.dedup_key() for chunk in chunks}
        self._result.chunks_produced = len(chunks)
        self._result.papers_fetched = len(papers)
        self._result.sources_used = sources

    def _apply_recipe_stats(self, recipe: ChemicalRecipe) -> None:
        """Update aggregate result counters for one extracted recipe."""

        self._result.recipes_extracted += 1
        self._result.total_tokens_used += recipe.total_tokens_used
        self._result.total_cost_usd += recipe.estimated_cost_usd
        self._result.total_corrections += recipe.correction_count()
        self._result.llm_provider = recipe.llm_provider
        self._result.llm_model = recipe.llm_model
        match recipe.validation_status.value:
            case "PASSED":
                self._result.passed_first_try += 1
            case "CORRECTED":
                self._result.self_corrected += 1
            case "FAILED":
                self._result.failed += 1

    def _get_store(self):
        """Return the shared graph store used by FastAPI or standalone mode."""

        from src.storage.resolver import get_pipeline_store

        return get_pipeline_store(self.config.graph_backend)

    def _get_metrics_store(self):
        """Return the shared metrics store when one has been registered."""

        try:
            from src.storage.resolver import get_pipeline_metrics_store

            return get_pipeline_metrics_store()
        except Exception:
            return None

    def _get_provider(self):
        """Return a configured LLM provider or deterministic fallback for preloaded demos."""

        from src.llm.resolver import get_llm_provider

        return get_llm_provider(self.config.llm_provider)

    def _chunk_timeout_seconds(self) -> float:
        """Return a bounded timeout for one chunk extraction attempt."""

        return self.config.chunk_timeout_seconds

    def _emit(self, event: ProgressEvent) -> ProgressEvent:
        """Record and enqueue one event before handing it to the caller."""

        self._result.events.append(event)
        self._events.put_nowait(event)
        return event
