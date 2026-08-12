from __future__ import annotations

from src.pipeline import ChemExtractPipeline, PipelineConfig
from demo import SAMPLE_CHUNKS


def test_preloaded_pipeline_uses_production_chunk_timeout() -> None:
    """User PDFs must not inherit the short offline/demo timeout."""

    pipeline = ChemExtractPipeline(
        PipelineConfig(query="user PDF", preloaded_chunks=[])
    )

    assert pipeline._chunk_timeout_seconds() == 180.0

    pdf_pipeline = ChemExtractPipeline(
        PipelineConfig(query="user PDF", preloaded_chunks=[], chunk_timeout_seconds=180.0)
    )
    assert pdf_pipeline._chunk_timeout_seconds() == 180.0


async def _collect_chunk_events(pipeline: ChemExtractPipeline, chunk) -> list:
    """Collect one chunk's progress events for the domain gate test."""

    return [event async for event in pipeline._process_chunk(chunk, 1, 1)]


def test_non_chemistry_chunk_is_skipped_before_extraction() -> None:
    """Generic prose should not construct or invoke the extraction pipeline."""

    import asyncio

    pipeline = ChemExtractPipeline(PipelineConfig(query="test synthesis", preloaded_chunks=[]))
    events = asyncio.run(_collect_chunk_events(pipeline, SAMPLE_CHUNKS[2]))

    assert [event.stage for event in events] == ["skipped"]
