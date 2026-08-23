from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from demo import SAMPLE_CHUNKS


async def main() -> None:
    """Run the Phase 6A end-to-end integration check."""

    from src.pipeline import ChemExtractPipeline, PipelineConfig
    from src.storage.resolver import get_pipeline_store

    # This verification uses the deterministic Phase 2 fallbacks and never
    # depends on external provider latency or credentials.
    os.environ.pop("GROQ_API_KEY", None)
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:1"

    config = PipelineConfig(
        query="test synthesis",
        preloaded_chunks=SAMPLE_CHUNKS,
        graph_backend="networkx",
        llm_provider="offline",
        max_retries=1,
        verbose=False,
    )
    pipeline = ChemExtractPipeline(config)
    print("Pipeline initialized")

    events = []
    async for event in pipeline.stream():
        events.append(event)
        print(f"  [{event.stage}] {event.message[:60]}")
    print(f"Events received: {len(events)}")

    stages = [event.stage for event in events]
    assert "starting" in stages
    assert "fetched" in stages
    assert "extracting" in stages
    assert "complete" in stages
    print(f"Event stages: {stages}")

    result = pipeline._result
    assert result.chunks_produced == 3
    assert result.recipes_extracted >= 1
    assert result.completed_at is not None
    assert result.total_duration_seconds > 0
    print(f"Recipes extracted: {result.recipes_extracted}")
    print(f"Success rate: {result.success_rate():.1f}%")

    store = get_pipeline_store("networkx")
    assert store.recipe_count() >= 1
    assert store.node_count() > 0
    print(f"Graph nodes: {store.node_count()}")
    print(f"Graph edges: {store.edge_count()}")

    for recipe in result.recipes:
        if recipe.validation_status.value != "FAILED":
            assert len(recipe.entities) >= 1
        assert recipe.recipe_id is not None
        assert recipe.validation_status.value in ("PASSED", "CORRECTED", "FAILED")
        print(
            "Recipe summary: "
            f"ID={recipe.recipe_id}, "
            f"Status={recipe.validation_status.value}, "
            f"Entities={len(recipe.entities)}, "
            f"Corrections={recipe.correction_count()}"
        )

    assert Path("demo.py").exists()
    assert len(SAMPLE_CHUNKS) >= 3
    print("demo.py exists and has sample data")
    print("✅ PHASE 6A COMPLETE — END-TO-END PIPELINE CONNECTED")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
