"""Extraction job API routes wired to the Phase 6A pipeline."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket
from pydantic import BaseModel, Field

from src.pipeline import ProgressEvent

logger = logging.getLogger(__name__)

router = APIRouter()
jobs: dict[str, "JobStatus"] = {}
job_queues: dict[str, asyncio.Queue[ProgressEvent]] = {}


class ExtractionRequest(BaseModel):
    """ExtractionRequest validates user-triggered extraction jobs."""

    query: str = Field(..., min_length=3)
    max_papers: int = Field(default=10, ge=1, le=25)
    graph_backend: Optional[str] = None


class JobStatus(BaseModel):
    """JobStatus tracks async extraction progress for polling and fallback clients."""

    job_id: str
    status: str
    query: str
    chunks_total: int = 0
    chunks_done: int = 0
    recipes_extracted: int = 0
    recipes_passed: int = 0
    error: Optional[str] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: Optional[datetime] = None


@router.post("/extract")
async def start_extraction(request: ExtractionRequest) -> dict[str, str]:
    """Queue a full ChemExtract pipeline job and return WebSocket connection data."""

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = JobStatus(job_id=job_id, status="queued", query=request.query)
    job_queues[job_id] = asyncio.Queue()
    asyncio.create_task(run_pipeline_task(job_id, request))
    return {
        "job_id": job_id,
        "status": "queued",
        "ws_url": f"ws://localhost:8000/api/ws/{job_id}",
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JobStatus:
    """Return the current status for one extraction job."""

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return jobs[job_id]


@router.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str) -> None:
    """Stream real-time pipeline progress events to one browser client."""

    await websocket.accept()
    if job_id not in job_queues:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return

    queue = job_queues[job_id]
    try:
        while True:
            event = await asyncio.wait_for(queue.get(), timeout=120.0)
            await websocket.send_json(event.model_dump(mode="json"))
            if event.stage in ("complete", "error"):
                break
    except asyncio.TimeoutError:
        await websocket.send_json({"stage": "error", "message": "Pipeline timed out after 120 seconds"})
    finally:
        await websocket.close()
        asyncio.create_task(cleanup_job(job_id, delay=300))


async def cleanup_job(job_id: str, delay: int) -> None:
    """Remove job data after a delay so transient state does not grow forever."""

    await asyncio.sleep(delay)
    jobs.pop(job_id, None)
    job_queues.pop(job_id, None)


async def run_pipeline_task(job_id: str, request: ExtractionRequest) -> None:
    """Run the full pipeline in the background and publish progress events."""

    from src.pipeline import ChemExtractPipeline, PipelineConfig

    jobs[job_id].status = "running"
    queue = job_queues.get(job_id)
    config = PipelineConfig(
        query=request.query,
        max_papers=request.max_papers,
        graph_backend=request.graph_backend,
        verbose=True,
    )
    pipeline = ChemExtractPipeline(config)

    try:
        async for event in pipeline.stream():
            if event.stage == "fetched":
                jobs[job_id].chunks_total = pipeline._result.chunks_produced
            elif event.stage == "stored":
                jobs[job_id].chunks_done += 1
                jobs[job_id].recipes_extracted = pipeline._result.recipes_extracted
                jobs[job_id].recipes_passed = pipeline._result.passed_first_try + pipeline._result.self_corrected
            elif event.stage == "complete":
                jobs[job_id].status = "complete"
                jobs[job_id].completed_at = datetime.now(UTC)
            elif event.stage == "error":
                jobs[job_id].status = "failed"
                jobs[job_id].error = event.message
                jobs[job_id].completed_at = datetime.now(UTC)

            if queue is not None:
                await queue.put(event)
    except Exception as exc:
        logger.error("Pipeline task %s failed: %s", job_id, exc, exc_info=True)
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(exc)
        jobs[job_id].completed_at = datetime.now(UTC)
        error_event = ProgressEvent(stage="error", message=str(exc))
        if queue is not None:
            await queue.put(error_event)
