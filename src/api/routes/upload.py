"""PDF upload and URL fetch routes for Phase 6B inputs."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, field_validator

from src.api.routes.extract import JobStatus, job_queues, jobs
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.url_fetcher import URLFetcher
from src.pipeline import ProgressEvent
from src.schemas.paper import TextChunk

logger = logging.getLogger(__name__)

upload_router = APIRouter()
UPLOAD_DIR = Path(__file__).resolve().parents[3] / "data" / "uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class PDFUploadResponse(BaseModel):
    """Response returned after a PDF is parsed and queued for extraction."""

    job_id: str
    filename: str
    pages: int
    chunks_found: int
    sections_found: list[str]
    status: str


class URLFetchRequest(BaseModel):
    """Request body for fetching one user-provided paper URL."""

    url: str = Field(..., description="URL to fetch")
    title: Optional[str] = Field(None, description="Optional title hint")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Reject non-HTTP URLs at the API boundary."""

        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("URL must be a valid http:// or https:// URL")
        return value


class URLFetchResponse(BaseModel):
    """Response returned after URL fetching or paywall detection."""

    job_id: Optional[str] = None
    url: str
    chunks_found: int
    sections_found: list[str]
    is_paywalled: bool = False
    error: Optional[str] = None
    alternatives: list[str] = Field(default_factory=list)
    status: str


@upload_router.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    graph_backend: Optional[str] = Form(None),
) -> PDFUploadResponse:
    """Accept a PDF upload, parse it into chunks, and queue extraction."""

    filename = Path(file.filename or "upload.pdf").name
    if not filename.lower().endswith(".pdf") or file.content_type not in {None, "application/pdf"}:
        raise HTTPException(status_code=422, detail="Uploaded file must be a PDF")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}_{filename}"
    total_bytes = 0
    try:
        with saved_path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="PDF uploads are limited to 50MB",
                    )
                handle.write(chunk)

        parser = PDFParser()
        parsed = await parser.process(str(saved_path), title_hint=title)
        if not parsed.success:
            raise HTTPException(status_code=422, detail=parsed.error or "Could not parse PDF")
        if not parsed.chunks:
            raise HTTPException(status_code=422, detail="No chemistry content found in PDF")

        job_id = _create_job(f"uploaded_pdf_{filename}")
        asyncio.create_task(run_pipeline_from_chunks(job_id, parsed.chunks, graph_backend))
        return PDFUploadResponse(
            job_id=job_id,
            filename=filename,
            pages=parsed.total_pages or 0,
            chunks_found=len(parsed.chunks),
            sections_found=parsed.sections_found,
            status="queued",
        )
    finally:
        try:
            saved_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to delete uploaded PDF %s: %s", saved_path, exc)
        await file.close()


@upload_router.post("/fetch-url", response_model=URLFetchResponse)
async def fetch_url(request: URLFetchRequest) -> URLFetchResponse:
    """Fetch an open paper URL, parse it into chunks, and queue extraction."""

    fetcher = URLFetcher()
    try:
        parsed = await fetcher.process(request.url, title_hint=request.title)
    finally:
        await fetcher.close()

    if parsed.is_paywalled:
        return URLFetchResponse(
            url=request.url,
            chunks_found=0,
            sections_found=[],
            is_paywalled=True,
            error=parsed.error,
            alternatives=parsed.open_access_alternatives,
            status="paywalled",
        )
    if not parsed.success:
        raise HTTPException(status_code=422, detail=parsed.error or "Could not fetch URL")
    if not parsed.chunks:
        raise HTTPException(status_code=422, detail="No chemistry content found at this URL")

    job_id = _create_job(request.url)
    asyncio.create_task(run_pipeline_from_chunks(job_id, parsed.chunks, None))
    return URLFetchResponse(
        job_id=job_id,
        url=request.url,
        chunks_found=len(parsed.chunks),
        sections_found=parsed.sections_found,
        status="queued",
    )


async def run_pipeline_from_chunks(
    job_id: str,
    chunks: list[TextChunk],
    graph_backend: Optional[str],
) -> None:
    """Run the Phase 6A pipeline from already-acquired chunks."""

    from src.pipeline import ChemExtractPipeline, PipelineConfig

    jobs[job_id].status = "running"
    queue = job_queues.get(job_id)
    pipeline = ChemExtractPipeline(
        PipelineConfig(
            query=f"preloaded_input_{job_id}",
            preloaded_chunks=chunks,
            max_chunks_per_paper=10,
            graph_backend=graph_backend,
            chunk_timeout_seconds=180.0,
            verbose=True,
        )
    )
    try:
        async for event in pipeline.stream():
            _update_job_from_event(job_id, event, pipeline)
            if queue is not None:
                await queue.put(event)
    except Exception as exc:
        logger.error("Preloaded pipeline task %s failed: %s", job_id, exc, exc_info=True)
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(exc)
        jobs[job_id].completed_at = datetime.now(UTC)
        if queue is not None:
            await queue.put(ProgressEvent(stage="error", message=str(exc)))


def _create_job(query: str) -> str:
    """Create shared Phase 6A job state and event queue."""

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = JobStatus(job_id=job_id, status="queued", query=query)
    job_queues[job_id] = asyncio.Queue()
    return job_id


def _update_job_from_event(job_id: str, event: ProgressEvent, pipeline) -> None:
    """Mirror pipeline progress into JobStatus for polling fallbacks."""

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
