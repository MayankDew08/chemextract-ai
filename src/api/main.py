"""FastAPI application entrypoint for ChemExtract AI Phase 5.

This file creates the local web app, initializes graph and metrics stores during
lifespan startup, mounts static assets, and registers API plus HTML routers.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.routes.extract import router as extract_router
from src.api.routes.graph import router as graph_router
from src.api.routes.metrics import router as metrics_router
from src.api.routes.recipes import router as recipes_router
from src.api.routes.upload import upload_router
from src.api.routes.views import router as views_router
from src.api.routes.settings import settings_router, _create_store_from_settings
from src.api.models.settings import (
    GeminiSettings,
    GraphBackendChoice,
    GroqSettings,
    LLMProviderChoice,
    Neo4jSettings,
    OllamaSettings,
    ObsidianSettings,
    PipelineSettings,
)
from src.observability.metrics_store import MetricsStore
from src.storage.resolver import get_graph_store, set_pipeline_metrics_store, set_pipeline_store

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and close graph and metrics stores exactly once per app lifecycle."""

    initial_settings = PipelineSettings(
        llm_provider=LLMProviderChoice(os.getenv("LLM_PROVIDER", "ollama")),
        graph_backend=GraphBackendChoice(os.getenv("GRAPH_BACKEND", "networkx")),
        ollama=OllamaSettings(
            model_name=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ),
        groq=GroqSettings(
            api_key=os.getenv("GROQ_API_KEY", ""),
            model_name=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        ),
        gemini=GeminiSettings(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        ),
        neo4j=Neo4jSettings(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "chemextract123"),
        ),
        obsidian=ObsidianSettings(vault_path=os.getenv("OBSIDIAN_VAULT_PATH", "./data/obsidian_vault")),
    )
    app.state.settings = initial_settings
    store = _create_store_from_settings(initial_settings)
    metrics = MetricsStore()
    await metrics.initialize()
    set_pipeline_store(store)
    set_pipeline_metrics_store(metrics)
    app.state.graph_store = store
    app.state.metrics_store = metrics
    logger.info("ChemExtract AI ready at http://localhost:8000")
    yield
    store.close()
    await metrics.close()
    logger.info("ChemExtract AI shut down cleanly")


app = FastAPI(
    title="ChemExtract AI",
    description="Local-first chemistry knowledge graph extraction engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.state.templates = templates

app.include_router(extract_router, prefix="/api")
app.include_router(recipes_router, prefix="/api")
app.include_router(graph_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(settings_router, prefix="/api")
app.include_router(views_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a lightweight service health response."""

    return {"status": "ok", "service": "ChemExtract AI"}
