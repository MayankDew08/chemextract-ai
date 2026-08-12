"""Runtime settings and connection diagnostics for the dashboard."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request

from src.api.models.settings import (
    ConnectionStatus,
    FullConnectionStatus,
    GeminiSettings,
    GraphBackendChoice,
    OllamaModelsResponse,
    PipelineSettings,
)
from src.storage.base import BaseGraphStore

settings_router = APIRouter(prefix="/settings")


def _masked_settings(settings: PipelineSettings) -> dict[str, Any]:
    """Return JSON-safe settings without exposing configured API keys."""

    data = settings.model_dump(mode="json")
    for provider in ("groq", "gemini"):
        if data[provider]["api_key"]:
            data[provider]["api_key"] = "***"
    if data["neo4j"]["password"]:
        data["neo4j"]["password"] = "***"
    return data


def _merge_settings(current: PipelineSettings, update: dict[str, Any]) -> PipelineSettings:
    """Merge a partial JSON update into current validated settings."""

    merged = current.model_dump()
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    for provider in ("groq", "gemini"):
        if update.get(provider, {}).get("api_key") == "***":
            merged[provider]["api_key"] = current.model_dump()[provider]["api_key"]
    if update.get("neo4j", {}).get("password") == "***":
        merged["neo4j"]["password"] = current.neo4j.password
    return PipelineSettings.model_validate(merged)


def _create_store_from_settings(settings: PipelineSettings) -> BaseGraphStore:
    """Create and initialize the graph store selected in session settings."""

    if settings.graph_backend == GraphBackendChoice.NEO4J:
        from src.storage.neo4j_store import Neo4jStore

        store = Neo4jStore(settings.neo4j.uri, settings.neo4j.username, settings.neo4j.password)
    elif settings.graph_backend == GraphBackendChoice.OBSIDIAN:
        from src.storage.obsidian_store import ObsidianVaultStore

        store = ObsidianVaultStore(settings.obsidian.vault_path)
    else:
        from src.storage.networkx_store import NetworkXStore

        store = NetworkXStore()
    store.initialize()
    return store


@settings_router.get("")
async def get_settings(request: Request) -> dict[str, Any]:
    """Return current settings with secrets masked."""

    return _masked_settings(request.app.state.settings)


@settings_router.post("")
async def update_settings(request: Request) -> dict[str, Any]:
    """Validate and apply a partial settings update for the current session."""

    try:
        update = await request.json()
        if not isinstance(update, dict):
            raise ValueError("settings body must be a JSON object")
        current = request.app.state.settings
        settings = _merge_settings(current, update)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid settings: {exc}") from exc

    old_backend = current.graph_backend
    request.app.state.settings = settings
    os.environ.update(
        {
            "LLM_PROVIDER": settings.llm_provider.value,
            "OLLAMA_MODEL": settings.ollama.model_name,
            "OLLAMA_BASE_URL": settings.ollama.base_url,
            "GROQ_MODEL": settings.groq.model_name,
            "GEMINI_MODEL": settings.gemini.model_name,
            "GRAPH_BACKEND": settings.graph_backend.value,
            "OBSIDIAN_VAULT_PATH": settings.obsidian.vault_path,
        }
    )
    if settings.groq.api_key:
        os.environ["GROQ_API_KEY"] = settings.groq.api_key
    if settings.gemini.api_key:
        os.environ["GEMINI_API_KEY"] = settings.gemini.api_key

    backend_configuration_changed = (
        old_backend != settings.graph_backend
        or (settings.graph_backend == GraphBackendChoice.NEO4J and current.neo4j != settings.neo4j)
        or (settings.graph_backend == GraphBackendChoice.OBSIDIAN and current.obsidian != settings.obsidian)
    )
    if backend_configuration_changed:
        old_store = request.app.state.graph_store
        new_store = _create_store_from_settings(settings)
        request.app.state.graph_store = new_store
        from src.storage.resolver import set_pipeline_store

        set_pipeline_store(new_store)
        old_store.close()
    return _masked_settings(settings)


async def _request_status(url: str, **kwargs: Any) -> httpx.Response:
    """Make a short-lived HTTP request for connection diagnostics."""

    async with httpx.AsyncClient() as client:
        return await client.get(url, timeout=kwargs.pop("timeout", 5.0), **kwargs)


async def _test_ollama(settings: PipelineSettings) -> ConnectionStatus:
    """Test the configured Ollama endpoint."""

    try:
        response = await _request_status(f"{settings.ollama.base_url.rstrip('/')}/api/tags", timeout=3.0)
        if response.status_code == 200:
            models = [item["name"] for item in response.json().get("models", [])]
            return ConnectionStatus(connected=True, message=f"Connected — {len(models)} models available", details=", ".join(models[:3]))
        return ConnectionStatus(connected=False, message=f"Ollama returned HTTP {response.status_code}")
    except Exception:
        return ConnectionStatus(connected=False, message="Ollama server not running", details="Start with: ollama serve")


async def _test_cloud(url: str, credential: str, valid_message: str, missing_message: str, **params: str) -> ConnectionStatus:
    """Test a cloud provider API key without logging or returning the key."""

    if not credential:
        return ConnectionStatus(connected=False, message=missing_message)
    try:
        response = await _request_status(url, headers={"Authorization": f"Bearer {credential}"} if "groq" in url else {}, params=params, timeout=5.0)
        if response.status_code == 200:
            return ConnectionStatus(connected=True, message=valid_message)
        return ConnectionStatus(connected=False, message=f"Invalid API key (HTTP {response.status_code})")
    except Exception as exc:
        return ConnectionStatus(connected=False, message=f"Connection failed: {str(exc)[:50]}")


async def _test_neo4j(settings: PipelineSettings) -> ConnectionStatus:
    """Test Neo4j connectivity while gracefully handling an optional dependency."""

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(settings.neo4j.uri, auth=(settings.neo4j.username, settings.neo4j.password))
        try:
            driver.verify_connectivity()
        finally:
            driver.close()
        return ConnectionStatus(connected=True, message="Neo4j connected", details=f"URI: {settings.neo4j.uri}")
    except ImportError:
        return ConnectionStatus(connected=False, message="neo4j package not installed", details="pip install neo4j")
    except Exception:
        return ConnectionStatus(connected=False, message="Neo4j not reachable", details="Run: docker compose up -d neo4j")


def _test_obsidian(settings: PipelineSettings) -> ConnectionStatus:
    """Ensure the configured Obsidian vault directory exists."""

    path = Path(settings.obsidian.vault_path).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        return ConnectionStatus(connected=True, message="Vault exists", details=str(path))
    except Exception as exc:
        return ConnectionStatus(connected=False, message=f"Cannot create vault: {exc}")


@settings_router.get("/status", response_model=FullConnectionStatus)
async def connection_status(request: Request) -> FullConnectionStatus:
    """Test all configured connections concurrently and return their statuses."""

    settings = request.app.state.settings
    ollama, groq, gemini, neo4j = await asyncio.gather(
        _test_ollama(settings),
        _test_cloud("https://api.groq.com/openai/v1/models", settings.groq.api_key or os.getenv("GROQ_API_KEY", ""), "Groq API key valid", "No API key provided"),
        _test_cloud("https://generativelanguage.googleapis.com/v1beta/models", settings.gemini.api_key or os.getenv("GEMINI_API_KEY", ""), "Gemini API key valid", "No API key provided", key=settings.gemini.api_key or os.getenv("GEMINI_API_KEY", "")),
        _test_neo4j(settings),
    )
    return FullConnectionStatus(ollama=ollama, groq=groq, gemini=gemini, neo4j=neo4j, obsidian=_test_obsidian(settings))


@settings_router.get("/ollama-models", response_model=OllamaModelsResponse)
async def ollama_models(request: Request) -> OllamaModelsResponse:
    """Return models currently advertised by the configured Ollama server."""

    settings = request.app.state.settings
    try:
        response = await _request_status(f"{settings.ollama.base_url.rstrip('/')}/api/tags", timeout=3.0)
        if response.status_code == 200:
            return OllamaModelsResponse(models=[item["name"] for item in response.json().get("models", [])])
    except Exception:
        pass
    return OllamaModelsResponse(models=[])
