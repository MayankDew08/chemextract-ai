"""Runtime pipeline settings exposed by the dashboard API."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class LLMProviderChoice(str, Enum):
    """Supported language-model providers."""

    OLLAMA = "ollama"
    GROQ = "groq"
    GEMINI = "gemini"


class GraphBackendChoice(str, Enum):
    """Supported graph storage backends."""

    NETWORKX = "networkx"
    OBSIDIAN = "obsidian"
    NEO4J = "neo4j"


class OllamaSettings(BaseModel):
    """Settings for a local Ollama server."""

    model_name: str = Field(default="qwen2.5:7b")
    base_url: str = Field(default="http://localhost:11434")


class GroqSettings(BaseModel):
    """Settings for the Groq provider."""

    api_key: str = Field(default="")
    model_name: str = Field(default="llama-3.1-8b-instant")


class GeminiSettings(BaseModel):
    """Settings for the Gemini provider."""

    api_key: str = Field(default="")
    model_name: str = Field(default="gemini-1.5-flash")


class Neo4jSettings(BaseModel):
    """Settings for a Neo4j Bolt connection."""

    uri: str = Field(default="bolt://localhost:7687")
    username: str = Field(default="neo4j")
    password: str = Field(default="chemextract123")


class ObsidianSettings(BaseModel):
    """Settings for the Obsidian-compatible vault."""

    vault_path: str = Field(default="./data/obsidian_vault")


class PipelineSettings(BaseModel):
    """Complete session-scoped settings controlled by the dashboard."""

    llm_provider: LLMProviderChoice = LLMProviderChoice.OLLAMA
    graph_backend: GraphBackendChoice = GraphBackendChoice.NETWORKX
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    obsidian: ObsidianSettings = Field(default_factory=ObsidianSettings)
    max_papers: int = Field(default=10, ge=1, le=25)
    max_retries: int = Field(default=3, ge=1, le=5)
    inter_chunk_delay: float = Field(default=1.0, ge=0.0, le=10.0)


class ConnectionStatus(BaseModel):
    """Result of testing one configured connection."""

    connected: bool
    message: str
    details: Optional[str] = None


class FullConnectionStatus(BaseModel):
    """Status of all connections displayed by the dashboard."""

    ollama: ConnectionStatus
    groq: ConnectionStatus
    gemini: ConnectionStatus
    neo4j: ConnectionStatus
    obsidian: ConnectionStatus


class OllamaModelsResponse(BaseModel):
    """Locally available Ollama model names."""

    models: list[str]
    recommended: str = "qwen2.5:7b"
