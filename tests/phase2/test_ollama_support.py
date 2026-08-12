from __future__ import annotations

import json

from src.agents.json_utils import extract_json_from_text, parse_to_pydantic, repair_json
from src.llm.ollama_provider import OllamaProvider
from src.schemas.chemical import EntityList


def test_json_utils_repairs_local_model_output() -> None:
    """Markdown-wrapped JSON with trailing commas should parse cleanly."""

    raw = "```json\n{'entities': [],}\n```"
    assert extract_json_from_text(raw) == "{'entities': [],}"
    assert json.loads(repair_json(extract_json_from_text(raw))) == {"entities": []}
    assert isinstance(parse_to_pydantic(raw, EntityList, label="test"), EntityList)


def test_ollama_provider_exposes_local_configuration() -> None:
    """Provider settings should be available without contacting Ollama."""

    provider = OllamaProvider(model_name="qwen2.5:7b", base_url="http://localhost:11434")

    assert provider.provider_name == "ollama"
    assert provider.model_name == "qwen2.5:7b"
    assert provider.estimate_cost(100, 50) == 0.0
