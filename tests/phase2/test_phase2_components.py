"""Pytest coverage for Phase 2 extraction components.

These tests stay in the pytest surface and avoid live LLM calls. Runnable demos
belong under demo/, while this file verifies schemas, resolver behavior, and
deterministic fallback behavior at public component seams.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel

from src.agents.condition_agent import ConditionExtractionAgent
from src.agents.entity_agent import EntityIdentificationAgent
from src.agents.quantity_agent import QuantityAlignmentAgent
from src.agents.pipeline import ExtractionPipeline
from src.llm.base import BaseLLMProvider
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.resolver import get_llm_provider
from src.schemas.chemical import ChemicalRole, Quantity
from src.schemas.pipeline import make_initial_state
from src.schemas.paper import AcquisitionMethod, PaperMetadata, TextChunk, TextCompleteness
from src.schemas.recipe import ChemicalRecipe, CorrectionRecord, ExtractionMethod, ReactionConditions
from src.schemas.validation import (
    ErrorSeverity,
    ErrorType,
    ResponsibleAgent,
    ValidationError,
)


class BrokenProvider(BaseLLMProvider):
    """Provider test double that forces agents onto deterministic fallback paths."""

    @property
    def provider_name(self) -> str:
        """Return a stable fake provider name for observability assertions."""

        return "broken"

    @property
    def model_name(self) -> str:
        """Return a stable fake model name for observability assertions."""

        return "broken-model"

    def get_llm(self) -> BaseChatModel:
        """Raise deliberately so agents exercise fallback extraction."""

        raise ValueError("forced provider failure")

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return zero because this provider never performs an LLM call."""

        return 0.0


class SchemaRejectedProvider(BrokenProvider):
    """Provider double that reproduces an empty structured tool response."""

    def get_llm(self) -> BaseChatModel:
        """Raise the provider error returned for an empty EntityList tool call."""

        raise ValueError("tool call validation failed: /entities minimum 1 items")


def test_quantity_schema_owns_shape_but_not_chemistry_semantics() -> None:
    """Quantity parsing should leave unit vocabulary and positivity to validation."""

    assert str(Quantity(value=2.195, unit="g")) == "2.195 g"
    assert Quantity(value=-1.0, unit="bananas").model_dump() == {"value": -1.0, "unit": "bananas"}
    with pytest.raises(ValueError):
        Quantity(value=1.0, unit=" ")
    with pytest.raises(ValueError):
        Quantity(value=float("nan"), unit="g")


def test_reaction_conditions_preserve_absent_values_and_defer_ranges() -> None:
    """Missing conditions stay absent and outliers remain available to the validator."""

    conditions = ReactionConditions(
        temperature_celsius=4000.0,
        duration_hours=-1.0,
        pressure_atm=0.0,
        atmosphere=None,
        yield_percent=101.0,
    )

    assert conditions.pressure_atm == 0.0
    assert conditions.atmosphere is None
    assert ReactionConditions().pressure_atm is None
    assert ReactionConditions().atmosphere is None
    with pytest.raises(ValueError):
        ReactionConditions(temperature_celsius=float("inf"))


def test_recipe_carries_typed_warnings_and_extraction_provenance() -> None:
    """The durable recipe should expose warnings and provenance without invented defaults."""

    warning = ValidationError(
        error_type=ErrorType.NO_CONDITIONS,
        severity=ErrorSeverity.WARNING,
        responsible_agent=ResponsibleAgent.CONDITION_AGENT,
        field_path="conditions",
        message="No conditions found.",
        suggested_fix="Inspect the source.",
    )
    recipe = ChemicalRecipe(
        source_chunk_id="chunk",
        source_text_completeness=TextCompleteness.FULL_TEXT,
        source_acquisition_method=AcquisitionMethod.USER_UPLOAD,
        node_extraction_methods={"entity_agent": ExtractionMethod.DETERMINISTIC_FALLBACK},
        validation_warnings=[warning],
    )

    assert recipe.validation_warnings == [warning]
    assert recipe.source_text_completeness == TextCompleteness.FULL_TEXT
    assert recipe.source_acquisition_method == AcquisitionMethod.USER_UPLOAD
    assert recipe.node_extraction_methods == {"entity_agent": ExtractionMethod.DETERMINISTIC_FALLBACK}


def test_correction_record_uses_truthful_name_and_accepts_legacy_payloads() -> None:
    """Audit records should say who was routed to while still loading old data."""

    legacy = CorrectionRecord(
        attempt_number=1,
        error_type="UNIT_MISSING",
        error_field="entities[0].quantity",
        error_message="Missing quantity",
        agent_that_fixed="quantity_agent",
    )

    assert legacy.agent_routed_to == "quantity_agent"
    assert legacy.agent_that_fixed == "quantity_agent"
    assert "agent_routed_to" in legacy.model_dump()
    assert "agent_that_fixed" not in legacy.model_dump()


def test_entity_list_accepts_empty_provider_response() -> None:
    """The validator, not tool-schema parsing, owns the no-entity failure."""

    from src.schemas.chemical import EntityList

    assert EntityList().entities == []


def test_reaction_conditions_normalize_common_provider_outputs() -> None:
    """ReactionConditions should normalize atmosphere and null additional fields."""

    conditions = ReactionConditions(atmosphere="N2", additional_conditions=None)

    assert conditions.atmosphere == "nitrogen"
    assert conditions.additional_conditions == {}


def test_make_initial_state_preserves_textchunk_metadata() -> None:
    """Initial LangGraph state should retain source identity and empty node outputs."""

    state = make_initial_state(
        "sample text",
        "chunk-1",
        "Paper title",
        "10.1/demo",
        paper_url="https://example.org/papers/source",
    )

    assert state["source_text"] == "sample text"
    assert state["source_chunk_id"] == "chunk-1"
    assert state["source_paper_title"] == "Paper title"
    assert state["source_paper_url"] == "https://example.org/papers/source"
    assert state["identified_entities"] == []
    assert state["recipe"] is None


def test_llm_resolver_selects_provider_without_constructing_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolver should switch provider classes through env vars only."""

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    gemini = get_llm_provider()
    groq = get_llm_provider("groq", "llama-3.1-8b-instant")

    assert isinstance(gemini, GeminiProvider)
    assert gemini.model_name == "gemini-2.5-flash"
    assert isinstance(groq, GroqProvider)
    assert groq.model_name == "llama-3.1-8b-instant"


@pytest.mark.asyncio
async def test_agents_have_deterministic_fallbacks_when_provider_fails() -> None:
    """Agents should still extract the demo chemistry when the LLM provider fails."""

    text = (
        "Zinc acetate dihydrate (2.195 g, 10 mmol) was dissolved in "
        "100 mL of methanol. The solution was refluxed at 65°C for "
        "2 hours with 1.12 g of KOH (20 mmol) as base catalyst. "
        "The ZnO precipitate was filtered and dried at 120°C."
    )
    provider = BrokenProvider()

    entity_list, _ = await EntityIdentificationAgent(provider).extract(text)
    quantified, _ = await QuantityAlignmentAgent(provider).align(text, entity_list.entities)
    conditions, _ = await ConditionExtractionAgent(provider).extract(text, quantified)

    roles = {entity.role for entity in quantified.entities}
    assert ChemicalRole.REACTANT in roles
    assert ChemicalRole.SOLVENT in roles
    assert ChemicalRole.CATALYST in roles
    assert ChemicalRole.PRODUCT in roles
    assert any(entity.quantity and entity.quantity.unit == "g" for entity in quantified.entities)
    assert conditions.temperature_celsius == 65.0
    assert conditions.duration_hours == 2.0
    assert conditions.technique == "reflux"


@pytest.mark.asyncio
async def test_extraction_pipeline_preserves_paper_url() -> None:
    """The recipe handed to storage must retain its input chunk's exact paper URL."""

    source_url = "https://example.org/papers/zno-synthesis?download=html"
    text = (
        "Zinc acetate dihydrate (2.195 g, 10 mmol) was dissolved in 100 mL methanol. "
        "The solution was refluxed at 65°C for 2 hours with 1.12 g KOH, and the resulting "
        "ZnO precipitate was filtered, washed, and dried at 120°C."
    )
    chunk = TextChunk(
        chunk_id="url-provenance",
        paper_metadata=PaperMetadata(
            title="ZnO synthesis paper",
            source_db="url_fetch",
            open_access=True,
            open_access_url=source_url,
        ),
        text=text,
        completeness=TextCompleteness.FULL_TEXT,
        acquisition_method=AcquisitionMethod.USER_UPLOAD,
        chemical_density_score=1.0,
        word_count=len(text.split()),
    )
    captured_state = {}

    class RecordingGraph:
        """Stand in for LangGraph while recording its public input state."""

        async def ainvoke(self, state: dict) -> dict:
            """Return a minimal recipe built from the received provenance."""

            captured_state.update(state)
            return {
                "recipe": ChemicalRecipe(
                    source_chunk_id=state["source_chunk_id"],
                    source_paper_url=state["source_paper_url"],
                )
            }

    pipeline = ExtractionPipeline.__new__(ExtractionPipeline)
    pipeline._graph = RecordingGraph()
    pipeline._persist_recipes = False
    pipeline._graph_store = None

    recipe = await pipeline.extract(chunk)

    assert captured_state["source_paper_url"] == source_url
    assert recipe.source_paper_url == source_url


@pytest.mark.asyncio
async def test_entity_schema_rejection_uses_local_fallback_without_second_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty structured entity output should not trigger another slow LLM call."""

    text = "Zinc acetate was dissolved in methanol and ZnO precipitate formed."
    monkeypatch.setattr(
        "src.agents.entity_agent.get_fallback_provider",
        lambda _provider: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )
    entity_list, _ = await EntityIdentificationAgent(SchemaRejectedProvider()).extract(text)

    assert entity_list.entities
    assert {entity.name.lower() for entity in entity_list.entities} >= {
        "zinc acetate",
        "methanol",
        "zno",
    }
