"""Integration coverage for validation state carried through graph nodes."""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel

from src.agents.condition_agent import ConditionExtractionAgent
from src.agents.pipeline import ExtractionPipeline
from src.llm.base import BaseLLMProvider
from src.schemas.chemical import ChemicalEntity, ChemicalRole, EntityList, Quantity
from src.schemas.paper import AcquisitionMethod, TextCompleteness
from src.schemas.pipeline import make_initial_state
from src.schemas.recipe import ExtractionMethod, ReactionConditions, ValidationStatus


class FailingProvider(BaseLLMProvider):
    """Force an extraction agent onto its deterministic fallback."""

    @property
    def provider_name(self) -> str:
        return "failing"

    @property
    def model_name(self) -> str:
        return "failing-model"

    def get_llm(self) -> BaseChatModel:
        raise ValueError("forced provider failure")

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


def _valid_entities() -> list[ChemicalEntity]:
    return [
        ChemicalEntity(
            name="Zinc acetate",
            role=ChemicalRole.REACTANT,
            quantity=Quantity(value=2.0, unit="g"),
        ),
        ChemicalEntity(name="Methanol", role=ChemicalRole.SOLVENT),
        ChemicalEntity(name="ZnO", role=ChemicalRole.PRODUCT),
    ]


@pytest.mark.asyncio
async def test_warnings_and_provenance_survive_validation_and_assembly() -> None:
    """A passed recipe retains the latest warning and provenance state."""

    state = make_initial_state(
        "A sufficiently detailed source procedure for graph state testing.",
        "warning-provenance",
        text_completeness=TextCompleteness.FULL_TEXT,
        acquisition_method=AcquisitionMethod.USER_UPLOAD,
    )
    state.update(
        quantified_entities=_valid_entities(),
        conditions=ReactionConditions(temperature_celsius=65.0, duration_hours=2.0),
        llm_provider="failing",
        llm_model="failing-model",
        node_extraction_methods={
            "entity_node": ExtractionMethod.DETERMINISTIC_FALLBACK,
            "quantity_node": ExtractionMethod.DETERMINISTIC_FALLBACK,
            "condition_node": ExtractionMethod.DETERMINISTIC_FALLBACK,
        },
    )
    pipeline = ExtractionPipeline.__new__(ExtractionPipeline)

    state.update(await pipeline.validator_node(state))
    result = await pipeline.assemble_recipe_node(state)
    recipe = result["recipe"]

    assert recipe.validation_status == ValidationStatus.PASSED
    assert [warning.entity_name for warning in recipe.validation_warnings] == ["Methanol"]
    assert recipe.source_text_completeness == TextCompleteness.FULL_TEXT
    assert recipe.source_acquisition_method == AcquisitionMethod.USER_UPLOAD
    assert set(recipe.node_extraction_methods.values()) == {ExtractionMethod.DETERMINISTIC_FALLBACK}


@pytest.mark.asyncio
async def test_missing_conditions_are_not_fabricated_by_fallback() -> None:
    """Absent pressure and atmosphere remain null in deterministic output."""

    text = "Zinc acetate was stirred for 2 hours to produce ZnO nanoparticles."

    conditions, metadata = await ConditionExtractionAgent(FailingProvider()).extract(text, EntityList())

    assert conditions.pressure_atm is None
    assert conditions.atmosphere is None
    assert metadata["extraction_method"] == ExtractionMethod.DETERMINISTIC_FALLBACK.value


@pytest.mark.asyncio
async def test_retry_audit_groups_every_error_of_the_routed_type() -> None:
    """One retry reports every missing quantity without claiming it was fixed."""

    state = make_initial_state("Incomplete quantity source text for testing.", "grouped-errors")
    state.update(
        quantified_entities=[
            ChemicalEntity(name="Zinc acetate", role=ChemicalRole.REACTANT),
            ChemicalEntity(name="KOH", role=ChemicalRole.CATALYST),
            ChemicalEntity(name="ZnO", role=ChemicalRole.PRODUCT),
        ],
        conditions=ReactionConditions(temperature_celsius=65.0, duration_hours=2.0),
        llm_provider="failing",
        llm_model="failing-model",
    )
    pipeline = ExtractionPipeline.__new__(ExtractionPipeline)

    state.update(await pipeline.validator_node(state))
    state.update(await pipeline.error_router_node(state))

    first_attempt = state["correction_history"][0]
    assert first_attempt.error_type == "UNIT_MISSING"
    assert first_attempt.agent_routed_to == "quantity_agent"
    assert "Zinc acetate" in first_attempt.error_message
    assert "KOH" in first_attempt.error_message
    assert "agent_that_fixed" not in first_attempt.model_dump()
