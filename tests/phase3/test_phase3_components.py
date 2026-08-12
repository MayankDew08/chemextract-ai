"""Pytest coverage for Phase 3 validation and routing.

These tests keep automated verification under tests/ and avoid live LLM calls.
They exercise the deterministic validator, blocking/warning distinction, routing
priority, and the extended pipeline's failure-safe sink behavior.
"""

from __future__ import annotations

import pytest

from src.agents.pipeline import route_after_error_router, route_after_validation
from src.schemas.chemical import ChemicalEntity, ChemicalRole, Quantity
from src.schemas.pipeline import make_initial_state
from src.schemas.recipe import ChemicalRecipe, ReactionConditions, ValidationStatus
from src.validator.engine import ValidationEngine
from src.validator.error_types import ErrorType, ResponsibleAgent


def valid_recipe() -> ChemicalRecipe:
    """Build a valid recipe fixture so each test changes one behavior."""

    return ChemicalRecipe(
        source_chunk_id="test",
        entities=[
            ChemicalEntity(
                name="Zinc acetate dihydrate",
                role=ChemicalRole.REACTANT,
                quantity=Quantity(value=2.195, unit="g"),
                moles=Quantity(value=10, unit="mmol"),
            ),
            ChemicalEntity(name="Methanol", role=ChemicalRole.SOLVENT, quantity=Quantity(value=100, unit="mL")),
            ChemicalEntity(
                name="KOH",
                role=ChemicalRole.CATALYST,
                quantity=Quantity(value=1.12, unit="g"),
                moles=Quantity(value=20, unit="mmol"),
            ),
            ChemicalEntity(name="ZnO", role=ChemicalRole.PRODUCT),
        ],
        conditions=ReactionConditions(temperature_celsius=65.0, duration_hours=2.0, technique="reflux"),
    )


def test_validation_engine_registers_all_rules_and_passes_valid_recipe() -> None:
    """The pure validator should run all 15 rules and accept a valid recipe."""

    engine = ValidationEngine()

    result = engine.validate(valid_recipe())

    assert engine.rule_count == 15
    assert result.passed
    assert result.rules_checked == 15
    assert not result.has_blocking_errors()


def test_blocking_errors_route_to_primary_responsible_agent() -> None:
    """Entity failures should outrank quantity and condition failures."""

    recipe = valid_recipe().model_copy(update={"entities": []})

    result = ValidationEngine().validate(recipe)

    assert not result.passed
    assert result.blocking_errors[0].error_type == ErrorType.NO_ENTITIES
    assert result.primary_responsible_agent() == ResponsibleAgent.ENTITY_AGENT


def test_warnings_do_not_fail_recipe() -> None:
    """Warnings should be recorded without blocking trusted recipes."""

    recipe = valid_recipe()
    product = recipe.entities[-1].model_copy(update={"quantity": Quantity(value=1.0, unit="g")})
    recipe = recipe.model_copy(update={"entities": [*recipe.entities[:-1], product]})

    result = ValidationEngine().validate(recipe)

    assert result.passed
    assert result.warnings
    assert result.warnings[0].error_type == ErrorType.PRODUCT_QUANTITY_AS_INPUT


def test_route_helpers_send_pass_fail_and_exhausted_states_correctly() -> None:
    """LangGraph route helpers should keep conditional edges deterministic."""

    state = make_initial_state("text", "chunk")
    state["route_to"] = "assemble"
    assert route_after_validation(state) == "assemble_recipe_node"

    state["route_to"] = "quantity_agent"
    assert route_after_validation(state) == "error_router_node"
    assert route_after_error_router(state) == "quantity_agent"

    state["route_to"] = "exhausted"
    assert route_after_error_router(state) == "failure_sink_node"


@pytest.mark.asyncio
async def test_failure_sink_returns_failed_recipe_without_raising() -> None:
    """The failure sink must make failed recipes explicit instead of crashing."""

    from src.agents.pipeline import ExtractionPipeline

    state = make_initial_state("text", "chunk")
    state["retry_count"] = 3
    state["last_validation_errors"] = ["No entities"]
    pipeline = ExtractionPipeline.__new__(ExtractionPipeline)

    result = await ExtractionPipeline.failure_sink_node(pipeline, state)

    recipe = result["recipe"]
    assert recipe.validation_status == ValidationStatus.FAILED
    assert recipe.validation_errors == ["No entities"]
