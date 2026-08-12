"""LangGraph extraction and validation pipeline for Phases 2 and 3.

This file contains ExtractionPipeline. Phase 2 performs extraction and Phase 3
adds deterministic validation plus bounded correction loops so failed recipes
are corrected or safely marked FAILED without crashing chunk processing.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from src.agents.condition_agent import ConditionExtractionAgent
from src.agents.entity_agent import EntityIdentificationAgent
from src.agents.quantity_agent import QuantityAlignmentAgent
from src.llm.base import BaseLLMProvider
from src.llm.resolver import get_llm_provider
from src.schemas.paper import TextChunk
from src.schemas.pipeline import PipelineState, make_initial_state
from src.schemas.recipe import ChemicalRecipe, CorrectionRecord, ValidationStatus
from src.storage.base import BaseGraphStore
from src.storage.resolver import get_graph_store
from src.validator.correction_prompts import (
    build_condition_correction_prompt,
    build_entity_correction_prompt,
    build_quantity_correction_prompt,
)
from src.validator.engine import ValidationEngine
from src.validator.error_types import ResponsibleAgent

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """ExtractionPipeline runs one TextChunk through the Phase 2 LangGraph graph."""

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        graph_store: BaseGraphStore | None = None,
        persist_recipes: bool = True,
    ) -> None:
        """Create agents and optional graph storage from abstract dependencies."""

        self._provider = provider or get_llm_provider()
        self._graph_store = graph_store
        self._owns_graph_store = graph_store is None
        self._persist_recipes = persist_recipes
        self._entity_agent = EntityIdentificationAgent(self._provider)
        self._quantity_agent = QuantityAlignmentAgent(self._provider)
        self._condition_agent = ConditionExtractionAgent(self._provider)
        self._graph = self._build_graph()

    async def extract(self, chunk: TextChunk) -> ChemicalRecipe:
        """Extract one ChemicalRecipe from a Phase 1 TextChunk."""

        initial_state = make_initial_state(
            source_text=chunk.text,
            chunk_id=chunk.chunk_id,
            paper_title=chunk.paper_metadata.title,
            paper_doi=chunk.paper_metadata.doi,
            paper_url=chunk.paper_metadata.open_access_url,
        )
        final_state = await self._graph.ainvoke(initial_state)
        recipe = final_state.get("recipe")
        if recipe is None:
            raise RuntimeError("Extraction graph completed without a recipe")
        self._persist_recipe(recipe)
        return recipe

    def close(self) -> None:
        """Close owned graph storage so Obsidian/Neo4j resources flush cleanly."""

        if self._owns_graph_store and self._graph_store is not None:
            self._graph_store.close()
            self._graph_store = None

    async def entity_node(self, state: PipelineState) -> dict:
        """Identify entities and attach node-level observability metadata."""

        source_text = state["source_text"]
        if _is_retry_for(state, ResponsibleAgent.ENTITY_AGENT):
            source_text = _with_correction_context(
                source_text,
                build_entity_correction_prompt(
                    source_text,
                    state["identified_entities"],
                    [],
                    state["retry_count"],
                    state["max_retries"],
                ),
            )
        result, meta = await self._entity_agent.extract(source_text)
        return {
            "identified_entities": result.entities,
            "total_tokens": state["total_tokens"] + int(meta["tokens"]),
            "total_latency": state["total_latency"] + float(meta["latency"]),
            "node_latencies": {**state["node_latencies"], "entity_node": float(meta["latency"])},
            "node_tokens": {**state["node_tokens"], "entity_node": int(meta["tokens"])},
            "llm_provider": str(meta["provider"]),
            "llm_model": str(meta["model"]),
        }

    async def quantity_node(self, state: PipelineState) -> dict:
        """Align quantities to the entities identified by the previous node."""

        source_text = state["source_text"]
        if _is_retry_for(state, ResponsibleAgent.QUANTITY_AGENT):
            source_text = _with_correction_context(
                source_text,
                build_quantity_correction_prompt(
                    source_text,
                    state["identified_entities"],
                    [],
                    state["retry_count"],
                    state["max_retries"],
                ),
            )
        result, meta = await self._quantity_agent.align(source_text, state["identified_entities"])
        return {
            "quantified_entities": result.entities,
            "total_tokens": state["total_tokens"] + int(meta["tokens"]),
            "total_latency": state["total_latency"] + float(meta["latency"]),
            "node_latencies": {**state["node_latencies"], "quantity_node": float(meta["latency"])},
            "node_tokens": {**state["node_tokens"], "quantity_node": int(meta["tokens"])},
            "llm_provider": str(meta["provider"]),
            "llm_model": str(meta["model"]),
        }

    async def condition_node(self, state: PipelineState) -> dict:
        """Extract conditions and assemble the final ChemicalRecipe."""

        source_text = state["source_text"]
        if _is_retry_for(state, ResponsibleAgent.CONDITION_AGENT):
            source_text = _with_correction_context(
                source_text,
                build_condition_correction_prompt(
                    source_text,
                    state["conditions"],
                    [],
                    state["retry_count"],
                    state["max_retries"],
                ),
            )
        result, meta = await self._condition_agent.extract(
            source_text,
            entities=_entity_list_from_state(state),
        )
        total_tokens = state["total_tokens"] + int(meta["tokens"])
        total_latency = state["total_latency"] + float(meta["latency"])
        node_latencies = {**state["node_latencies"], "condition_node": float(meta["latency"])}
        node_tokens = {**state["node_tokens"], "condition_node": int(meta["tokens"])}
        cost_provider = get_llm_provider(str(meta["provider"]), str(meta["model"]))
        estimated_cost = cost_provider.estimate_cost(total_tokens // 2, total_tokens // 2)
        recipe = ChemicalRecipe(
            source_chunk_id=state["source_chunk_id"],
            source_paper_title=state["source_paper_title"],
            source_paper_doi=state["source_paper_doi"],
            source_paper_url=state["source_paper_url"],
            title=_make_recipe_title(result.technique, state["quantified_entities"]),
            entities=state["quantified_entities"],
            conditions=result,
            correction_history=state["correction_history"],
            total_tokens_used=total_tokens,
            estimated_cost_usd=estimated_cost,
            total_latency_seconds=total_latency,
            node_latencies=node_latencies,
            node_tokens=node_tokens,
            llm_provider=str(meta["provider"]),
            llm_model=str(meta["model"]),
        )
        return {
            "conditions": result,
            "recipe": recipe,
            "total_tokens": total_tokens,
            "total_latency": total_latency,
            "node_latencies": node_latencies,
            "node_tokens": node_tokens,
            "llm_provider": str(meta["provider"]),
            "llm_model": str(meta["model"]),
        }

    async def validator_node(self, state: PipelineState) -> dict:
        """Run deterministic validation and choose the next graph route."""

        temp_recipe = _build_recipe_from_state(state, ValidationStatus.PENDING)
        result = ValidationEngine().validate(temp_recipe)
        if result.passed:
            return {"route_to": "assemble", "last_validation_errors": []}
        responsible = result.primary_responsible_agent()
        return {
            "route_to": responsible.value if responsible else "exhausted",
            "retry_count": state["retry_count"] + 1,
            "last_validation_errors": [error.message for error in result.blocking_errors],
        }

    async def error_router_node(self, state: PipelineState) -> dict:
        """Record correction attempts and stop retries after the configured limit."""

        if state["retry_count"] >= state["max_retries"]:
            return {"route_to": "exhausted"}
        record = CorrectionRecord(
            attempt_number=state["retry_count"],
            error_type=state["last_validation_errors"][0] if state["last_validation_errors"] else "UNKNOWN",
            error_field="see errors",
            error_message="; ".join(state["last_validation_errors"]),
            agent_that_fixed=state["route_to"] or "unknown",
        )
        return {"correction_history": state["correction_history"] + [record]}

    async def assemble_recipe_node(self, state: PipelineState) -> dict:
        """Assemble the accepted recipe and mark corrected recipes distinctly."""

        status = ValidationStatus.CORRECTED if state["correction_history"] else ValidationStatus.PASSED
        return {"recipe": _build_recipe_from_state(state, status)}

    async def failure_sink_node(self, state: PipelineState) -> dict:
        """Return a FAILED recipe after retries are exhausted without raising."""

        recipe = _build_recipe_from_state(state, ValidationStatus.FAILED)
        recipe = recipe.model_copy(update={"validation_errors": state["last_validation_errors"]})
        logger.warning(
            "Recipe %s failed after %s retries. Errors: %s",
            recipe.recipe_id,
            state["retry_count"],
            state["last_validation_errors"],
        )
        return {"recipe": recipe}

    def _build_graph(self):
        """Build and compile the extraction graph with Phase 3 validation loops."""

        graph = StateGraph(PipelineState)
        graph.add_node("entity_node", self.entity_node)
        graph.add_node("quantity_node", self.quantity_node)
        graph.add_node("condition_node", self.condition_node)
        graph.add_node("validator_node", self.validator_node)
        graph.add_node("error_router_node", self.error_router_node)
        graph.add_node("assemble_recipe_node", self.assemble_recipe_node)
        graph.add_node("failure_sink_node", self.failure_sink_node)
        graph.add_edge(START, "entity_node")
        graph.add_edge("entity_node", "quantity_node")
        graph.add_edge("quantity_node", "condition_node")
        graph.add_edge("condition_node", "validator_node")
        graph.add_conditional_edges(
            "validator_node",
            route_after_validation,
            {
                "assemble_recipe_node": "assemble_recipe_node",
                "error_router_node": "error_router_node",
            },
        )
        graph.add_conditional_edges(
            "error_router_node",
            route_after_error_router,
            {
                "entity_agent": "entity_node",
                "quantity_agent": "quantity_node",
                "condition_agent": "condition_node",
                "failure_sink_node": "failure_sink_node",
            },
        )
        graph.add_edge("assemble_recipe_node", END)
        graph.add_edge("failure_sink_node", END)
        return graph.compile()

    def _get_graph_store(self) -> BaseGraphStore:
        """Resolve and cache graph storage so repeated extracts write to one backend."""

        if self._graph_store is None:
            self._graph_store = get_graph_store()
        return self._graph_store

    def _persist_recipe(self, recipe: ChemicalRecipe) -> None:
        """Persist only validated recipes through the BaseGraphStore abstraction."""

        if not self._persist_recipes:
            return
        if recipe.validation_status not in {ValidationStatus.PASSED, ValidationStatus.CORRECTED}:
            logger.warning("Skipping graph persistence for %s recipe %s", recipe.validation_status.value, recipe.recipe_id)
            return
        store = self._get_graph_store()
        store.write_recipe_full(recipe)
        logger.info("Persisted recipe %s to %s graph store", recipe.recipe_id, store.backend_name)


def _entity_list_from_state(state: PipelineState):
    """Construct EntityList lazily to avoid importing it in node signatures."""

    from src.schemas.chemical import EntityList

    return EntityList(entities=state["quantified_entities"])


def _make_recipe_title(technique: str | None, entities: list) -> str | None:
    """Create a compact descriptive recipe title from extracted fields."""

    products = [entity.name for entity in entities if getattr(entity.role, "value", entity.role) == "PRODUCT"]
    if not products:
        return None
    if technique:
        return f"{technique.title()} synthesis of {products[0]}"
    return f"Synthesis of {products[0]}"


def route_after_validation(state: PipelineState) -> str:
    """Route accepted recipes to assembly and failures to the retry router."""

    if state["route_to"] == "assemble":
        return "assemble_recipe_node"
    return "error_router_node"


def route_after_error_router(state: PipelineState) -> str:
    """Route retryable failures to the responsible agent or to the failure sink."""

    if state["route_to"] == "exhausted":
        return "failure_sink_node"
    return state["route_to"] or "failure_sink_node"


def _build_recipe_from_state(state: PipelineState, status: ValidationStatus) -> ChemicalRecipe:
    """Build a recipe from state so validation, assembly, and failure agree."""

    try:
        cost_provider = get_llm_provider(state["llm_provider"] or None, state["llm_model"] or None)
        estimated_cost = cost_provider.estimate_cost(state["total_tokens"] // 2, state["total_tokens"] // 2)
    except ValueError:
        estimated_cost = 0.0
    return ChemicalRecipe(
        source_chunk_id=state["source_chunk_id"],
        source_paper_title=state["source_paper_title"],
        source_paper_doi=state["source_paper_doi"],
        source_paper_url=state["source_paper_url"],
        title=_make_recipe_title(state["conditions"].technique if state["conditions"] else None, state["quantified_entities"]),
        entities=state["quantified_entities"],
        conditions=state["conditions"],
        validation_status=status,
        correction_history=state["correction_history"],
        total_tokens_used=state["total_tokens"],
        estimated_cost_usd=estimated_cost,
        total_latency_seconds=state["total_latency"],
        node_latencies=state["node_latencies"],
        node_tokens=state["node_tokens"],
        llm_provider=state["llm_provider"] or "unknown",
        llm_model=state["llm_model"] or "unknown",
    )


def _is_retry_for(state: PipelineState, agent: ResponsibleAgent) -> bool:
    """Return whether a node is being revisited for a correction attempt."""

    return state["retry_count"] > 0 and bool(state["last_validation_errors"]) and state["route_to"] == agent.value


def _with_correction_context(source_text: str, correction_prompt: str) -> str:
    """Inject correction instructions while preserving the original source text."""

    return f"{source_text}\n\n{correction_prompt}"
