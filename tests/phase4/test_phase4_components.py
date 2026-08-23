"""Pytest coverage for Phase 4 graph storage components.

These tests keep automated verification under tests/ and avoid external Neo4j
requirements. They verify the shared BaseGraphStore contract using NetworkX and
Obsidian backends, including query parity and markdown persistence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.schemas.chemical import ChemicalEntity, ChemicalRole, Quantity
from src.schemas.recipe import ChemicalRecipe, CorrectionRecord, ReactionConditions, ValidationStatus
from src.storage.markdown_writer import generate_recipe_markdown, parse_recipe_from_markdown, sanitize_filename
from src.storage.neo4j_store import Neo4jStore
from src.storage.obsidian_store import ObsidianVaultStore
from src.storage.resolver import get_graph_store
from src.validator.error_types import ErrorSeverity, ErrorType, ResponsibleAgent, ValidationError


def sample_recipe(recipe_id: str = "rxn_test") -> ChemicalRecipe:
    """Build a validated ZnO recipe fixture for storage tests."""

    return ChemicalRecipe(
        recipe_id=recipe_id,
        source_chunk_id=recipe_id,
        source_paper_title="Storage test paper",
        source_paper_doi="10.0000/storage",
        source_paper_url="https://example.org/papers/storage-test",
        title="Reflux synthesis of ZnO",
        validation_status=ValidationStatus.PASSED,
        entities=[
            ChemicalEntity(name="zinc acetate", role=ChemicalRole.REACTANT, quantity=Quantity(value=2.0, unit="g")),
            ChemicalEntity(name="methanol", role=ChemicalRole.SOLVENT, quantity=Quantity(value=100.0, unit="mL")),
            ChemicalEntity(name="KOH", role=ChemicalRole.CATALYST, quantity=Quantity(value=1.0, unit="g")),
            ChemicalEntity(name="ZnO", role=ChemicalRole.PRODUCT),
        ],
        conditions=ReactionConditions(temperature_celsius=65.0, duration_hours=2.0, technique="reflux"),
    )


def recipe_with_audit_metadata(recipe_id: str = "rxn_audit") -> ChemicalRecipe:
    """Build a recipe carrying one warning and source/extraction provenance."""

    warning = ValidationError(
        error_type=ErrorType.NO_CONDITIONS,
        severity=ErrorSeverity.WARNING,
        responsible_agent=ResponsibleAgent.CONDITION_AGENT,
        field_path="conditions",
        message="No reaction conditions were found.",
        suggested_fix="Inspect the full source text.",
    )
    return ChemicalRecipe.model_validate(
        sample_recipe(recipe_id).model_dump()
        | {
            "validation_warnings": [warning],
            "source_text_completeness": "full_text",
            "source_acquisition_method": "user_upload",
            "node_extraction_methods": {"condition_agent": "deterministic_fallback"},
        }
    )


def test_networkx_store_writes_queries_and_exports_visjs() -> None:
    """NetworkXStore should persist one recipe idempotently and answer core queries."""

    store = get_graph_store("networkx")
    recipe = sample_recipe()
    store.write_recipe_full(recipe)
    store.write_recipe_full(recipe)

    solvents = store.query_solvents_for("ZnO")
    methanol_recipes = store.query_recipes_for_chemical("methanol")
    graph_json = store.export_graph_json()
    visjs = graph_json.to_visjs_dict()
    reaction = next(node for node in graph_json.nodes if node.node_type.value == "reaction")
    paper = next(node for node in graph_json.nodes if node.node_type.value == "paper")

    assert store.recipe_count() == 1
    assert solvents[0].solvent_name == "methanol"
    assert len(methanol_recipes) == 1
    assert visjs["nodes"] and all("id" in node and "label" in node for node in visjs["nodes"])
    assert visjs["edges"] and all("from" in edge and "to" in edge for edge in visjs["edges"])
    assert reaction.properties["source_url"] == "https://example.org/papers/storage-test"
    assert paper.properties["source_url"] == "https://example.org/papers/storage-test"
    store.close()


def test_networkx_store_exposes_warning_details_and_provenance() -> None:
    """Reaction graph nodes should expose durable audit metadata without flattening it."""

    store = get_graph_store("networkx")
    recipe = recipe_with_audit_metadata()

    store.write_recipe_full(recipe)
    reaction = next(node for node in store.export_graph_json().nodes if node.node_type.value == "reaction")

    assert store.get_recipe(recipe.recipe_id) is recipe
    assert reaction.properties["warning_count"] == 1
    assert reaction.properties["validation_warnings"][0]["error_type"] == "NO_CONDITIONS"
    assert reaction.properties["source_text_completeness"] == "full_text"
    assert reaction.properties["source_acquisition_method"] == "user_upload"
    assert reaction.properties["node_extraction_methods"] == {"condition_agent": "deterministic_fallback"}
    store.close()


def test_source_url_alone_creates_paper_provenance_node() -> None:
    """A URL-only source must still create a linked Paper node."""

    store = get_graph_store("networkx")
    recipe = sample_recipe().model_copy(
        update={
            "source_paper_title": None,
            "source_paper_doi": None,
            "source_paper_url": "https://example.org/papers/url-only",
        }
    )

    store.write_recipe_full(recipe)
    graph = store.export_graph_json()

    paper = next(node for node in graph.nodes if node.node_type.value == "paper")
    assert paper.properties["source_url"] == "https://example.org/papers/url-only"
    assert any(edge.relationship_type.value == "SOURCED_FROM" for edge in graph.edges)
    store.close()


def test_obsidian_store_creates_markdown_and_rebuilds_queries() -> None:
    """ObsidianVaultStore should persist markdown and rebuild NetworkX on restart."""

    with tempfile.TemporaryDirectory() as temp_dir:
        store = ObsidianVaultStore(vault_path=temp_dir)
        store.initialize()
        recipe = sample_recipe()
        store.write_recipe_full(recipe)
        assert store.recipe_count() == 1
        reaction_files = list((store._vault_path / "reactions").glob("*.md"))
        paper_files = list((store._vault_path / "papers").glob("*.md"))
        assert len(reaction_files) == 1
        assert len(paper_files) == 1
        assert "[Open original paper](https://example.org/papers/storage-test)" in reaction_files[0].read_text(encoding="utf-8")
        assert "[Open original paper](https://example.org/papers/storage-test)" in paper_files[0].read_text(encoding="utf-8")
        assert (store._vault_path / "chemicals" / "ZnO.md").exists()
        store.close()

        rebuilt = ObsidianVaultStore(vault_path=temp_dir)
        rebuilt.initialize()
        assert rebuilt.recipe_count() == 1
        assert rebuilt.query_solvents_for("ZnO")
        assert rebuilt.get_recipe(recipe.recipe_id).source_paper_url == "https://example.org/papers/storage-test"
        rebuilt.close()


def test_obsidian_persists_warning_details_and_provenance() -> None:
    """Embedded recipe JSON and readable sections should survive a vault rebuild."""

    with tempfile.TemporaryDirectory() as temp_dir:
        store = ObsidianVaultStore(vault_path=temp_dir)
        store.initialize()
        recipe = recipe_with_audit_metadata()
        store.write_recipe_full(recipe)
        reaction_file = next((store._vault_path / "reactions").glob("*.md"))
        markdown = reaction_file.read_text(encoding="utf-8")
        store.close()

        assert "warning_count: 1" in markdown
        assert "## Validation Warnings" in markdown
        assert "NO_CONDITIONS" in markdown
        assert "## Extraction Provenance" in markdown
        assert "full_text" in markdown
        assert "user_upload" in markdown
        assert "| Atmosphere | unknown |" in markdown

        rebuilt = ObsidianVaultStore(vault_path=temp_dir)
        rebuilt.initialize()
        restored = rebuilt.get_recipe(recipe.recipe_id)
        assert len(restored.validation_warnings) == 1
        assert restored.source_text_completeness.value == "full_text"
        assert restored.source_acquisition_method.value == "user_upload"
        assert restored.node_extraction_methods["condition_agent"].value == "deterministic_fallback"
        rebuilt.close()


def test_obsidian_audit_records_truthfully_describe_routing() -> None:
    """Human-readable correction history must not claim an unverified fix."""

    correction = CorrectionRecord(
        attempt_number=1,
        error_type="UNIT_MISSING",
        error_field="entities[0].quantity",
        error_message="Reactant quantity was missing.",
        agent_routed_to="quantity_agent",
    )
    recipe = sample_recipe().model_copy(update={"correction_history": [correction]})

    markdown = generate_recipe_markdown(recipe)

    assert "routed to quantity_agent" in markdown
    assert "quantity_agent fixed" not in markdown


def test_obsidian_rebuild_accepts_legacy_agent_that_fixed_payload(tmp_path: Path) -> None:
    """Existing embedded recipe JSON should load through the renamed audit field."""

    correction = CorrectionRecord(
        attempt_number=1,
        error_type="UNIT_MISSING",
        error_field="entities[0].quantity",
        error_message="Reactant quantity was missing.",
        agent_routed_to="quantity_agent",
    )
    recipe = sample_recipe("legacy-audit").model_copy(update={"correction_history": [correction]})
    legacy_payload = recipe.model_dump_json().replace('"agent_routed_to"', '"agent_that_fixed"')
    reactions_dir = tmp_path / "reactions"
    (tmp_path / "chemicals").mkdir()
    reactions_dir.mkdir()
    (tmp_path / "papers").mkdir()
    (reactions_dir / "legacy-audit.md").write_text(
        f"```chemextract-recipe-json\n{legacy_payload}\n```\n",
        encoding="utf-8",
    )

    store = ObsidianVaultStore(vault_path=str(tmp_path))
    store.initialize()
    restored = store.get_recipe("legacy-audit")

    assert restored is not None
    assert restored.correction_history[0].agent_routed_to == "quantity_agent"
    assert "agent_that_fixed" not in restored.model_dump_json()
    store.close()


def test_legacy_markdown_parser_does_not_invent_pressure(tmp_path: Path) -> None:
    """A missing legacy pressure value must remain absent after a vault rebuild."""

    markdown_path = tmp_path / "legacy.md"
    markdown_path.write_text(
        """---
recipe_id: legacy
validation_status: PASSED
---

# Legacy recipe

## Reaction Conditions
| Parameter | Value |
|---|---|
| Temperature | 25.0°C |
| Duration | unknown |
| Technique | unknown |
| Atmosphere | unknown |
| Pressure | unknown |
""",
        encoding="utf-8",
    )

    restored = parse_recipe_from_markdown(markdown_path)

    assert restored is not None
    assert restored.conditions is not None
    assert restored.conditions.pressure_atm is None
    assert restored.conditions.atmosphere is None


def test_resolver_uses_graph_backend_env(monkeypatch) -> None:
    """GRAPH_BACKEND should switch backend through the resolver only."""

    monkeypatch.setenv("GRAPH_BACKEND", "networkx")
    store = get_graph_store()

    assert store.backend_name == "networkx"
    store.close()


def test_sanitize_filename_handles_complex_chemical_name() -> None:
    """sanitize_filename should remove unsafe punctuation while preserving chemistry text."""

    assert sanitize_filename("Zinc acetate dihydrate (Zn(CH3COO)2·2H2O)") == "Zinc_acetate_dihydrate_Zn_CH3COO_2·2H2O"


def test_obsidian_source_link_rejects_non_http_protocols() -> None:
    """Untrusted provenance must not become an executable Markdown link."""

    recipe = sample_recipe().model_copy(update={"source_paper_url": "javascript:alert(1)"})

    markdown = generate_recipe_markdown(recipe)

    assert "[Open original paper](javascript:" not in markdown
    assert "URL: invalid source URL" in markdown


def test_obsidian_store_rejects_recipe_id_path_traversal() -> None:
    """Untrusted recipe IDs must not escape the reactions directory."""

    with tempfile.TemporaryDirectory() as temp_dir:
        store = ObsidianVaultStore(vault_path=temp_dir)
        store.initialize()
        unsafe = sample_recipe().model_copy(update={"recipe_id": "../escaped"})
        with pytest.raises(ValueError, match="Unsafe recipe ID"):
            store.write_recipe(unsafe)
        assert not (Path(temp_dir) / "escaped.md").exists()
        store.close()


def test_neo4j_store_writes_audit_metadata_to_reaction_and_paper_nodes() -> None:
    """Neo4j Cypher writes must retain source provenance and validation audit data."""

    calls: list[tuple[str, dict]] = []

    class RecordingSession:
        """Record Cypher calls while implementing the driver's context protocol."""

        def __enter__(self):
            """Return this recording session."""

            return self

        def __exit__(self, *_args):
            """Leave the fake session without suppressing errors."""

            return False

        def run(self, query: str, parameters: dict | None = None):
            """Capture one parameterized Cypher statement."""

            calls.append((query, parameters or {}))
            return None

    class RecordingDriver:
        """Return recording sessions for Neo4jStore writes."""

        def session(self):
            """Create a recording session."""

            return RecordingSession()

    store = Neo4jStore()
    store._driver = RecordingDriver()
    store._nx_mirror.initialize()
    recipe = recipe_with_audit_metadata()
    store.write_recipe_full(recipe)

    reaction_query, reaction_params = next(item for item in calls if "MERGE (r:ChemExtract:" in item[0])
    paper_params = next(params for query, params in calls if "MERGE (p:ChemExtract:" in query)
    assert reaction_params["source_url"] == "https://example.org/papers/storage-test"
    assert paper_params["source_url"] == "https://example.org/papers/storage-test"
    assert "r.warning_count = $warning_count" in reaction_query
    assert reaction_params["warning_count"] == 1
    assert '"error_type": "NO_CONDITIONS"' in reaction_params["validation_warnings"]
    assert reaction_params["source_text_completeness"] == "full_text"
    assert reaction_params["source_acquisition_method"] == "user_upload"
    assert '"condition_agent": "deterministic_fallback"' in reaction_params["node_extraction_methods"]
    assert ChemicalRecipe.model_validate_json(reaction_params["recipe_json"]) == recipe


def test_neo4j_store_rehydrates_persisted_recipes_on_initialize(monkeypatch) -> None:
    """A fresh Neo4jStore process should expose recipes already persisted in Neo4j."""

    persisted = recipe_with_audit_metadata("rxn_persisted")

    class HydrationResult:
        """Return persisted recipe payloads through the Neo4j result iterator."""

        def __init__(self, session):
            self._session = session

        def __iter__(self):
            if not self._session.active:
                raise RuntimeError("Neo4j results must be consumed before the session closes")
            return iter([{"recipe_json": persisted.model_dump_json()}])

    class HydrationSession:
        """Serve constraints and the startup hydration query."""

        active = False

        def __enter__(self):
            self.active = True
            return self

        def __exit__(self, *_args):
            self.active = False
            return False

        def run(self, query: str, parameters: dict | None = None):
            if "RETURN r.recipe_json AS recipe_json" in query:
                return HydrationResult(self)
            return None

    class HydrationDriver:
        """Minimal Neo4j driver used at the public initialize seam."""

        def verify_connectivity(self):
            return None

        def session(self):
            return HydrationSession()

        def close(self):
            return None

    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *_args, **_kwargs: HydrationDriver())
    store = Neo4jStore()

    store.initialize()

    assert store.get_recipe(persisted.recipe_id) == persisted
    assert store.get_all_recipes() == [persisted]
    assert store.export_graph_json().metadata["recipe_count"] == 1
    store.close()


@pytest.mark.asyncio
async def test_extraction_pipeline_persists_validated_recipe_to_graph_store() -> None:
    """ExtractionPipeline should write PASSED recipes to the configured graph store."""

    from src.agents.pipeline import ExtractionPipeline
    from src.schemas.paper import AcquisitionMethod, PaperMetadata, TextChunk, TextCompleteness

    class StaticPipeline(ExtractionPipeline):
        """Pipeline test double that bypasses LLM extraction but uses persistence code."""

        def __init__(self, graph_store):
            """Keep only the persistence dependencies needed for this integration seam."""

            self._graph_store = graph_store
            self._owns_graph_store = False
            self._persist_recipes = True

        async def extract(self, chunk):
            """Return a valid recipe and invoke the production persistence method."""

            recipe = sample_recipe("rxn_pipeline")
            self._persist_recipe(recipe)
            return recipe

    store = get_graph_store("networkx")
    metadata = PaperMetadata(title="Pipeline paper", source_db="test")
    chunk = TextChunk(
        chunk_id="pipeline-chunk",
        paper_metadata=metadata,
        text="Zinc acetate was dissolved in methanol to produce ZnO at 65°C for 2 hours.",
        completeness=TextCompleteness.ABSTRACT,
        acquisition_method=AcquisitionMethod.ABSTRACT_ONLY,
        chemical_density_score=1.0,
        word_count=12,
    )

    recipe = await StaticPipeline(store).extract(chunk)

    assert recipe.validation_status == ValidationStatus.PASSED
    assert store.recipe_count() == 1
    assert store.query_solvents_for("ZnO")
    store.close()
