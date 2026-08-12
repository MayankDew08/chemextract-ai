"""Standalone Phase 4 graph storage demonstration.

This demo writes five validated recipes into NetworkX and Obsidian backends,
compares query results, verifies vis.js export shape, and checks Obsidian
markdown persistence/rebuild behavior. It lives under demo/, not tests/.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.schemas.chemical import ChemicalEntity, ChemicalRole, Quantity
from src.schemas.recipe import ChemicalRecipe, ReactionConditions, ValidationStatus
from src.storage.resolver import get_graph_store


def print_header(text: str, width: int = 72) -> None:
    """Print a section heading for demo readability."""

    print("\n" + "═" * width)
    print(text.center(width))
    print("═" * width)


def print_check(passed: bool, message: str) -> None:
    """Print a completion checklist item."""

    print(f"{'✅' if passed else '❌'} {message}")


def recipes() -> list[ChemicalRecipe]:
    """Create the five required handcrafted validated recipes."""

    return [
        _recipe("rxn_zno_1", "ZnO", "zinc acetate", "methanol", "KOH", "reflux", 65, 2),
        _recipe("rxn_zno_2", "ZnO", "zinc nitrate", "ethanol", "NaOH", "hydrothermal", 120, 12),
        _recipe("rxn_zno_3", "ZnO", "zinc chloride", "water", "ammonia", "stirring", 25, 4),
        _recipe("rxn_tio2_1", "TiO2", "titanium isopropoxide", "ethanol", "HCl", "sol-gel", 80, 3),
        _recipe("rxn_tio2_2", "TiO2", "TiCl4", "water", "HNO3", "hydrothermal", 150, 8),
    ]


def _recipe(
    recipe_id: str,
    product: str,
    reactant: str,
    solvent: str,
    catalyst: str,
    technique: str,
    temperature: float,
    duration: float,
) -> ChemicalRecipe:
    """Build one validated recipe with consistent storage metadata."""

    return ChemicalRecipe(
        recipe_id=recipe_id,
        source_chunk_id=recipe_id,
        source_paper_title=f"Paper for {recipe_id}",
        source_paper_doi=f"10.0000/{recipe_id}",
        title=f"{technique.title()} synthesis of {product}",
        validation_status=ValidationStatus.PASSED,
        entities=[
            ChemicalEntity(name=reactant, role=ChemicalRole.REACTANT, quantity=Quantity(value=1.0, unit="g")),
            ChemicalEntity(name=solvent, role=ChemicalRole.SOLVENT, quantity=Quantity(value=100.0, unit="mL")),
            ChemicalEntity(name=catalyst, role=ChemicalRole.CATALYST, quantity=Quantity(value=1.0, unit="g")),
            ChemicalEntity(name=product, role=ChemicalRole.PRODUCT),
        ],
        conditions=ReactionConditions(temperature_celsius=temperature, duration_hours=duration, technique=technique),
    )


def run_backend(backend: str, vault_path: Path | None = None) -> dict[str, bool | list[str]]:
    """Run the required storage sequence for one backend."""

    print_header(f"Testing backend: {backend}")
    previous_vault = os.environ.get("OBSIDIAN_VAULT_PATH")
    if vault_path:
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault_path)
    store = get_graph_store(backend)
    for recipe in recipes():
        store.write_recipe_full(recipe)
        print(f"Written: {recipe.recipe_id} ({len(recipe.entities)} entities)")
    print(f"Node count: {store.node_count()}")
    print(f"Edge count: {store.edge_count()}")
    print(f"Recipe count: {store.recipe_count()}")

    solvents = store.query_solvents_for("ZnO")
    print("Solvents for ZnO synthesis (3 recipes):")
    for index, solvent in enumerate(solvents, start=1):
        print(f"{index}. {solvent.solvent_name:<10} -> {solvent.frequency} reactions ({solvent.percentage}%)")

    methanol_recipes = store.query_recipes_for_chemical("methanol")
    print(f"Recipes containing methanol: {len(methanol_recipes)}")
    cooccurring = store.query_cooccurring_chemicals("ZnO")
    print("Top co-occurring chemicals with ZnO:")
    for item in cooccurring[:3]:
        print(f"- {item.chemical_name}: {item.co_occurrence_count}")

    graph_json = store.export_graph_json()
    visjs = graph_json.to_visjs_dict()
    print(f"Graph exported: Nodes={len(graph_json.nodes)} Edges={len(graph_json.edges)}")

    checks = {
        "recipes_written": store.recipe_count() == 5,
        "solvent_query": len(solvents) == 3,
        "methanol_query": len(methanol_recipes) == 1,
        "cooccurrence_query": len(cooccurring) > 0,
        "graph_json": bool(graph_json.nodes) and bool(graph_json.edges),
        "visjs": "nodes" in visjs
        and "edges" in visjs
        and all("id" in node and "label" in node for node in visjs["nodes"])
        and all("from" in edge and "to" in edge for edge in visjs["edges"]),
    }

    obsidian_checks = {}
    if backend == "obsidian" and vault_path:
        reaction_files = list((vault_path / "reactions").glob("*.md"))
        chemical_files = list((vault_path / "chemicals").glob("*.md"))
        zno_path = vault_path / "chemicals" / "ZnO.md"
        obsidian_checks = {
            "markdown_files": len(reaction_files) == 5 and bool(chemical_files),
            "zno_wikilinks": zno_path.exists() and "[[rxn_" in zno_path.read_text(encoding="utf-8"),
            "frontmatter": all(path.read_text(encoding="utf-8").startswith("---") for path in reaction_files),
        }
        if reaction_files:
            print("\nSample reaction markdown:")
            print(reaction_files[0].read_text(encoding="utf-8")[:800])

    store.write_recipe_full(recipes()[0])
    checks["idempotent"] = store.recipe_count() == 5
    print(f"Idempotency: {'PASS' if checks['idempotent'] else 'FAIL'}")
    store.close()
    print("Store closed cleanly")

    if previous_vault is not None:
        os.environ["OBSIDIAN_VAULT_PATH"] = previous_vault
    elif "OBSIDIAN_VAULT_PATH" in os.environ:
        del os.environ["OBSIDIAN_VAULT_PATH"]
    return {**checks, **obsidian_checks, "solvents": [item.solvent_name for item in solvents]}


def main() -> None:
    """Run Phase 4 backend checks and print the completion checklist."""

    print_header("ChemExtract AI Phase 4 Demo")
    vault_path = PROJECT_ROOT / "data" / "obsidian_vault_demo"
    if vault_path.exists():
        shutil.rmtree(vault_path)

    networkx_results = run_backend("networkx")
    obsidian_results = run_backend("obsidian", vault_path)

    os.environ["GRAPH_BACKEND"] = "networkx"
    resolver_networkx = get_graph_store()
    resolver_ok = resolver_networkx.backend_name == "networkx"
    resolver_networkx.close()
    os.environ["GRAPH_BACKEND"] = "obsidian"
    os.environ["OBSIDIAN_VAULT_PATH"] = str(vault_path)
    resolver_obsidian = get_graph_store()
    env_switch_ok = resolver_obsidian.backend_name == "obsidian"
    rebuild_ok = resolver_obsidian.recipe_count() == 5
    resolver_obsidian.close()

    same_queries = networkx_results["solvents"] == obsidian_results["solvents"]
    checks = [
        (networkx_results["recipes_written"], "NetworkXStore: all 5 recipes written"),
        (networkx_results["solvent_query"], "NetworkXStore: solvent query returns 3 results"),
        (networkx_results["cooccurrence_query"], "NetworkXStore: co-occurrence query works"),
        (networkx_results["graph_json"] and networkx_results["visjs"], "NetworkXStore: graph JSON valid for vis.js"),
        (networkx_results["idempotent"], "NetworkXStore: idempotent writes confirmed"),
        (obsidian_results["recipes_written"], "ObsidianVaultStore: all 5 recipes written"),
        (obsidian_results["markdown_files"], "ObsidianVaultStore: markdown files created"),
        (obsidian_results["zno_wikilinks"], "ObsidianVaultStore: ZnO.md has WikiLinks"),
        (rebuild_ok, "ObsidianVaultStore: vault rebuild works"),
        (same_queries, "ObsidianVaultStore: same query results as NetworkX"),
        (resolver_ok, "Backend resolver works (networkx / obsidian)"),
        (env_switch_ok, "GRAPH_BACKEND env var switches backends"),
    ]

    print_header("Completion Checklist")
    for passed, message in checks:
        print_check(bool(passed), message)
    if all(passed for passed, _message in checks):
        print("\n✅ PHASE 4 COMPLETE — PROCEED TO PHASE 5")
    else:
        print("\n❌ PHASE 4 INCOMPLETE")
        sys.exit(1)


if __name__ == "__main__":
    main()
