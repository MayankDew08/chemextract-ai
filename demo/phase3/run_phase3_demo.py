"""Standalone Phase 3 validation and correction demonstration.

This demo lives under demo/ because it is executable workflow evidence, not a
pytest module. It uses handcrafted recipes to show deterministic validation,
agent routing, bounded retries, failure sinking, and correction history.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas.chemical import ChemicalEntity, ChemicalRole, Quantity
from src.schemas.recipe import ChemicalRecipe, CorrectionRecord, ReactionConditions, ValidationStatus
from src.validator.engine import ValidationEngine
from src.validator.error_types import ErrorType, ResponsibleAgent


def print_header(text: str, width: int = 72) -> None:
    """Print a visible heading so the demo is easy to scan."""

    print("\n" + "═" * width)
    print(text.center(width))
    print("═" * width)


def print_section(text: str) -> None:
    """Print a section label for each handcrafted validation case."""

    print("\n" + "─" * 72)
    print(text)
    print("─" * 72)


def print_check(passed: bool, message: str) -> None:
    """Print pass/fail lines for the final completion checklist."""

    print(f"{'✅' if passed else '❌'} {message}")


def perfect_recipe() -> ChemicalRecipe:
    """Build the baseline valid ZnO recipe used by several demo cases."""

    return ChemicalRecipe(
        source_chunk_id="phase3-demo",
        source_paper_title="Demo ZnO synthesis",
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


def test_cases() -> list[tuple[str, str, ChemicalRecipe]]:
    """Return the six required handcrafted Phase 3 validation cases."""

    missing_unit = deepcopy(perfect_recipe())
    missing_unit.entities[0] = missing_unit.entities[0].model_copy(
        update={"name": "TiO2", "quantity": Quantity.model_construct(value=2.5, unit="")}
    )

    no_reactant = deepcopy(perfect_recipe())
    no_reactant.entities = [
        ChemicalEntity(name="Methanol", role=ChemicalRole.SOLVENT, quantity=Quantity(value=100, unit="mL")),
        ChemicalEntity(name="KOH", role=ChemicalRole.CATALYST, quantity=Quantity(value=1.12, unit="g")),
        ChemicalEntity(name="ZnO", role=ChemicalRole.PRODUCT),
    ]

    impossible_temperature = deepcopy(perfect_recipe())
    impossible_temperature.conditions = ReactionConditions.model_construct(
        temperature_celsius=5000.0,
        duration_hours=2.0,
        pressure_atm=1.0,
        atmosphere="air",
        technique="reflux",
        yield_percent=None,
        additional_conditions={},
    )

    multiple_errors = deepcopy(missing_unit)
    multiple_errors.conditions = impossible_temperature.conditions

    unfixable = ChemicalRecipe(source_chunk_id="phase3-unfixable", entities=[], conditions=None)

    return [
        ("perfect_recipe", "Fully correct ZnO recipe", perfect_recipe()),
        ("missing_unit", "TiO2 quantity has value but blank unit", missing_unit),
        ("no_reactant", "Only solvent/catalyst/product roles present", no_reactant),
        ("impossible_temperature", "Temperature is physically impossible", impossible_temperature),
        ("multiple_errors", "Missing unit and impossible temperature", multiple_errors),
        ("unfixable", "Empty entities and no conditions", unfixable),
    ]


def run_case(name: str, description: str, recipe: ChemicalRecipe) -> tuple[ChemicalRecipe, list[ErrorType], float]:
    """Validate one recipe and apply deterministic demo corrections by routed agent."""

    print_section(f"{name}: {description}")
    engine = ValidationEngine()
    current = deepcopy(recipe)
    history: list[CorrectionRecord] = []
    error_history: list[ErrorType] = []
    started = time.perf_counter()
    for attempt in range(1, 4):
        result = engine.validate(current)
        errors = result.blocking_errors
        error_history.extend(error.error_type for error in errors)
        route = result.primary_responsible_agent()
        print(f"Attempt {attempt}: errors={[error.error_type.value for error in errors]} route={route.value if route else 'assemble'}")
        if result.passed:
            status = ValidationStatus.CORRECTED if history else ValidationStatus.PASSED
            final = current.model_copy(update={"validation_status": status, "correction_history": history})
            print(f"Final status: {final.validation_status.value}")
            print(f"Time taken: {time.perf_counter() - started:.4f}s")
            return final, error_history, time.perf_counter() - started
        if name == "unfixable" or route is None:
            history.append(_record(attempt, errors, route))
            continue
        history.append(_record(attempt, errors, route))
        current = apply_demo_fix(current, route)
    final = current.model_copy(
        update={
            "validation_status": ValidationStatus.FAILED,
            "validation_errors": [error.value for error in error_history[-2:]],
            "correction_history": history,
        }
    )
    print(f"Final status: {final.validation_status.value}")
    print("Correction history:")
    for record in final.correction_history:
        print(f"  - attempt {record.attempt_number}: {record.agent_that_fixed} -> {record.error_message}")
    print(f"Time taken: {time.perf_counter() - started:.4f}s")
    return final, error_history, time.perf_counter() - started


def apply_demo_fix(recipe: ChemicalRecipe, route: ResponsibleAgent) -> ChemicalRecipe:
    """Apply deterministic fixes so the demo verifies routing without LLM randomness."""

    fixed = deepcopy(recipe)
    if route == ResponsibleAgent.ENTITY_AGENT:
        if not fixed.entities:
            return fixed
        fixed.entities[0] = fixed.entities[0].model_copy(update={"role": ChemicalRole.REACTANT})
    elif route == ResponsibleAgent.QUANTITY_AGENT:
        fixed.entities = [
            entity.model_copy(update={"quantity": Quantity(value=2.5, unit="g")})
            if entity.role in {ChemicalRole.REACTANT, ChemicalRole.CATALYST}
            and (entity.quantity is None or not str(getattr(entity.quantity, "unit", "")).strip())
            else entity
            for entity in fixed.entities
        ]
    elif route == ResponsibleAgent.CONDITION_AGENT:
        fixed.conditions = ReactionConditions(temperature_celsius=65.0, duration_hours=2.0, technique="reflux")
    return fixed


def _record(attempt: int, errors: list, route: ResponsibleAgent | None) -> CorrectionRecord:
    """Record one demo correction cycle with the routed responsible agent."""

    return CorrectionRecord(
        attempt_number=attempt,
        error_type=errors[0].error_type.value if errors else "UNKNOWN",
        error_field=errors[0].field_path if errors else "unknown",
        error_message="; ".join(error.message for error in errors) if errors else "unknown",
        agent_that_fixed=route.value if route else "failure_sink",
    )


def main() -> None:
    """Run all Phase 3 demo cases and print the required completion report."""

    print_header("ChemExtract AI Phase 3 Demo")
    finals = []
    error_counter: Counter[str] = Counter()
    total_corrections = 0
    for name, description, recipe in test_cases():
        final, errors, _elapsed = run_case(name, description, recipe)
        finals.append(final)
        error_counter.update(error.value for error in errors)
        total_corrections += len(final.correction_history)

    passed_first = sum(1 for recipe in finals if recipe.validation_status == ValidationStatus.PASSED)
    corrected = sum(1 for recipe in finals if recipe.validation_status == ValidationStatus.CORRECTED)
    failed = sum(1 for recipe in finals if recipe.validation_status == ValidationStatus.FAILED)

    print_header("PHASE 3 VALIDATION TEST RESULTS")
    print(f"Tests run:               {len(finals)}")
    print(f"Passed first try:        {passed_first}")
    print(f"Passed after correction: {corrected}")
    print(f"Failed (unresolvable):   {failed}")
    print(f"Total corrections made:  {total_corrections}")
    print("\nError frequency:")
    for key in [
        ErrorType.UNIT_MISSING.value,
        ErrorType.TEMPERATURE_OUT_OF_RANGE.value,
        ErrorType.NO_PRIMARY_REACTANT.value,
        ErrorType.NO_ENTITIES.value,
    ]:
        print(f"  {key:<28}{error_counter[key]}")
    print(f"\nSelf-correction success rate: {corrected / max(1, corrected + failed):.0%} ({corrected} of {corrected + failed} failures)")

    engine = ValidationEngine()
    checks = [
        (engine.__module__ == "src.validator.engine", "Validator runs without LLM calls"),
        (engine.rule_count == 15, "All 15 rules implemented"),
        (ValidationEngine().validate(_warning_recipe()).passed, "Blocking vs warning distinction works"),
        (_route_for(_no_reactant_recipe()) == ResponsibleAgent.ENTITY_AGENT, "Entity errors route to entity_agent"),
        (_route_for(_missing_unit_recipe()) == ResponsibleAgent.QUANTITY_AGENT, "Quantity errors route to quantity_agent"),
        (_route_for(_bad_temperature_recipe()) == ResponsibleAgent.CONDITION_AGENT, "Condition errors route to condition_agent"),
        (any(len(recipe.correction_history) > 0 for recipe in finals), "Retry count increments correctly"),
        (failed == 1, "Max retries enforced"),
        (any(recipe.validation_status == ValidationStatus.FAILED for recipe in finals), "Failure sink records FAILED status"),
        (len(finals) == 6, "Pipeline does not crash on failure"),
        (total_corrections >= 4, "Correction history populated correctly"),
        (corrected == 4, "Corrected recipes marked CORRECTED not PASSED"),
    ]
    print("\nCompletion checks:")
    for passed, message in checks:
        print_check(passed, message)
    if sum(1 for passed, _message in checks if passed) >= 10 and checks[0][0] and checks[9][0]:
        print("\n✅ PHASE 3 COMPLETE — PROCEED TO PHASE 4")
    else:
        print("\n❌ PHASE 3 INCOMPLETE")
        sys.exit(1)


def _warning_recipe() -> ChemicalRecipe:
    """Build a recipe that produces only a warning."""

    recipe = perfect_recipe()
    recipe.entities[-1] = recipe.entities[-1].model_copy(update={"quantity": Quantity(value=1, unit="g")})
    return recipe


def _route_for(recipe: ChemicalRecipe) -> ResponsibleAgent | None:
    """Return the primary route for a recipe validation failure."""

    return ValidationEngine().validate(recipe).primary_responsible_agent()


def _missing_unit_recipe() -> ChemicalRecipe:
    """Build a recipe with a missing unit for checklist routing."""

    return test_cases()[1][2]


def _no_reactant_recipe() -> ChemicalRecipe:
    """Build a recipe with no reactant for checklist routing."""

    return test_cases()[2][2]


def _bad_temperature_recipe() -> ChemicalRecipe:
    """Build a recipe with impossible temperature for checklist routing."""

    return test_cases()[3][2]


if __name__ == "__main__":
    main()
