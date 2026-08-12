"""Ten pre-computed synthesis recipes transcribed from open literature.

The chemistry fields below come from the linked papers. Observability values are
deterministic demo measurements shaped like real extraction runs; they are not
reported by the chemistry papers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.schemas.chemical import ChemicalEntity, ChemicalRole, Quantity
from src.schemas.recipe import ChemicalRecipe, CorrectionRecord, ReactionConditions, ValidationStatus

_EXTRACTED_AT = datetime(2026, 8, 12, 8, 0, tzinfo=UTC)


def _entity(
    name: str,
    formula: str,
    role: ChemicalRole,
    value: float | None = None,
    unit: str | None = None,
    notes: str | None = None,
) -> ChemicalEntity:
    """Build a measured entity without sharing mutable objects between recipes."""

    quantity = Quantity(value=value, unit=unit) if value is not None and unit is not None else None
    return ChemicalEntity(name=name, formula=formula, role=role, quantity=quantity, notes=notes)


def _recipe(
    *,
    recipe_id: str,
    title: str,
    paper_title: str,
    doi: str,
    pmcid: str,
    entities: list[ChemicalEntity],
    temperature: float,
    duration: float,
    technique: str,
    atmosphere: str = "air",
    additional: dict[str, str] | None = None,
    corrected: tuple[str, str, str] | None = None,
    tokens: tuple[int, int, int],
    latencies: tuple[float, float, float],
    cost: float,
    offset_hours: int,
) -> ChemicalRecipe:
    """Create one fully populated immutable-by-construction seed recipe."""

    node_names = ("entity_node", "quantity_node", "condition_node")
    history = []
    if corrected:
        error_type, error_field, message = corrected
        history = [
            CorrectionRecord(
                attempt_number=1,
                error_type=error_type,
                error_field=error_field,
                error_message=message,
                agent_that_fixed="seed_correction_agent",
                corrected_at=_EXTRACTED_AT + timedelta(hours=offset_hours, minutes=1),
            )
        ]
    return ChemicalRecipe(
        recipe_id=recipe_id,
        source_chunk_id=f"{pmcid.lower()}-experimental",
        source_paper_title=paper_title,
        source_paper_doi=doi,
        source_paper_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        title=title,
        entities=entities,
        conditions=ReactionConditions(
            temperature_celsius=temperature,
            duration_hours=duration,
            atmosphere=atmosphere,
            technique=technique,
            additional_conditions=additional or {},
        ),
        validation_status=ValidationStatus.CORRECTED if history else ValidationStatus.PASSED,
        correction_history=history,
        total_tokens_used=sum(tokens),
        estimated_cost_usd=cost,
        total_latency_seconds=round(sum(latencies), 3),
        node_latencies=dict(zip(node_names, latencies, strict=True)),
        node_tokens=dict(zip(node_names, tokens, strict=True)),
        extracted_at=_EXTRACTED_AT + timedelta(hours=offset_hours),
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
    )


def load_seed_recipes() -> list[ChemicalRecipe]:
    """Return fresh recipe objects for five ZnO, three Fe3O4, and two TiO2 syntheses."""

    reactant = ChemicalRole.REACTANT
    solvent = ChemicalRole.SOLVENT
    additive = ChemicalRole.ADDITIVE
    product = ChemicalRole.PRODUCT
    return [
        _recipe(
            recipe_id="lit_zno_wet_chemical_2024",
            title="Wet-chemical synthesis of ZnO nanoparticles",
            paper_title="Synthesis of ZnO and PEG-ZnO nanoparticles (NPs) with controlled size for biological evaluation",
            doi="10.1039/D3RA07441B",
            pmcid="PMC10783289",
            entities=[
                _entity("zinc acetate dihydrate", "Zn(CH3COO)2·2H2O", reactant, 4.18, "g"),
                _entity("methanol", "CH3OH", solvent, 220, "mL"),
                _entity("deionized water", "H2O", solvent, 15, "mL"),
                _entity("sodium hydroxide", "NaOH", reactant, 1.45, "g", "Dissolved in 120 mL methanol"),
                _entity("ZnO", "ZnO", product),
            ],
            temperature=60,
            duration=2.333,
            technique="wet chemical precipitation",
            atmosphere="argon",
            additional={"stirring_rate": "400 rpm", "drying": "60 °C overnight"},
            tokens=(1372, 948, 711),
            latencies=(1.842, 1.204, 0.891),
            cost=0.00182,
            offset_hours=0,
        ),
        _recipe(
            recipe_id="lit_zno_rice_hydrothermal_2013",
            title="Rice-biTemplate hydrothermal synthesis of ZnO",
            paper_title="Hydrothermal synthesis of zinc oxide nanoparticles using rice as soft biotemplate",
            doi="10.1186/1752-153X-7-136",
            pmcid="PMC3751658",
            entities=[
                _entity("zinc acetate dihydrate", "Zn(CH3COO)2·2H2O", reactant, 1.0, "g"),
                _entity("sodium hydroxide", "NaOH", reactant, 0.8, "g"),
                _entity("distilled water", "H2O", solvent, 25, "mL"),
                _entity("uncooked rice flour", "C6H10O5", additive, 0.25, "g", "Soft biotemplate"),
                _entity("ZnO", "ZnO", product),
            ],
            temperature=120,
            duration=18,
            technique="hydrothermal biotemplate",
            additional={"initial_stirring": "1 hour", "pH": "13"},
            corrected=("ROLE_MISCLASSIFICATION", "entities[3].role", "Rice flour was corrected from reactant to additive."),
            tokens=(1540, 1062, 826),
            latencies=(2.115, 1.447, 1.023),
            cost=0.00211,
            offset_hours=1,
        ),
        _recipe(
            recipe_id="lit_zno_hydrothermal_160c_2023",
            title="Citric-acid-assisted hydrothermal synthesis of ZnO at 160 °C",
            paper_title="Hydrothermal Synthesis of ZnO Superstructures with Controlled Morphology via Temperature and pH Optimization",
            doi="10.3390/ma16041641",
            pmcid="PMC9960931",
            entities=[
                _entity("zinc acetate dihydrate", "Zn(CH3COO)2·2H2O", reactant, 4.4, "g"),
                _entity("distilled water", "H2O", solvent, 1320, "mL"),
                _entity("citric acid monohydrate", "C6H8O7·H2O", additive, 2.2, "g"),
                _entity("sodium hydroxide solution", "NaOH", reactant, 440, "mL", "1 mol L−1"),
                _entity("ZnO", "ZnO", product),
            ],
            temperature=160,
            duration=24,
            technique="hydrothermal synthesis",
            additional={"drying": "60 °C for 24 hours", "washing": "water and ethanol, three times"},
            tokens=(1468, 1004, 792),
            latencies=(1.963, 1.326, 0.974),
            cost=0.00196,
            offset_hours=2,
        ),
        _recipe(
            recipe_id="lit_zno_naoh_forced_hydrolysis_2018",
            title="NaOH forced-hydrolysis synthesis of ZnO",
            paper_title="ZnO nanoparticle preparation route influences surface reactivity, dissolution and cytotoxicity",
            doi="10.1039/C7EN00888K",
            pmcid="PMC5823520",
            entities=[
                _entity("zinc nitrate hexahydrate solution", "Zn(NO3)2·6H2O", reactant, 0.5, "M"),
                _entity("sodium hydroxide solution", "NaOH", reactant, 1.0, "M"),
                _entity("nanopure water", "H2O", solvent),
                _entity("ZnO", "ZnO", product),
            ],
            temperature=80,
            duration=2,
            technique="forced hydrolysis",
            additional={"mixing": "equal solution volumes", "drying": "65 °C for more than 12 hours"},
            corrected=("QUANTITY_CONTEXT", "entities[0].quantity", "The reported 0.5 M value was normalized as concentration, not mass."),
            tokens=(1291, 903, 668),
            latencies=(1.705, 1.158, 0.803),
            cost=0.00167,
            offset_hours=3,
        ),
        _recipe(
            recipe_id="lit_zno_oriented_attachment_2019",
            title="Oriented-attachment synthesis of dispersible ZnO",
            paper_title="Preparation of ZnO Nanoparticles with High Dispersibility Based on Oriented Attachment (OA) Process",
            doi="10.1186/s11671-019-3038-3",
            pmcid="PMC6586737",
            entities=[
                _entity("zinc acetate dihydrate", "Zn(CH3COO)2·2H2O", reactant, 3.73, "mmol"),
                _entity("sodium hydroxide", "NaOH", reactant, 7.22, "mmol"),
                _entity("ethanol", "C2H5OH", solvent, 40, "mL", "Solution A; Solution B used a further 25 mL"),
                _entity("bi-distilled water", "H2O", solvent, 320, "uL"),
                _entity("ZnO", "ZnO", product),
            ],
            temperature=60,
            duration=2.25,
            technique="oriented attachment sol-gel",
            additional={"addition": "Solution B added dropwise under vigorous stirring"},
            tokens=(1417, 978, 736),
            latencies=(1.887, 1.278, 0.916),
            cost=0.00189,
            offset_hours=4,
        ),
        _recipe(
            recipe_id="lit_fe3o4_chamomile_microwave_2024",
            title="Chamomile-mediated microwave hydrothermal synthesis of Fe3O4",
            paper_title="Plant-Mediated Synthesis of Magnetite Nanoparticles with Matricaria chamomilla Aqueous Extract",
            doi="10.3390/nano14080729",
            pmcid="PMC11053587",
            entities=[
                _entity("iron(III) chloride solution", "FeCl3", reactant, 4, "mL", "1 M"),
                _entity("sodium hydroxide solution", "NaOH", reactant, 4, "mL", "8 M"),
                _entity("aqueous chamomile flower extract", "H2O", additive, 32, "mL"),
                _entity("Fe3O4", "Fe3O4", product),
            ],
            temperature=200,
            duration=0.333,
            technique="microwave hydrothermal green synthesis",
            additional={"sample": "S1"},
            corrected=("FORMULA_NORMALIZATION", "entities[3].formula", "Magnetite product name was normalized to Fe3O4."),
            tokens=(1613, 1098, 857),
            latencies=(2.246, 1.506, 1.117),
            cost=0.00227,
            offset_hours=5,
        ),
        _recipe(
            recipe_id="lit_fe3o4_monodisperse_2022",
            title="One-pot hydrothermal synthesis of monodisperse Fe3O4 microparticles",
            paper_title="Size-Controllable Synthesis of Monodisperse Magnetite Microparticles Leading to Magnetically Tunable Colloidal Crystals",
            doi="10.3390/ma15144943",
            pmcid="PMC9323182",
            entities=[
                _entity("anhydrous iron(III) chloride", "FeCl3", reactant, 0.52, "g"),
                _entity("ethylene glycol", "C2H6O2", solvent, 32, "mL"),
                _entity("anhydrous sodium acetate", "CH3COONa", additive, 2.4, "g"),
                _entity("PSSMA", "C12H9NaO7S", additive, 0.8, "g"),
                _entity("ultrapure water", "H2O", solvent, 120, "uL"),
                _entity("sodium hydroxide", "NaOH", reactant, 0.36, "g"),
                _entity("Fe3O4", "Fe3O4", product),
            ],
            temperature=190,
            duration=16,
            technique="one-pot hydrothermal synthesis",
            additional={"pre_stirring": "20 minutes", "vacuum_drying": "24 hours"},
            tokens=(1738, 1184, 901),
            latencies=(2.418, 1.614, 1.209),
            cost=0.00243,
            offset_hours=6,
        ),
        _recipe(
            recipe_id="lit_fe3o4_room_temperature_2013",
            title="Room-temperature co-precipitation of Fe3O4 with NaOH",
            paper_title="Room Temperature Co-Precipitation Synthesis of Magnetite Nanoparticles in a Large pH Window with Different Bases",
            doi="10.3390/ma6125549",
            pmcid="PMC5452734",
            entities=[
                _entity("iron(II) chloride tetrahydrate", "FeCl2·4H2O", reactant, 0.01, "mol"),
                _entity("iron(III) chloride hexahydrate", "FeCl3·6H2O", reactant, 0.02, "mol"),
                _entity("degassed deionized water", "H2O", solvent, 100, "mL"),
                _entity("sodium hydroxide", "NaOH", reactant, 0.08, "mol"),
                _entity("Fe3O4", "Fe3O4", product),
            ],
            temperature=25,
            duration=1,
            technique="slow co-precipitation",
            atmosphere="nitrogen",
            additional={"base_addition_rate": "1.0 mL/min", "final_pH": "10.34", "sample": "S1"},
            tokens=(1506, 1022, 781),
            latencies=(2.034, 1.369, 0.998),
            cost=0.00204,
            offset_hours=7,
        ),
        _recipe(
            recipe_id="lit_tio2_ticl4_solgel_2017",
            title="TiCl4 sol-gel synthesis of TiO2 nanoparticles",
            paper_title="Synthesis and Characterization of TiO2 Nanoparticles for the Reduction of Water Pollutants",
            doi="10.3390/ma10101208",
            pmcid="PMC5667014",
            entities=[
                _entity("titanium tetrachloride", "TiCl4", reactant, 5, "mL"),
                _entity("ethanol", "C2H5OH", solvent, 50, "mL"),
                _entity("bidistilled water", "H2O", solvent, 200, "mL"),
                _entity("TiO2", "TiO2", product),
            ],
            temperature=50,
            duration=24,
            technique="sol-gel synthesis and drying",
            additional={"stirring": "30 minutes before and after water addition"},
            tokens=(1349, 927, 704),
            latencies=(1.798, 1.224, 0.872),
            cost=0.00179,
            offset_hours=8,
        ),
        _recipe(
            recipe_id="lit_tio2_coprecipitation_2019",
            title="Co-precipitation and calcination synthesis of TiO2 nanoparticles",
            paper_title="Appraisal of Comparative Therapeutic Potential of Undoped and Nitrogen-Doped Titanium Dioxide Nanoparticles",
            doi="10.3390/molecules24213916",
            pmcid="PMC6864622",
            entities=[
                _entity("titanium isopropoxide", "C12H28O4Ti", reactant, 15, "mL"),
                _entity("isopropyl alcohol", "C3H8O", solvent, 100, "mL"),
                _entity("deionized water", "H2O", solvent, 10, "mL"),
                _entity("TiO2", "TiO2", product),
            ],
            temperature=800,
            duration=4,
            technique="co-precipitation and calcination",
            additional={"initial_stirring": "30 minutes", "drying": "100 °C before calcination"},
            corrected=("CONDITION_STAGE", "conditions.temperature_celsius", "The final 800 °C calcination stage was selected as the normalized temperature."),
            tokens=(1442, 986, 759),
            latencies=(1.926, 1.301, 0.951),
            cost=0.00193,
            offset_hours=9,
        ),
    ]
