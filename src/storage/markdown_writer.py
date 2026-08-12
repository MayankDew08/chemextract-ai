"""Pure markdown helpers for the Obsidian graph backend.

This file contains only stateless functions for generating and parsing markdown.
ObsidianVaultStore owns persistence and graph writes; these helpers only format
recipe, chemical, and paper notes with YAML frontmatter and WikiLinks.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from src.schemas.chemical import ChemicalEntity, ChemicalRole, Quantity
from src.schemas.paper import PaperMetadata
from src.schemas.recipe import ChemicalRecipe, ReactionConditions, ValidationStatus

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """Return an Obsidian-safe filename that preserves readable chemistry names."""

    safe = re.sub(r"[/\\:\*\?\"<>\|\.\(\)\[\]]+", "_", name)
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe[:100]


def generate_recipe_markdown(recipe: ChemicalRecipe) -> str:
    """Generate a reaction markdown note with WikiLinks to chemicals and paper."""

    status_icon = {
        ValidationStatus.PASSED: "✅ PASSED",
        ValidationStatus.CORRECTED: "⚠️ CORRECTED",
        ValidationStatus.FAILED: "❌ FAILED",
        ValidationStatus.PENDING: "⏳ PENDING",
    }.get(recipe.validation_status, recipe.validation_status.value)
    rows = "\n".join(
        f"| [[{entity.name}]] | {_quantity_text(entity.quantity)} | {_quantity_text(entity.moles)} | {entity.role.value} |"
        for entity in recipe.entities
    )
    conditions = recipe.conditions
    correction_history = (
        "\n".join(
            f"- Attempt {record.attempt_number}: {record.agent_that_fixed} fixed {record.error_type} — {record.error_message}"
            for record in recipe.correction_history
        )
        if recipe.correction_history
        else "No corrections needed."
    )
    title = recipe.title or f"Reaction {recipe.recipe_id}"
    paper_title = recipe.source_paper_title or "Unknown source"
    technique_tag = (conditions.technique if conditions and conditions.technique else "unknown").replace(" ", "_")
    atmosphere_tag = (conditions.atmosphere if conditions else "air").replace(" ", "_")
    recipe_payload = recipe.model_dump_json()
    return f"""---
recipe_id: {recipe.recipe_id}
validation_status: {recipe.validation_status.value}
corrections: {recipe.correction_count()}
extracted_at: {recipe.extracted_at.isoformat()}
llm_model: {recipe.llm_model}
source_url: {recipe.source_paper_url or ""}
---

# {title}

**Status:** {status_icon}
**Source:** [[{paper_title[:60]}]]
**Corrections made:** {recipe.correction_count()}

## Reactants
| Chemical | Quantity | Moles | Role |
|---|---|---|---|
{rows}

## Reaction Conditions
| Parameter | Value |
|---|---|
| Temperature | {_condition_value(conditions.temperature_celsius if conditions else None, "°C")} |
| Duration | {_condition_value(conditions.duration_hours if conditions else None, " hours")} |
| Technique | {conditions.technique if conditions and conditions.technique else "unknown"} |
| Atmosphere | {conditions.atmosphere if conditions else "unknown"} |
| Pressure | {_condition_value(conditions.pressure_atm if conditions else None, " atm")} |

## Source Text
Paper: {paper_title}
DOI: {recipe.source_paper_doi or "unknown"}
URL: {_paper_url_link(recipe.source_paper_url)}

## Correction History
{correction_history}

```chemextract-recipe-json
{recipe_payload}
```

## Tags
#reaction #{technique_tag} #{atmosphere_tag}
"""


def generate_chemical_markdown(entity: ChemicalEntity, recipe_id: str) -> str:
    """Generate a chemical note with an initial reaction WikiLink."""

    return f"""---
name: {entity.name}
formula: {entity.formula or ""}
role_first_seen: {entity.role.value}
---

# {entity.name}

**Formula:** {entity.formula or "unknown"}
**Most common role:** {entity.role.value}
**Tags:** #{entity.role.value.lower()} #chemical

## Appears In Reactions
- [[{recipe_id}]] — as {entity.role.value}

## Co-occurs With
"""


def append_recipe_link_to_chemical(md_path: Path, recipe_id: str, role: str) -> None:
    """Append a reaction WikiLink to an existing chemical note idempotently."""

    content = md_path.read_text(encoding="utf-8")
    line = f"- [[{recipe_id}]] — as {role}"
    if line in content:
        return
    section = "## Appears In Reactions"
    if section not in content:
        content = f"{content.rstrip()}\n\n{section}\n{line}\n"
    else:
        content = content.replace(section, f"{section}\n{line}", 1)
    md_path.write_text(content, encoding="utf-8")


def generate_paper_markdown(metadata: PaperMetadata) -> str:
    """Generate a source paper note for provenance."""

    authors = ", ".join(metadata.authors)
    return f"""---
doi: {metadata.doi or ""}
year: {metadata.year or ""}
source_url: {metadata.open_access_url or ""}
---

# {metadata.title}

**DOI:** {metadata.doi or "unknown"}
**Year:** {metadata.year or "unknown"}
**Authors:** {authors}
**Source DB:** {metadata.source_db}
**URL:** {_paper_url_link(metadata.open_access_url)}

## Recipes Extracted From This Paper

## Abstract
{metadata.abstract or ""}
"""


def parse_recipe_from_markdown(md_path: Path) -> Optional[ChemicalRecipe]:
    """Parse a reaction markdown file into a partial ChemicalRecipe, never raising."""

    try:
        content = md_path.read_text(encoding="utf-8")
        payload_match = re.search(
            r"```chemextract-recipe-json\s*\n(?P<payload>.*?)\n```",
            content,
            flags=re.DOTALL,
        )
        if payload_match:
            try:
                return ChemicalRecipe.model_validate_json(payload_match.group("payload"))
            except Exception as exc:
                logger.warning("Invalid embedded recipe payload in %s: %s", md_path, exc)
        frontmatter = _frontmatter(content)
        recipe_id = frontmatter.get("recipe_id")
        if not recipe_id:
            return None
        status = ValidationStatus(frontmatter.get("validation_status", ValidationStatus.PASSED.value))
        entities = _parse_entities_table(content)
        conditions = _parse_conditions_table(content)
        title_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        return ChemicalRecipe(
            recipe_id=recipe_id,
            source_chunk_id=recipe_id,
            title=title_match.group(1) if title_match else None,
            entities=entities,
            conditions=conditions,
            validation_status=status,
            llm_model=frontmatter.get("llm_model", "unknown"),
            source_paper_url=frontmatter.get("source_url") or None,
        )
    except Exception as exc:
        logger.warning("Failed to parse recipe markdown %s: %s", md_path, exc)
        return None


def _quantity_text(quantity: Optional[Quantity]) -> str:
    """Render optional quantities consistently in markdown tables."""

    return str(quantity) if quantity else "-"


def _paper_url_link(source_url: Optional[str]) -> str:
    """Render the original paper URL as a readable Markdown link."""

    if not source_url:
        return "unknown"
    parsed = urlparse(source_url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return "invalid source URL"
    return f"[Open original paper]({source_url})"


def _condition_value(value: Optional[float], suffix: str) -> str:
    """Render optional condition values for tables."""

    return f"{value}{suffix}" if value is not None else "unknown"


def _frontmatter(content: str) -> dict[str, str]:
    """Parse simple YAML frontmatter without introducing a runtime dependency."""

    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    values = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _parse_entities_table(content: str) -> list[ChemicalEntity]:
    """Parse the markdown entities table emitted by generate_recipe_markdown."""

    entities = []
    for line in content.splitlines():
        if not line.startswith("| [[") or " | " not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        name = cells[0].removeprefix("[[").removesuffix("]]")
        quantity = _parse_quantity(cells[1])
        moles = _parse_quantity(cells[2])
        role = ChemicalRole(cells[3])
        entities.append(ChemicalEntity(name=name, role=role, quantity=quantity, moles=moles))
    return entities


def _parse_conditions_table(content: str) -> Optional[ReactionConditions]:
    """Parse the generated conditions table into ReactionConditions."""

    values = {}
    for line in content.splitlines():
        if line.startswith("| Temperature |"):
            values["temperature_celsius"] = _parse_float(line)
        elif line.startswith("| Duration |"):
            values["duration_hours"] = _parse_float(line)
        elif line.startswith("| Technique |"):
            values["technique"] = line.strip("|").split("|")[1].strip()
        elif line.startswith("| Atmosphere |"):
            values["atmosphere"] = line.strip("|").split("|")[1].strip()
        elif line.startswith("| Pressure |"):
            values["pressure_atm"] = _parse_float(line) or 1.0
    if not values:
        return None
    return ReactionConditions(**values)


def _parse_quantity(text: str) -> Optional[Quantity]:
    """Parse a quantity cell from generated markdown."""

    if text == "-":
        return None
    match = re.match(r"(\d+(?:\.\d+)?)\s+(.+)", text)
    if not match:
        return None
    return Quantity(value=float(match.group(1)), unit=match.group(2))


def _parse_float(line: str) -> Optional[float]:
    """Extract the first numeric value from a markdown table line."""

    match = re.search(r"-?\d+(?:\.\d+)?", line)
    return float(match.group(0)) if match else None
