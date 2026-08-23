"""Recipe API routes for browsing extracted recipes."""

from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.dependencies import get_store
from src.schemas.chemical import ChemicalRole
from src.storage.base import BaseGraphStore

logger = logging.getLogger(__name__)

router = APIRouter()


class RecipeSummary(BaseModel):
    """RecipeSummary is the compact recipe list representation."""

    recipe_id: str
    title: Optional[str]
    validation_status: str
    entity_count: int
    correction_count: int
    warning_count: int
    temperature_celsius: Optional[float]
    duration_hours: Optional[float]
    product_names: list[str]
    latency_seconds: float
    cost_usd: float
    extracted_at: str


@router.get("/recipes")
async def list_recipes(
    store: Annotated[BaseGraphStore, Depends(get_store)],
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Return compact summaries for stored recipes with optional status filtering."""

    recipes = store.get_all_recipes()
    if status:
        recipes = [recipe for recipe in recipes if recipe.validation_status.value == status]
    return [_summarize(recipe) for recipe in recipes[:limit]]


@router.get("/recipes/{recipe_id}")
async def get_recipe(recipe_id: str, store: Annotated[BaseGraphStore, Depends(get_store)]):
    """Return a full recipe or 404 when the recipe is unknown."""

    recipe = store.get_recipe(recipe_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail=f"Recipe {recipe_id} not found")
    return recipe


def _summarize(recipe) -> RecipeSummary:
    """Build one recipe summary without leaking large recipe payloads."""

    conditions = recipe.conditions
    return RecipeSummary(
        recipe_id=recipe.recipe_id,
        title=recipe.title,
        validation_status=recipe.validation_status.value,
        entity_count=len(recipe.entities),
        correction_count=recipe.correction_count(),
        warning_count=len(recipe.validation_warnings),
        temperature_celsius=conditions.temperature_celsius if conditions else None,
        duration_hours=conditions.duration_hours if conditions else None,
        product_names=[entity.name for entity in recipe.entities if entity.role == ChemicalRole.PRODUCT],
        latency_seconds=recipe.total_latency_seconds,
        cost_usd=recipe.estimated_cost_usd,
        extracted_at=recipe.extracted_at.isoformat(),
    )
