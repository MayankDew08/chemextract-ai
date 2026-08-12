"""HTML view routes for the local ChemExtract dashboard."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from src.api.dependencies import get_templates

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root() -> RedirectResponse:
    """Redirect the app root to the dashboard."""

    return RedirectResponse("/dashboard")


@router.get("/dashboard")
async def dashboard(request: Request, templates: Annotated[Jinja2Templates, Depends(get_templates)]):
    """Serve the dashboard HTML shell."""

    return templates.TemplateResponse(request=request, name="dashboard.html", context={})


@router.get("/recipes")
async def recipes(request: Request, templates: Annotated[Jinja2Templates, Depends(get_templates)]):
    """Serve the recipe browser HTML shell."""

    return templates.TemplateResponse(request=request, name="recipes.html", context={})


@router.get("/recipes/{recipe_id}")
async def recipe_detail(recipe_id: str, request: Request, templates: Annotated[Jinja2Templates, Depends(get_templates)]):
    """Serve the recipe detail HTML shell."""

    return templates.TemplateResponse(request=request, name="recipe_detail.html", context={"recipe_id": recipe_id})


@router.get("/graph")
async def graph(request: Request, templates: Annotated[Jinja2Templates, Depends(get_templates)]):
    """Serve the knowledge graph HTML shell."""

    return templates.TemplateResponse(request=request, name="graph.html", context={})
