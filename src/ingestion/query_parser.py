"""LLM-backed and deterministic query parsers for ingestion.

This file contains GeminiQueryParser, GroqQueryParser, ResilientQueryParser, and
FallbackQueryParser. Provider-specific classes keep model integrations swappable,
while the resilient wrapper prevents a retired or unavailable model from taking
down the ingestion pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from src.ingestion.base import BaseQueryParser
from src.schemas.query import ChemistryDomain, SearchPlan, SourceSearchQuery

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are an expert chemistry literature search specialist.
A user wants to find synthesis procedures for a chemical compound.

User query: {raw_query}
Registered database sources: {source_names}

Produce a structured search plan:
1. Identify the target compound and chemical formula if known
2. List common synonyms (minimum 2, maximum 8 real names only)
3. Classify domain (nanomaterials/battery_materials/pharmaceuticals/
   polymers/catalysis/organic_synthesis/inorganic_synthesis/general)
4. For EACH registered source write an optimized query:
   - pubmed: compound name + synthesis + [tiab] field tags
   - arxiv: compound name + synthesis, no special syntax
   - openalex: compound full name, complete phrases work best
   - semantic_scholar: compound name + synthesis + methods
5. List 2-5 chemicals commonly used to synthesize this compound

Return valid JSON matching SearchPlan schema exactly."""

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


class GeminiQueryParser(BaseQueryParser):
    """GeminiQueryParser uses structured LLM output to optimize per-source search."""

    def __init__(self) -> None:
        """Delay LLM client creation so missing keys do not break imports."""

        self._llm: Optional[object] = None

    def _get_llm(self) -> object:
        """Create the Gemini client only when parsing first requires it."""

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured")
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
            self._llm = ChatGoogleGenerativeAI(model=model, temperature=0.1, google_api_key=api_key)
        return self._llm

    async def parse(self, raw_query: str, registered_sources: list[str]) -> SearchPlan:
        """Parse a raw query with Gemini and backfill any missing source query."""

        try:
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            llm = self._get_llm()
            structured_llm = llm.with_structured_output(SearchPlan)
            chain = prompt | structured_llm
            plan = await chain.ainvoke({"raw_query": raw_query, "source_names": ", ".join(registered_sources)})
            return self._ensure_registered_sources(plan, registered_sources)
        except Exception as exc:
            logger.error("Gemini query parsing failed: %s", exc)
            raise ValueError(str(exc)) from exc

    def _ensure_registered_sources(self, plan: SearchPlan, registered_sources: list[str]) -> SearchPlan:
        """Guarantee every registered source has a source-specific search query."""

        source_queries = dict(plan.source_queries)
        for source in registered_sources:
            if source not in source_queries:
                source_queries[source] = SourceSearchQuery(
                    source_name=source,
                    query_string=f"{plan.target_compound} synthesis experimental",
                )
        return plan.model_copy(update={"source_queries": source_queries})


class GroqQueryParser(BaseQueryParser):
    """GroqQueryParser gives Phase 1 a fast Llama-backed query planning option."""

    def __init__(self) -> None:
        """Delay Groq client creation so tests do not require provider credentials."""

        self._llm: Optional[object] = None

    def _get_llm(self) -> object:
        """Create the Groq chat model only when parsing first requires it."""

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured")
        if self._llm is None:
            from langchain_groq import ChatGroq

            model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
            self._llm = ChatGroq(model=model, temperature=0.1, groq_api_key=api_key)
        return self._llm

    async def parse(self, raw_query: str, registered_sources: list[str]) -> SearchPlan:
        """Parse with Groq by requesting JSON and validating it with Pydantic."""

        try:
            prompt = ChatPromptTemplate.from_template(
                f"{PROMPT_TEMPLATE}\n\nReturn only one JSON object. Do not wrap it in markdown fences."
            )
            llm = self._get_llm()
            chain = prompt | llm
            response = await chain.ainvoke({"raw_query": raw_query, "source_names": ", ".join(registered_sources)})
            plan = SearchPlan.model_validate_json(_extract_response_text(response))
            return _ensure_registered_sources(plan, registered_sources)
        except Exception as exc:
            logger.error("Groq query parsing failed: %s", exc)
            raise ValueError(str(exc)) from exc


class ResilientQueryParser(BaseQueryParser):
    """ResilientQueryParser cascades providers before falling back deterministically."""

    def __init__(self, parsers: list[BaseQueryParser]) -> None:
        """Accept ordered parsers so environment policy can choose provider priority."""

        self._parsers = parsers

    async def parse(self, raw_query: str, registered_sources: list[str]) -> SearchPlan:
        """Try each parser in order so one provider outage does not stop ingestion."""

        last_error: Optional[Exception] = None
        for parser in self._parsers:
            try:
                return await parser.parse(raw_query, registered_sources)
            except ValueError as exc:
                last_error = exc
                logger.warning("%s failed, trying next parser: %s", parser.__class__.__name__, exc)
        raise ValueError(str(last_error) if last_error else "No query parsers configured")


class FallbackQueryParser(BaseQueryParser):
    """FallbackQueryParser provides deterministic plans without external LLM calls."""

    STOPWORDS = {
        "synthesis",
        "of",
        "how",
        "to",
        "make",
        "prepare",
        "synthesize",
        "show",
        "me",
        "find",
        "routes",
        "for",
    }

    async def parse(self, raw_query: str, registered_sources: list[str]) -> SearchPlan:
        """Build a conservative search plan from non-stopword query tokens."""

        tokens = [token.strip() for token in raw_query.lower().split() if token.strip()]
        compound_tokens = [token for token in tokens if token not in self.STOPWORDS]
        compound_name = " ".join(compound_tokens).strip() or raw_query.strip()
        query_string = f"{compound_name} synthesis experimental methods"
        source_queries = {
            source: SourceSearchQuery(source_name=source, query_string=query_string)
            for source in registered_sources
        }
        return SearchPlan(
            raw_query=raw_query,
            target_compound=compound_name,
            synonyms=[compound_name] if compound_name else [],
            source_queries=source_queries,
            domain=ChemistryDomain.GENERAL,
        )


def build_query_parser_from_env() -> BaseQueryParser:
    """Build a provider cascade from env so demos and apps share one policy."""

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    parsers: list[BaseQueryParser] = []
    if provider == "groq":
        if os.getenv("GROQ_API_KEY"):
            parsers.append(GroqQueryParser())
        if os.getenv("GEMINI_API_KEY"):
            parsers.append(GeminiQueryParser())
    else:
        if os.getenv("GEMINI_API_KEY"):
            parsers.append(GeminiQueryParser())
        if os.getenv("GROQ_API_KEY"):
            parsers.append(GroqQueryParser())
    parsers.append(FallbackQueryParser())
    return ResilientQueryParser(parsers)


def _ensure_registered_sources(plan: SearchPlan, registered_sources: list[str]) -> SearchPlan:
    """Guarantee every registered source has a query for non-Gemini parsers."""

    source_queries = dict(plan.source_queries)
    for source in registered_sources:
        if source not in source_queries:
            source_queries[source] = SourceSearchQuery(
                source_name=source,
                query_string=f"{plan.target_compound} synthesis experimental",
            )
    return plan.model_copy(update={"source_queries": source_queries})


def _extract_response_text(response: object) -> str:
    """Extract and clean JSON text from a LangChain chat response."""

    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    text = str(content).strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(text)
    return json.dumps(parsed)
