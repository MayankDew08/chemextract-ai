"""Pytest coverage for Phase 1 public ingestion seams.

These tests exercise deterministic behavior at the schema, registry, parser,
deduplication, ranking, and PubMed configuration boundaries. They avoid live
network calls so the regular test suite stays fast and stable while the demo
folder remains responsible for end-to-end API verification.
"""

from __future__ import annotations

import pytest

from src.ingestion.deduplicator import DOITitleDeduplicator
from src.ingestion.fetchers.pubmed import PubMedFetcher
from src.ingestion.fetchers.openalex import OpenAlexFetcher
from src.ingestion.fetchers.semantic_scholar import SemanticScholarFetcher
from src.ingestion.query_parser import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    FallbackQueryParser,
    ResilientQueryParser,
    build_query_parser_from_env,
)
from src.ingestion.ranker import ChemistryRelevanceRanker
from src.ingestion.registry import FetcherRegistry, bootstrap_registry
from src.ingestion.utils import (
    calculate_chemical_density,
    detect_experimental_section,
    generate_chunk_id,
    is_chemistry_text,
)
from src.schemas.paper import PaperMetadata
from src.schemas.query import ChemistryDomain


@pytest.mark.asyncio
async def test_pipeline_uses_query_parser_fallback_when_configured_llm_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured but unavailable query-planning LLM must not erase Phase 1."""

    from src import pipeline as pipeline_module
    from src.ingestion import orchestrator as orchestrator_module
    from src.ingestion import query_parser as query_parser_module

    class BrokenGeminiParser:
        """Simulate a configured Gemini adapter that cannot initialize."""

        async def parse(self, raw_query: str, registered_sources: list[str]):
            """Raise the provider failure seen in production."""

            raise ValueError("langchain_google_genai is unavailable")

    class CapturingOrchestrator:
        """Capture the parser and prove it can produce a fallback plan."""

        def __init__(self, **kwargs):
            """Store the parser passed by the pipeline."""

            self._parser = kwargs["query_parser"]

        async def run(self, query: str, skip_health_check: bool = False):
            """Parse the query and return a sentinel result when planning succeeds."""

            await self._parser.parse(query, ["pubmed"])
            return ["fallback-plan-worked"]

    monkeypatch.setenv("GEMINI_API_KEY", "configured-but-broken")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(query_parser_module, "GeminiQueryParser", BrokenGeminiParser)
    monkeypatch.setattr(orchestrator_module, "FetchOrchestrator", CapturingOrchestrator)

    result = await pipeline_module.ChemExtractPipeline(
        pipeline_module.PipelineConfig(query="ZnO synthesis")
    )._run_phase1()

    assert result == ["fallback-plan-worked"]


@pytest.mark.asyncio
async def test_fallback_query_parser_builds_queries_for_all_registered_sources() -> None:
    """Fallback parsing must keep Phase 1 usable when Gemini is unavailable."""

    parser = FallbackQueryParser()

    plan = await parser.parse("synthesis of zinc oxide nanoparticles", ["pubmed", "openalex"])

    assert plan.target_compound == "zinc oxide nanoparticles"
    assert plan.domain == ChemistryDomain.GENERAL
    assert set(plan.source_queries) == {"pubmed", "openalex"}
    assert plan.source_queries["pubmed"].query_string == "zinc oxide nanoparticles synthesis experimental methods"


def test_pubmed_fetcher_prefers_pubmed_env_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """PubMed E-utilities should honor the user's PUBMED_API and PUBMED_EMAIL names."""

    monkeypatch.setenv("PUBMED_API", "pubmed-key")
    monkeypatch.setenv("PUBMED_EMAIL", "chemist@example.com")
    monkeypatch.setenv("NCBI_API_KEY", "legacy-key")
    monkeypatch.setenv("API_EMAIL", "legacy@example.com")

    registry = bootstrap_registry()
    fetcher = next(item for item in registry.get_all_fetchers() if item.source_name == "pubmed")
    acquirer = next(item for item in registry.get_all_acquirers() if item.acquirer_name == "pubmed_central_xml")

    assert fetcher._params({"db": "pubmed"}) == {
        "db": "pubmed",
        "email": "chemist@example.com",
        "api_key": "pubmed-key",
    }
    assert acquirer._email == "chemist@example.com"


def test_pubmed_metadata_uses_canonical_paper_url_without_pmc() -> None:
    """PubMed-only search results must retain a usable source paper URL."""

    xml = """
    <PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>12345678</PMID>
      <Article><ArticleTitle>CuO synthesis</ArticleTitle><Abstract><AbstractText>
      Copper acetate was dissolved and heated to synthesize CuO nanoparticles.
      </AbstractText></Abstract></Article>
    </MedlineCitation><PubmedData><ArticleIdList>
      <ArticleId IdType="pubmed">12345678</ArticleId>
    </ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>
    """

    papers = PubMedFetcher()._parse_pubmed_xml(xml)

    assert papers[0].open_access_url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


def test_openalex_metadata_uses_work_page_without_oa_location() -> None:
    """OpenAlex metadata must retain its canonical work URL as provenance."""

    paper = OpenAlexFetcher()._parse_work(
        {
            "id": "https://openalex.org/W123456789",
            "title": "Fe3O4 nanoparticle synthesis",
            "open_access": {"is_oa": False, "oa_url": None},
        }
    )

    assert paper is not None
    assert paper.open_access_url == "https://openalex.org/W123456789"


def test_semantic_scholar_metadata_uses_paper_page_without_oa_pdf() -> None:
    """Semantic Scholar metadata must retain a canonical paper-page URL."""

    paper = SemanticScholarFetcher()._parse_paper(
        {
            "paperId": "abc123",
            "title": "Silver nanoparticle synthesis",
            "isOpenAccess": False,
            "openAccessPdf": None,
        }
    )

    assert paper is not None
    assert paper.open_access_url == "https://www.semanticscholar.org/paper/abc123"


def test_query_parser_env_builder_uses_current_provider_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment parser selection should expose current Gemini and Groq defaults."""

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    parser = build_query_parser_from_env()

    assert isinstance(parser, ResilientQueryParser)
    assert DEFAULT_GEMINI_MODEL == "gemini-2.5-flash"
    assert DEFAULT_GROQ_MODEL == "llama-3.3-70b-versatile"


def test_is_chemistry_text_rejects_generic_prose() -> None:
    """Chemistry procedures pass while generic paper prose is skipped."""

    chemistry_text = (
        "Zinc acetate was dissolved in methanol solution, stirred, heated at 65°C, "
        "and filtered to synthesize ZnO nanoparticles after the reaction."
    )
    generic_procedure = (
        "The compound was prepared using standard laboratory methods. Some chemicals were "
        "mixed and heated. The resulting material was isolated after routine processing."
    )
    chemistry_procedure_without_measurements = (
        "The catalyst was prepared by refluxing the solution overnight, then filtered, "
        "washed, and dried before the reaction mixture was collected for characterization."
    )
    interdisciplinary_prose = (
        "The dataset and neural model were prepared using standard methods. The classifier "
        "reaction to the benchmark mixture was collected and evaluated across all experiments."
    )
    assert is_chemistry_text(chemistry_text)
    assert is_chemistry_text(chemistry_text.replace("ZnO", "TiO2"))
    assert is_chemistry_text(chemistry_procedure_without_measurements)
    assert not is_chemistry_text(generic_procedure)
    assert not is_chemistry_text(interdisciplinary_prose)
    assert not is_chemistry_text("The results showed significant improvement.")
    assert not is_chemistry_text("")


def test_registry_bootstrap_registers_phase1_sources_and_priority_acquirers() -> None:
    """Bootstrap should expose all concrete plugins through the registry seam."""

    registry = bootstrap_registry()

    assert set(registry.get_registered_source_names()) == {"pubmed", "arxiv", "openalex", "semantic_scholar"}
    assert [acquirer.priority() for acquirer in registry.get_all_acquirers()] == sorted(
        acquirer.priority() for acquirer in registry.get_all_acquirers()
    )
    assert isinstance(registry, FetcherRegistry)


def test_deduplicator_merges_exact_doi_and_preserves_rich_metadata() -> None:
    """DOI collisions should collapse into one richer paper record."""

    deduplicator = DOITitleDeduplicator()
    primary = PaperMetadata(title="Zinc oxide synthesis", doi="https://doi.org/10.1/ABC", source_db="openalex")
    richer = PaperMetadata(
        title="Zinc oxide synthesis",
        doi="10.1/abc",
        source_db="semantic_scholar",
        abstract="A long abstract about synthesis methods for zinc oxide nanoparticles.",
        citation_count=12,
    )

    unique = deduplicator.deduplicate([primary, richer])

    assert len(unique) == 1
    assert unique[0].doi == "10.1/abc"
    assert unique[0].abstract == richer.abstract
    assert unique[0].source_db == "semantic_scholar+openalex"
    assert deduplicator.duplicate_count() == 1


@pytest.mark.asyncio
async def test_ranker_scores_experimental_open_recent_title_match_highest() -> None:
    """Ranking should prefer papers that are extractable and compound-specific."""

    parser = FallbackQueryParser()
    plan = await parser.parse("synthesis of zinc oxide nanoparticles", ["pubmed"])
    ranker = ChemistryRelevanceRanker()
    strong = PaperMetadata(
        title="Synthesis of zinc oxide nanoparticles",
        source_db="pubmed",
        year=2026,
        open_access=True,
        citation_count=100,
        has_experimental_section=True,
    )
    weak = PaperMetadata(title="General oxide review", source_db="openalex", year=1999)

    ranked = ranker.rank([weak, strong], plan, top_k=2)

    assert ranked[0].title == strong.title
    assert ranked[0].relevance_score > ranked[1].relevance_score


def test_utils_are_deterministic_and_chemistry_aware() -> None:
    """Utility functions should be pure, stable, and chemistry-specific."""

    text = "ZnO synthesis used 5 mmol zinc acetate dissolved in 20 mL of ethanol and was heated at 80 °C for 2 h."

    assert calculate_chemical_density(text) > 0.3
    assert detect_experimental_section(text)
    assert generate_chunk_id("doi:10.1/abc", "Methods") == generate_chunk_id("doi:10.1/abc", "methods")
