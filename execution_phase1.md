# ChemExtract AI — Phase 1: Paper Fetching & Text Acquisition

## Overview

Phase 1 is the data ingestion layer of ChemExtract AI. It is responsible for:
1. Accepting a chemistry query from the user
2. Firing parallel async searches across multiple academic databases
3. Deduplicating results across sources
4. Ranking papers by relevance and text availability
5. Acquiring the best possible text for each paper (full text > abstract)
6. Returning a unified list of TextChunk objects ready for the LangGraph pipeline

This layer is built on abstract base classes throughout. Every fetcher, every
parser, every ranker implements a defined interface. Swapping PubMed for 
ChemRxiv, or adding a new source, requires zero changes to any other file.
Only a new class that inherits from BaseFetcher needs to be written.

---

## Architecture Principle: Abstract-First Design

Every component in Phase 1 follows this contract:

    BaseFetcher (abstract)
        ├── PubMedFetcher
        ├── ArXivFetcher
        ├── OpenAlexFetcher
        └── SemanticScholarFetcher
        └── [Any future fetcher inherits here, nothing else changes]

    BaseTextAcquirer (abstract)
        ├── PubMedCentralXMLAcquirer
        ├── ArXivHTMLAcquirer
        ├── OpenAccessPDFAcquirer
        └── AbstractOnlyAcquirer

    BaseRanker (abstract)
        └── ChemistryRelevanceRanker

    BaseDeduplicator (abstract)
        └── DOITitleDeduplicator

The orchestrator (FetchOrchestrator) knows only about the abstract interfaces.
It never imports a concrete class directly. Concrete classes are registered
at startup via a registry pattern. This means:
- Adding a new source = write one new class + register it
- Removing a source = unregister it
- Testing = swap real fetchers with mock fetchers trivially

---

## File Structure
chemextract-ai/
├── src/
│ ├── init.py
│ ├── schemas/
│ │ ├── init.py
│ │ ├── paper.py # PaperMetadata, TextChunk, FetchResult
│ │ └── query.py # ChemistryQuery, SearchPlan
│ │
│ └── ingestion/
│ ├── init.py
│ ├── base.py # ALL abstract base classes live here
│ ├── orchestrator.py # FetchOrchestrator — coordinates everything
│ ├── registry.py # Fetcher registry — register/lookup fetchers
│ ├── deduplicator.py # DOITitleDeduplicator
│ ├── ranker.py # ChemistryRelevanceRanker
│ ├── query_parser.py # LLM → structured SearchPlan
│ │
│ └── fetchers/
│ ├── init.py
│ ├── pubmed.py # PubMedFetcher + PubMedCentralAcquirer
│ ├── arxiv.py # ArXivFetcher + ArXivHTMLAcquirer
│ ├── openalex.py # OpenAlexFetcher + OpenAccessAcquirer
│ └── semantic_scholar.py # SemanticScholarFetcher
│
├── tests/
│ └── phase1/
│ ├── run_phase1_demo.py # THE VERBAT TEST — run this to verify phase 1
│ ├── test_individual_fetchers.py
│ └── mock_fetchers.py # Mock implementations for unit testing
│
├── data/
│ └── cache/ # API responses cached here to save quota
│
├── .env.example
├── requirements.txt

---