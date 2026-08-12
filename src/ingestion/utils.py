"""Pure ingestion helpers for chemistry-specific text normalization.

This file contains calculate_chemical_density, generate_chunk_id,
detect_experimental_section, and normalize_title. The helpers are side-effect
free so fetchers, rankers, and tests can reuse deterministic behavior without
network or registry dependencies.
"""

from __future__ import annotations

import hashlib
import logging
import re
import string

logger = logging.getLogger(__name__)


CHEMISTRY_WORDS = {
    "methanol",
    "ethanol",
    "water",
    "acetone",
    "chloroform",
    "toluene",
    "hexane",
    "dmso",
    "dmf",
    "thf",
    "acetonitrile",
    "sodium",
    "potassium",
    "calcium",
    "magnesium",
    "iron",
    "zinc",
    "titanium",
    "copper",
    "silver",
    "gold",
    "platinum",
    "hydroxide",
    "chloride",
    "sulfate",
    "nitrate",
    "acetate",
    "oxide",
    "nanoparticle",
    "precursor",
    "synthesis",
}

EXPERIMENTAL_SIGNALS = [
    "synthesis of",
    "prepared by",
    "dissolved in",
    "was added",
    "was heated",
    "experimental section",
    "materials and methods",
    "synthetic procedure",
    "was stirred",
    "was filtered",
    "refluxed",
    "mmol",
    "mL of",
    "g of",
    "°C for",
]


def calculate_chemical_density(text: str) -> float:
    """Score chemistry density so ranking can prefer extractable procedures."""

    if not text or not text.strip():
        return 0.0
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    if word_count == 0:
        return 0.0

    patterns = [
        (r"\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\b", 0),
        (r"\b\d+\.?\d*\s*(?:mg|g|kg|mL|L|mmol|mol|M|wt%|vol%)\b", re.IGNORECASE),
        (r"\b\d+\s*°C\b", re.IGNORECASE),
        (r"\b(?:dissolved|stirred|heated|refluxed|filtered|washed|dried|added|mixed)\b", re.IGNORECASE),
        (r"\b(?:solution|mixture|suspension|precipitate|catalyst|solvent|reactant)\b", re.IGNORECASE),
    ]
    pattern_hits = sum(len(re.findall(pattern, text, flags=flags)) for pattern, flags in patterns)
    lowered_words = [word.lower() for word in words]
    word_hits = sum(1 for word in lowered_words if word in CHEMISTRY_WORDS)
    score = (pattern_hits * 0.3 + word_hits * 0.5) / word_count * 100
    return min(score, 1.0)


CHEMISTRY_UNITS = {
    "mg", "g", "kg", "ml", "mm", "mmol", "mol", "nm", "µm",
    "wt%", "vol%", "°c", "molar", "litre", "liter", "gram", "milligram",
}
CHEMISTRY_VERBS = {
    "dissolved", "stirred", "heated", "refluxed", "filtered", "washed", "dried",
    "calcined", "synthesized", "prepared", "added", "mixed", "centrifuged",
    "annealed", "deposited",
}
CHEMISTRY_NOUNS = {
    "solution", "mixture", "precipitate", "suspension", "catalyst", "solvent",
    "reactant", "precursor", "nanoparticle", "synthesis", "reaction", "compound",
    "electrode", "electrolyte", "polymer", "reagent",
}
NON_CHEMISTRY = {
    "neutrino", "muon", "boson", "quark", "telescope", "luminosity", "galaxy",
    "algorithm", "neural", "dataset", "pixel", "spacecraft", "orbital", "astronomical",
}
FORMULA_RE = re.compile(r"\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+\b")


def is_chemistry_text(text: str, threshold: float = 0.3) -> bool:
    """Return whether a sufficiently long chunk contains chemistry signals."""

    words = text.lower().split()
    if len(words) < 20:
        return False
    word_set = set(re.findall(r"[a-zA-Zµ°%]+", text.lower()))
    unit_hits = len(word_set & CHEMISTRY_UNITS) * 2
    verb_hits = len(word_set & CHEMISTRY_VERBS) * 2
    noun_hits = len(word_set & CHEMISTRY_NOUNS)
    formula_hits = len(FORMULA_RE.findall(text)) * 1.5
    non_chemistry_hits = len(word_set & NON_CHEMISTRY)
    score = (unit_hits + verb_hits + noun_hits + formula_hits) / len(words) * 100
    if non_chemistry_hits > 3 and score < 1.0:
        return False
    return score >= threshold


def classify_chunk_domain(text: str) -> str:
    """Classify a chunk as chemistry, physics, biology, computer science, or unknown."""

    domain_signals = {
        "chemistry": CHEMISTRY_UNITS | CHEMISTRY_VERBS | CHEMISTRY_NOUNS,
        "physics": {"quantum", "particle", "boson", "neutrino", "muon", "relativity", "momentum"},
        "biology": {"cell", "protein", "gene", "dna", "rna", "enzyme", "bacteria", "organism"},
        "cs": {"algorithm", "neural", "dataset", "model", "classifier", "pixel", "embedding"},
    }
    words = set(re.findall(r"[a-zA-Zµ°%]+", text.lower()))
    scores = {domain: len(words & signals) for domain, signals in domain_signals.items()}
    domain, score = max(scores.items(), key=lambda item: item[1])
    return domain if score else "unknown"


def generate_chunk_id(paper_key: str, section: str) -> str:
    """Generate stable chunk IDs so repeated runs do not duplicate text chunks."""

    combined = f"{paper_key}::{section.lower().strip()}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def detect_experimental_section(text: str) -> bool:
    """Detect procedural language that indicates synthesis or methods content."""

    lowered = text.lower()
    return any(signal.lower() in lowered for signal in EXPERIMENTAL_SIGNALS)


def normalize_title(title: str) -> str:
    """Normalize titles for fuzzy matching without destroying word boundaries."""

    translation = str.maketrans({character: " " for character in string.punctuation})
    normalized = title.lower().translate(translation)
    return re.sub(r"\s+", " ", normalized).strip()
