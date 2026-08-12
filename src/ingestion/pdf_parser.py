"""PDF upload parser for Phase 6B user-provided inputs."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Optional

from src.ingestion.base import BaseTextInputProcessor, InputSourceType, TextInputResult
from src.ingestion.utils import calculate_chemical_density, generate_chunk_id
from src.schemas.paper import AcquisitionMethod, PaperMetadata, TextChunk, TextCompleteness

logger = logging.getLogger(__name__)


class PDFParser(BaseTextInputProcessor):
    """Convert uploaded PDF files into Phase 1-compatible TextChunks."""

    processor_name = "pdf_parser"
    supported_input_type = InputSourceType.PDF_UPLOAD
    MAX_CHUNK_WORDS = 1500
    CHUNK_OVERLAP_WORDS = 150
    MAX_CHUNKS_TOTAL = 10

    def __init__(self, max_chunks: int = MAX_CHUNKS_TOTAL, min_chunk_words: int = 50) -> None:
        """Configure chunk limits for parsed PDF content."""

        self.max_chunks = max_chunks
        self.min_chunk_words = min_chunk_words

    async def can_process(self, source: str) -> bool:
        """Return whether source is a non-empty PDF file."""

        path = Path(source)
        return path.exists() and path.suffix.lower() == ".pdf" and path.stat().st_size > 0

    async def process(self, source: str, title_hint: Optional[str] = None) -> TextInputResult:
        """Convert a PDF file to TextChunks using pymupdf4llm with pdfplumber fallback."""

        start_time = time.time()
        pdf_path = Path(source)
        result = TextInputResult(source_type=InputSourceType.PDF_UPLOAD, source_identifier=pdf_path.name)
        try:
            if not await self.can_process(source):
                result.success = False
                result.error = "Input is not a readable PDF file"
                result.error_type = "INVALID_PDF"
                return result

            markdown_text = await self._convert_with_pymupdf4llm(pdf_path)
            if not markdown_text:
                logger.warning("pymupdf4llm returned empty for %s. Trying pdfplumber fallback.", pdf_path.name)
                markdown_text = await self._convert_with_pdfplumber(pdf_path)
            if not markdown_text:
                result.success = False
                result.error = "Could not extract text from PDF"
                result.error_type = "EXTRACTION_FAILED"
                return result

            result.total_pages = await self._get_page_count(pdf_path)
            result.total_words_extracted = len(markdown_text.split())
            sections = self._find_experimental_sections(markdown_text)
            result.sections_found = [section[0] for section in sections]
            if not sections:
                sections = self._best_available_chunks(markdown_text)

            paper_meta = self._build_paper_metadata(pdf_path, title_hint)
            chunks = []
            for section_name, section_text in sections[: self.max_chunks]:
                words = len(section_text.split())
                if words < self.min_chunk_words:
                    continue
                chunks.append(
                    TextChunk(
                        chunk_id=generate_chunk_id(f"pdf::{pdf_path.name}", section_name),
                        paper_metadata=paper_meta,
                        text=section_text,
                        section_name=section_name,
                        completeness=TextCompleteness.FULL_TEXT,
                        acquisition_method=AcquisitionMethod.USER_UPLOAD,
                        chemical_density_score=calculate_chemical_density(section_text),
                        word_count=words,
                    )
                )
            result.chunks = chunks
            result.words_in_chunks = sum(chunk.word_count for chunk in chunks)
            logger.info("PDF parsed: %s -> %s chunks from %s pages", pdf_path.name, len(chunks), result.total_pages)
            return result
        except Exception as exc:
            logger.error("PDF parsing failed for %s: %s", pdf_path.name, exc)
            result.success = False
            result.error = str(exc)
            result.error_type = "UNEXPECTED_ERROR"
            return result
        finally:
            result.processing_seconds = time.time() - start_time

    async def _convert_with_pymupdf4llm(self, pdf_path: Path) -> Optional[str]:
        """Convert PDF to markdown with pymupdf4llm in an executor."""

        try:
            import pymupdf4llm

            loop = asyncio.get_event_loop()
            markdown = await loop.run_in_executor(None, pymupdf4llm.to_markdown, str(pdf_path))
            return markdown if markdown else None
        except ImportError:
            logger.warning("pymupdf4llm not installed. Falling back to pdfplumber.")
            return None
        except Exception as exc:
            logger.warning("pymupdf4llm conversion failed: %s", exc)
            return None

    async def _convert_with_pdfplumber(self, pdf_path: Path) -> Optional[str]:
        """Fallback PDF text extraction with pdfplumber in an executor."""

        try:
            import pdfplumber

            def extract() -> str:
                text_parts = []
                with pdfplumber.open(str(pdf_path)) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                return "\n\n".join(text_parts)

            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, extract)
            return text if text else None
        except ImportError:
            logger.error("pdfplumber not installed either. Cannot parse PDF.")
            return None
        except Exception as exc:
            logger.error("pdfplumber fallback failed: %s", exc)
            return None

    async def _get_page_count(self, pdf_path: Path) -> int:
        """Return PDF page count without requiring a specific parser."""

        try:
            import pymupdf

            doc = pymupdf.open(str(pdf_path))
            count = len(doc)
            doc.close()
            return count
        except Exception:
            try:
                import pdfplumber

                with pdfplumber.open(str(pdf_path)) as pdf:
                    return len(pdf.pages)
            except Exception:
                return 0

    def _find_experimental_sections(self, text: str) -> list[tuple[str, str]]:
        """Find experimental or methods sections in markdown/plain text."""

        patterns = [
            re.compile(
                r"^#{1,4}\s*(experimental\s*(?:section|methods?|procedures?|details?)?|"
                r"materials?\s+and\s+methods?|synthesis\s+(?:of\s+\w+|procedures?)?|"
                r"preparation\s+of|synthetic\s+procedures?)\s*$",
                re.IGNORECASE | re.MULTILINE,
            ),
            re.compile(
                r"\*{1,2}(experimental\s*(?:section|methods?)?|materials?\s+and\s+methods?|"
                r"synthesis\s+of\s+\w+)\*{1,2}",
                re.IGNORECASE,
            ),
            re.compile(
                r"^(EXPERIMENTAL(?:\s+SECTION)?|MATERIALS\s+AND\s+METHODS|"
                r"SYNTHESIS\s+OF\s+\w+|SYNTHETIC\s+PROCEDURE)\s*$",
                re.MULTILINE,
            ),
            re.compile(
                r"^\d+\.?\s+(experimental\s*(?:section|methods?)?|materials?\s+and\s+methods?|"
                r"synthesis\s+of)\s*$",
                re.IGNORECASE | re.MULTILINE,
            ),
            re.compile(
                r"^\d+\.\d+\.?\s+(synthesis|preparation|fabrication|"
                r"nanoparticle|experimental|materials?)\b",
                re.IGNORECASE | re.MULTILINE,
            ),
            re.compile(
                r"\*{2}(\d+\.\d+\.?\s+(?:synthesis|preparation|"
                r"fabrication|experimental).*?)\*{2}",
                re.IGNORECASE,
            ),
        ]
        sections = []
        lines = text.split("\n")
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            matched_name = None
            for pattern in patterns:
                if pattern.search(line):
                    matched_name = line.strip("#").strip("*").strip()
                    break
            if matched_name:
                section_lines = [line]
                next_index = index + 1
                while next_index < len(lines):
                    next_line = lines[next_index].strip()
                    is_header = (
                        re.match(r"^#{1,3}\s+\w", next_line)
                        or re.match(r"^\d+\.\s+[A-Z]", next_line)
                        or re.match(r"^\d+\.\d+\.?\s+[A-Z]", next_line)
                        or re.match(r"^\*{2}\d+\.\d+\.?\s+", next_line)
                    )
                    is_subsection = (
                        re.match(r"^\d+\.\d+\.?\s+[A-Z]", next_line)
                        or re.match(r"^\*{2}\d+\.\d+\.?\s+", next_line)
                    )
                    if is_subsection or (is_header and next_index > index + 3):
                        break
                    section_lines.append(lines[next_index])
                    next_index += 1
                section_text = "\n".join(section_lines).strip()
                for sub_name, sub_text in self._split_if_too_long(
                    matched_name,
                    section_text,
                    max_words=self.MAX_CHUNK_WORDS,
                ):
                    if len(sub_text.split()) >= self.min_chunk_words:
                        sections.append((sub_name, sub_text, calculate_chemical_density(sub_text)))
                index = next_index
            else:
                index += 1
        sections.sort(key=lambda section: section[2], reverse=True)
        return [(name, section_text) for name, section_text, _score in sections]

    def _best_available_chunks(self, text: str) -> list[tuple[str, str]]:
        """Return the most chemistry-dense paragraphs when no section is found."""

        paragraphs = re.split(r"\n{2,}", text)
        scored = []
        for index, paragraph in enumerate(paragraphs):
            for sub_name, sub_text in self._split_if_too_long(
                f"paragraph_{index}",
                paragraph,
                max_words=self.MAX_CHUNK_WORDS,
            ):
                if len(sub_text.split()) < self.min_chunk_words:
                    continue
                score = calculate_chemical_density(sub_text)
                if score > 0.1:
                    scored.append((sub_name, sub_text, score))
        scored.sort(key=lambda item: item[2], reverse=True)
        return [(name, paragraph) for name, paragraph, _score in scored[: self.max_chunks]]

    def _split_if_too_long(
        self,
        section_name: str,
        section_text: str,
        max_words: int = MAX_CHUNK_WORDS,
        overlap_words: int = CHUNK_OVERLAP_WORDS,
    ) -> list[tuple[str, str]]:
        """Split an oversized section into overlapping word-bounded chunks."""

        words = section_text.split()
        if len(words) <= max_words:
            return [(section_name, section_text)]

        logger.info(
            "Section '%s' has %s words. Splitting into chunks.",
            section_name,
            len(words),
        )
        result = []
        start = 0
        part = 1
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_name = section_name if part == 1 else f"{section_name}_part_{part}"
            result.append((chunk_name, " ".join(words[start:end])))
            if end >= len(words):
                break
            start = end - overlap_words
            part += 1
        return result

    def _build_paper_metadata(self, pdf_path: Path, title_hint: Optional[str]) -> PaperMetadata:
        """Build minimal metadata for a user-uploaded PDF."""

        return PaperMetadata(
            title=title_hint or pdf_path.stem.replace("_", " ").replace("-", " "),
            doi=None,
            authors=[],
            year=None,
            abstract=None,
            source_db="user_upload",
            open_access=True,
        )
