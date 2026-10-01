"""
Resume ingestion: folder scanning, PDF/DOCX/TXT reading, deduplication.

Every per-file error is caught and recorded as FailedResume — never raised.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Union

from .models import DuplicateInfo, FailedResume, ParsedResume

logger = logging.getLogger(__name__)

# Supported file extensions
_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Ligature replacements
_LIGATURES = {
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb00": "ff",
}


def _normalize_text(text: str) -> str:
    """Collapse whitespace, fix ligatures, keep newlines for section splitting."""
    for lig, replacement in _LIGATURES.items():
        text = text.replace(lig, replacement)
    # Collapse multiple spaces/tabs within a line but keep newlines
    lines = text.split("\n")
    normalized_lines = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def _read_pdf(file_path: Path) -> tuple[str, list[str]]:
    """
    Extract text and hyperlink URIs from a PDF.

    Strategy: PyMuPDF first; fall back to pdfplumber if text < 100 chars.
    Returns (text, links).
    """
    import fitz  # PyMuPDF

    text_parts: list[str] = []
    links: list[str] = []

    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF: {exc}") from exc

    try:
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text_parts.append(page_text)
            # Collect hyperlink URIs (GitHub URLs are often hidden in annotations)
            for link in page.get_links():
                uri = link.get("uri", "")
                if uri:
                    links.append(uri)
    finally:
        doc.close()

    full_text = "\n".join(text_parts)

    # If PyMuPDF yields very little text, try pdfplumber
    if len(full_text.strip()) < 100:
        logger.debug("PyMuPDF extracted < 100 chars for %s, trying pdfplumber", file_path.name)
        try:
            import pdfplumber

            with pdfplumber.open(str(file_path)) as pdf:
                plumber_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        plumber_parts.append(page_text)
                plumber_text = "\n".join(plumber_parts)
                if len(plumber_text.strip()) > len(full_text.strip()):
                    full_text = plumber_text
        except Exception:
            logger.debug("pdfplumber also failed for %s", file_path.name)

    return full_text, links


def _read_docx(file_path: Path) -> tuple[str, list[str]]:
    """Extract text from a DOCX file."""
    from docx import Document

    doc = Document(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs), []


def _read_txt(file_path: Path) -> tuple[str, list[str]]:
    """Read a plain text file."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return text, []


def _content_hash(text: str) -> str:
    """SHA-256 of normalized text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_resumes(
    input_dir: Path,
) -> tuple[list[ParsedResume], list[FailedResume], list[DuplicateInfo]]:
    """
    Scan input_dir (non-recursive), read supported files, deduplicate.

    Returns:
        (parsed_resumes, failed_resumes, duplicates)
    """
    from .parsing import parse_candidate_info

    parsed: list[ParsedResume] = []
    failed: list[FailedResume] = []
    duplicates: list[DuplicateInfo] = []

    # Content-hash -> first file_name for dedupe
    seen_hashes: dict[str, str] = {}
    seen_emails: dict[str, str] = {}

    # Scan and sort for deterministic order
    files = sorted(
        [f for f in input_dir.iterdir() if f.is_file() and not f.name.startswith(".")],
        key=lambda f: f.name,
    )

    for file_path in files:
        try:
            ext = file_path.suffix.lower()
            if ext not in _SUPPORTED_EXTENSIONS:
                failed.append(FailedResume(
                    file_name=file_path.name,
                    reason=f"unsupported file type: {ext}",
                    stage="read",
                ))
                continue

            # Read file
            reader = {".pdf": _read_pdf, ".docx": _read_docx, ".txt": _read_txt}[ext]
            try:
                raw_text, links = reader(file_path)
            except Exception as exc:
                failed.append(FailedResume(
                    file_name=file_path.name,
                    reason=str(exc),
                    stage="read",
                ))
                continue

            # Normalize
            normalized = _normalize_text(raw_text)

            # Check for empty / scanned PDF
            if len(normalized.strip()) < 50:
                failed.append(FailedResume(
                    file_name=file_path.name,
                    reason="no extractable text (scanned or empty PDF)",
                    stage="read",
                ))
                continue

            # Content hash dedupe
            chash = _content_hash(normalized)
            if chash in seen_hashes:
                duplicates.append(DuplicateInfo(
                    file_name=file_path.name,
                    duplicate_of=seen_hashes[chash],
                ))
                continue
            seen_hashes[chash] = file_path.name

            # Build ParsedResume
            resume = ParsedResume(
                file_name=file_path.name,
                file_path=str(file_path),
                content_hash=chash,
                raw_text=normalized,
                links=links,
            )

            # Parse candidate info (name, email, github)
            resume = parse_candidate_info(resume)

            # Email dedupe (after parsing)
            if resume.email:
                email_lower = resume.email.lower()
                if email_lower in seen_emails:
                    duplicates.append(DuplicateInfo(
                        file_name=file_path.name,
                        duplicate_of=seen_emails[email_lower],
                    ))
                    continue
                seen_emails[email_lower] = file_path.name

            parsed.append(resume)
            logger.info(
                "Ingested %s — name=%s, email=%s, github=%s",
                file_path.name,
                resume.candidate_name,
                resume.email,
                resume.github_username,
            )

        except Exception as exc:
            logger.error("Unexpected error processing %s: %s", file_path.name, exc)
            failed.append(FailedResume(
                file_name=file_path.name,
                reason=f"unexpected error: {exc}",
                stage="read",
            ))

    logger.info(
        "Ingestion complete: %d parsed, %d failed, %d duplicates",
        len(parsed), len(failed), len(duplicates),
    )
    return parsed, failed, duplicates
