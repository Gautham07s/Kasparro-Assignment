"""
Hard eligibility filter — PURE RULES, NO LLM.

A candidate is eligible ONLY IF both conditions are met:
1. Python evidence (direct or implied)
2. AI/agentic project evidence

Classic ML/CV/NLP-only evidence (without LLM/agentic qualifiers)
does NOT satisfy the AI requirement.

All lexicons are loaded from settings.yaml. Rejected candidates
never proceed to the LLM or GitHub enrichment.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .config import Settings
from .models import EligibilityResult, EvidenceSpan, ParsedResume

logger = logging.getLogger(__name__)


def _word_boundary_pattern(term: str) -> re.Pattern:
    """Build a word-boundary regex for the given term."""
    escaped = re.escape(term)
    return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)


def _find_evidence(
    text: str,
    term: str,
    kind: str,
    max_snippet_len: int = 160,
) -> Optional[EvidenceSpan]:
    """
    Search for a term in text and return an EvidenceSpan if found.

    The snippet is a window of up to max_snippet_len chars around the match.
    """
    pattern = _word_boundary_pattern(term)
    match = pattern.search(text)
    if not match:
        return None

    start = max(0, match.start() - 60)
    end = min(len(text), match.end() + 60)
    snippet = text[start:end].replace("\n", " ").strip()
    if len(snippet) > max_snippet_len:
        snippet = snippet[:max_snippet_len]

    return EvidenceSpan(term=term, snippet=snippet, kind=kind)


def _extract_matched_skills(text: str, settings: Settings) -> list[str]:
    """
    Extract canonical skill names found anywhere in the text.

    Uses the skill_aliases map from settings to normalize names.
    Reports skills even for rejected candidates (per spec).
    """
    text_lower = text.lower()
    matched: list[str] = []

    for canonical, aliases in settings.skill_aliases.items():
        for alias in aliases:
            pattern = _word_boundary_pattern(alias)
            if pattern.search(text_lower):
                # Use the canonical name with proper casing
                display_name = canonical.replace("_", " ").title()
                # Special cases for acronyms
                display_map = {
                    "Gcp": "GCP",
                    "Aws": "AWS",
                    "Postgresql": "PostgreSQL",
                    "Fastapi": "FastAPI",
                    "Langchain": "LangChain",
                    "Langgraph": "LangGraph",
                    "Llamaindex": "LlamaIndex",
                    "Pytorch": "PyTorch",
                    "Tensorflow": "TensorFlow",
                    "Nextjs": "Next.js",
                    "Numpy": "NumPy",
                    "Pandas": "Pandas",
                    "Javascript": "JavaScript",
                    "Kubernetes": "Kubernetes",
                }
                display_name = display_map.get(display_name, display_name)
                if display_name not in matched:
                    matched.append(display_name)
                break  # Found this canonical skill, move to next

    return sorted(matched)


def check_eligibility(
    resume: ParsedResume, settings: Settings
) -> EligibilityResult:
    """
    Apply hard eligibility rules to a parsed resume.

    Returns EligibilityResult with eligible=True/False, rejection_reasons,
    matched_skills, and evidence spans.
    """
    text = resume.raw_text.lower()
    elig = settings.eligibility
    rejection_reasons: list[str] = []
    python_evidence: list[EvidenceSpan] = []
    ai_evidence: list[EvidenceSpan] = []

    # ── Python Evidence ──────────────────────────────────────

    has_python = False

    # Direct: \bpython\b (exclude jython)
    for term in elig.python_direct:
        ev = _find_evidence(text, term, "direct")
        if ev:
            # Check exclusions: "jython" context
            # Only count if it's genuinely "python", not just "jython"
            exclude = False
            for excl in elig.python_exclude:
                excl_pattern = _word_boundary_pattern(excl)
                if excl_pattern.search(text):
                    # "jython" alone is not enough to reject — only if
                    # "python" itself does not appear independently
                    # Check if there's a real \bpython\b that isn't part of jython
                    # We already found \bpython\b, so python IS present even if jython is
                    pass
            python_evidence.append(ev)
            has_python = True

    # Implied: Python ecosystem terms
    if not has_python:
        for term in elig.python_implied:
            ev = _find_evidence(text, term, "implied")
            if ev:
                python_evidence.append(ev)
                has_python = True
                break  # One implied term is sufficient

    if not has_python:
        rejection_reasons.append("No evidence of Python stack")

    # ── AI / Agentic Evidence ────────────────────────────────

    has_ai = False
    has_ml_only = False

    # Check ML-only terms first to detect the ML-only case
    for term in elig.ml_only_terms:
        pattern = _word_boundary_pattern(term)
        if pattern.search(text):
            has_ml_only = True
            break

    # Check all AI qualifying categories
    all_ai_terms = (
        [(t, "frameworks") for t in elig.ai_frameworks]
        + [(t, "concepts") for t in elig.ai_concepts]
        + [(t, "provider_apis") for t in elig.ai_provider_apis]
        + [(t, "extended") for t in elig.ai_extended]
    )

    for term, category in all_ai_terms:
        ev = _find_evidence(text, term, "direct" if category == "frameworks" else "implied")
        if ev:
            ai_evidence.append(ev)
            has_ai = True
            break  # One qualifying term is sufficient

    if not has_ai:
        if has_ml_only:
            rejection_reasons.append("ML only, no LLM/agentic evidence")
        else:
            rejection_reasons.append("No AI/agentic project evidence")

    # ── Matched Skills ──────────────────────────────────────

    matched_skills = _extract_matched_skills(resume.raw_text, settings)

    # ── Result ──────────────────────────────────────────────

    eligible = has_python and has_ai

    if eligible:
        logger.info("ELIGIBLE: %s — python=%s, ai=%s", resume.file_name, 
                     [e.term for e in python_evidence], [e.term for e in ai_evidence])
    else:
        logger.info("REJECTED: %s — reasons=%s", resume.file_name, rejection_reasons)

    return EligibilityResult(
        candidate=resume.file_name,
        eligible=eligible,
        rejection_reasons=rejection_reasons,
        matched_skills=matched_skills,
        python_evidence=python_evidence,
        ai_evidence=ai_evidence,
    )
