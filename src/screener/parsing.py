"""
Resume parsing: extract candidate name, email, GitHub username,
and split resume into sections.

Eligibility NEVER depends on section detection; sections are only
used to distinguish "in project/work context" from "skills-list only".
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .models import ParsedResume

logger = logging.getLogger(__name__)

# ── Email Extraction ──────────────────────────────────────────

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)

# ── GitHub Extraction ─────────────────────────────────────────

_GITHUB_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9\-]+)(?:/[a-zA-Z0-9._\-]*)?",
    re.IGNORECASE,
)

# Reserved GitHub paths that are NOT usernames
_GITHUB_RESERVED: set[str] | None = None


def _get_reserved_paths() -> set[str]:
    """Lazy-load reserved paths from config or use defaults."""
    global _GITHUB_RESERVED
    if _GITHUB_RESERVED is not None:
        return _GITHUB_RESERVED

    _GITHUB_RESERVED = {
        "orgs", "settings", "topics", "features", "marketplace", "sponsors",
        "explore", "notifications", "new", "login", "signup", "pricing",
        "about", "enterprise", "team", "security", "customer-stories",
        "open-source", "readme", "collections", "trending", "events",
        "apps", "codespaces", "copilot", "issues", "pulls", "discussions",
        "actions", "projects", "packages", "stars", "repositories",
        "people", "organizations",
    }
    return _GITHUB_RESERVED


def _extract_github_username(
    text: str, links: list[str], reserved_paths: set[str] | None = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract GitHub username from text and link annotations.

    Strategy:
    1. Scan link annotations for github.com/<username>.
    2. Scan text for github.com/<username>.
    3. Prefer username from link annotation or near "GitHub" keyword.
    4. If multiple, prefer most frequent.

    Returns (username, url).
    """
    if reserved_paths is None:
        reserved_paths = _get_reserved_paths()

    candidates_from_links: list[str] = []
    candidates_from_text: list[str] = []
    url_map: dict[str, str] = {}  # username -> first full URL

    # Scan link annotations
    for link in links:
        match = _GITHUB_URL_RE.search(link)
        if match:
            username = match.group(1).lower()
            if username not in reserved_paths:
                candidates_from_links.append(username)
                if username not in url_map:
                    url_map[username] = match.group(0)

    # Scan text
    for match in _GITHUB_URL_RE.finditer(text):
        username = match.group(1).lower()
        if username not in reserved_paths:
            candidates_from_text.append(username)
            if username not in url_map:
                url_map[username] = match.group(0)

    # Prefer link-annotation username
    if candidates_from_links:
        username = candidates_from_links[0]
        return username, url_map.get(username)

    # Look for username near "GitHub" keyword in text
    github_context_re = re.compile(
        r"github[:\s]*(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9\-]+)",
        re.IGNORECASE,
    )
    ctx_match = github_context_re.search(text)
    if ctx_match:
        username = ctx_match.group(1).lower()
        if username not in reserved_paths:
            return username, url_map.get(username)

    # Fall back to most frequent in text
    if candidates_from_text:
        freq: dict[str, int] = {}
        for u in candidates_from_text:
            freq[u] = freq.get(u, 0) + 1
        username = max(freq, key=lambda u: freq[u])
        return username, url_map.get(username)

    return None, None


# ── Name Extraction ───────────────────────────────────────────

_NAME_SKIP_PATTERNS = re.compile(
    r"[@/\\]|http|www\.|\.com|\.org|\.net|\d{3}|\bresume\b|\bcurriculum\b|\bvitae\b|\bcv\b",
    re.IGNORECASE,
)

# Common resume headings that should NOT be treated as names
_HEADING_SKIP = {
    "professional summary", "summary", "objective", "profile", "about me",
    "career objective", "personal profile", "professional profile",
    "career summary", "executive summary", "skills", "technical skills",
    "experience", "education", "projects", "certifications", "achievements",
    "work experience", "professional experience", "contact information",
    "personal information", "personal details", "contact details",
    "contact", "about", "interests", "hobbies", "references",
}


def _extract_name(text: str, file_name: str) -> str:
    """
    Extract candidate name from the first non-empty line that looks like a name.

    Fallback: prettified filename stem.
    """
    lines = text.split("\n")
    for line in lines[:15]:  # Check first 15 lines
        line = line.strip()
        if not line or len(line) < 2:
            continue
        # Skip lines with emails, URLs, digits, etc.
        if _NAME_SKIP_PATTERNS.search(line):
            continue
        # Skip common resume headings
        if line.lower().rstrip(":").strip() in _HEADING_SKIP:
            continue
        tokens = line.split()
        if 2 <= len(tokens) <= 4 and all(
            t.replace("-", "").replace("'", "").isalpha() for t in tokens
        ):
            return line.title() if line.isupper() or line.islower() else line

    # Fallback to filename stem
    stem = file_name.rsplit(".", 1)[0]
    # Prettify: candidate_01 -> Candidate 01
    return stem.replace("_", " ").replace("-", " ").title()


# ── Section Splitting ─────────────────────────────────────────

_SECTION_HEADINGS = [
    (re.compile(r"^(?:technical\s+)?skills?\b", re.IGNORECASE), "skills"),
    (re.compile(r"^projects?\b", re.IGNORECASE), "projects"),
    (re.compile(r"^(?:work\s+)?experience\b", re.IGNORECASE), "experience"),
    (re.compile(r"^internships?\b", re.IGNORECASE), "internship"),
    (re.compile(r"^education\b", re.IGNORECASE), "education"),
    (re.compile(r"^certifications?\b", re.IGNORECASE), "certifications"),
    (re.compile(r"^summary\b|^(?:professional\s+)?profile\b|^objective\b", re.IGNORECASE), "summary"),
    (re.compile(r"^achievements?\b|^awards?\b", re.IGNORECASE), "achievements"),
    (re.compile(r"^publications?\b|^research\b", re.IGNORECASE), "research"),
]


def split_sections(text: str) -> dict[str, str]:
    """
    Split resume text into named sections by fuzzy heading detection.

    Returns a dict of section_name -> section_text.
    If detection fails, returns {"full": text}.
    """
    lines = text.split("\n")
    sections: list[tuple[str, int]] = []  # (name, start_line_index)

    for i, line in enumerate(lines):
        stripped = line.strip().rstrip(":").strip()
        if not stripped or len(stripped) > 60:
            continue
        for pattern, name in _SECTION_HEADINGS:
            if pattern.match(stripped):
                sections.append((name, i))
                break

    if not sections:
        return {"full": text}

    result: dict[str, str] = {}
    for idx, (name, start) in enumerate(sections):
        end = sections[idx + 1][1] if idx + 1 < len(sections) else len(lines)
        section_text = "\n".join(lines[start + 1: end]).strip()
        if section_text:
            result[name] = section_text

    # Include header (before first section)
    if sections[0][1] > 0:
        header = "\n".join(lines[: sections[0][1]]).strip()
        if header:
            result["header"] = header

    return result


# ── Main Parse Function ──────────────────────────────────────


def parse_candidate_info(resume: ParsedResume) -> ParsedResume:
    """
    Enrich a ParsedResume with candidate_name, email, github_username.

    Mutates and returns the same object.
    """
    # Email
    email_match = _EMAIL_RE.search(resume.raw_text)
    if email_match:
        resume.email = email_match.group(0)

    # GitHub
    username, url = _extract_github_username(resume.raw_text, resume.links)
    resume.github_username = username
    resume.github_url = url

    # Name
    resume.candidate_name = _extract_name(resume.raw_text, resume.file_name)

    return resume
