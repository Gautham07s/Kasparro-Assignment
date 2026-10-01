"""
Deterministic fallback analyzer.

Produces the same ResumeAnalysis structure as the LLM, using keyword
and context heuristics on sections. Used when:
- --no-llm / use_llm=false
- No API key configured
- An LLM call fails for a specific resume (increments llm_fallbacks)

Sets analysis_source="fallback".
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..config import Settings
from ..models import (
    ProjectAnalysis,
    ResumeAnalysis,
    SkillEvidence,
)
from ..parsing import split_sections

logger = logging.getLogger(__name__)


# ── Feature keyword maps ────────────────────────────────────

_FEATURE_KEYWORDS: dict[str, list[str]] = {
    "retrieval": ["rag", "retrieval", "vector search", "vector store", "faiss",
                   "chroma", "pinecone", "weaviate", "qdrant", "semantic search",
                   "document search", "embedding search", "vector database"],
    "tool_calling": ["tool calling", "function calling", "tool use", "tool execution",
                      "function execution", "api integration", "dynamic api"],
    "state_orchestration": ["state", "orchestration", "workflow", "langgraph",
                             "multi-step", "agent workflow", "state machine",
                             "multi-agent", "pipeline orchestration"],
    "evaluation": ["evaluation", "eval", "metrics", "ragas", "faithfulness",
                    "relevancy", "benchmark", "quality assessment", "model evaluation"],
    "data_processing": ["data processing", "data pipeline", "etl", "data ingestion",
                         "data transformation", "preprocessing", "data cleaning"],
    "business_logic": ["business logic", "business rule", "domain logic",
                        "custom logic", "rule engine", "decision engine"],
    "backend_api": ["api", "endpoint", "rest", "fastapi", "flask", "django",
                     "backend", "server", "microservice"],
    "deployment": ["deploy", "docker", "kubernetes", "cloud run", "ec2",
                    "heroku", "production", "ci/cd", "github actions"],
}

_PROJECT_CLASSIFICATION_KEYWORDS = {
    "substantive": ["retrieval", "tool calling", "state", "orchestration",
                     "multi-agent", "evaluation", "pipeline", "business logic",
                     "langgraph", "crewai", "workflow engine"],
    "basic": ["rag", "chatbot", "data processing", "backend", "api",
              "question answering", "document", "search"],
    "thin_wrapper": ["wrapper", "streamlit", "gradio", "simple chat",
                      "basic chatbot", "openai api"],
    "tutorial_style": ["tutorial", "demo", "example", "sample", "starter",
                        "template", "boilerplate", "hello world"],
}

# Skill detection patterns
_SKILL_PATTERNS: dict[str, dict[str, list[str]]] = {
    "python_backend": {
        "python": ["python", "python3"],
        "fastapi": ["fastapi", "fast api"],
        "async": ["async", "asyncio", "await", "async/await", "asynchronous"],
        "postgresql": ["postgresql", "postgres", "psql"],
        "redis": ["redis"],
        "flask_django": ["flask", "django"],
    },
    "cloud": {
        "gcp": ["gcp", "google cloud", "cloud run", "cloud functions",
                 "bigquery", "vertex ai", "google cloud platform"],
        "other_cloud": ["aws", "amazon web services", "ec2", "s3", "lambda",
                         "azure", "microsoft azure", "sagemaker"],
        "docker": ["docker", "dockerfile", "docker-compose", "containerize"],
        "deployment": ["deploy", "deployed", "production", "ci/cd", "github actions",
                        "heroku", "vercel", "netlify", "cloud run"],
        "react_nextjs": ["react", "reactjs", "next.js", "nextjs"],
    },
    "engineering_signals": {
        "testing": ["test", "pytest", "unittest", "jest", "coverage", "tdd",
                     "test suite", "integration test", "unit test"],
        "architecture": ["architecture", "design pattern", "microservice",
                          "clean architecture", "modular", "separation of concerns"],
        "caching": ["cache", "caching", "redis cache", "memoize", "lru_cache"],
        "queues": ["queue", "celery", "rabbitmq", "kafka", "background job",
                    "task queue", "message queue", "worker"],
        "observability": ["logging", "monitoring", "prometheus", "grafana",
                           "observability", "metrics", "tracing", "structured logging"],
        "concurrency": ["concurrent", "concurrency", "threading", "multiprocessing",
                          "asyncio", "parallel", "worker pool", "thread pool"],
        "failure_handling": ["retry", "circuit breaker", "fallback", "error handling",
                              "graceful degradation", "fault tolerant", "resilience",
                              "exception handling", "tenacity"],
    },
}

# Ownership evidence keywords
_OWNERSHIP_KEYWORDS = [
    "built", "developed", "designed", "created", "implemented", "architected",
    "led", "owned", "responsible for", "from scratch", "authored",
]


def _check_skill_in_text(text: str, keywords: list[str]) -> bool:
    """Check if any keyword matches in text."""
    text_lower = text.lower()
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            return True
    return False


def _find_quote(text: str, keywords: list[str], max_len: int = 120) -> Optional[str]:
    """Find a verbatim snippet containing the keyword."""
    text_lower = text.lower()
    for kw in keywords:
        match = re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
        if match:
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            snippet = text[start:end].replace("\n", " ").strip()
            return snippet[:max_len]
    return None


def _classify_project(text: str) -> str:
    """Classify a project section text."""
    text_lower = text.lower()

    # Check for substantive indicators
    substantive_count = sum(
        1 for kw in _PROJECT_CLASSIFICATION_KEYWORDS["substantive"]
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
    )
    if substantive_count >= 2:
        return "substantive"

    basic_count = sum(
        1 for kw in _PROJECT_CLASSIFICATION_KEYWORDS["basic"]
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
    )

    thin_count = sum(
        1 for kw in _PROJECT_CLASSIFICATION_KEYWORDS["thin_wrapper"]
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
    )

    tutorial_count = sum(
        1 for kw in _PROJECT_CLASSIFICATION_KEYWORDS["tutorial_style"]
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
    )

    # If project text has detail (multiple lines, technical terms)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    has_detail = len(lines) >= 3

    if substantive_count >= 1 and has_detail:
        return "substantive"
    if basic_count >= 2 and has_detail:
        return "basic"
    if tutorial_count >= 1:
        return "tutorial_style"
    if thin_count >= 1:
        return "thin_wrapper"
    if basic_count >= 1:
        return "basic" if has_detail else "thin_wrapper"

    return "tutorial_style"


def _extract_projects(text: str, sections: dict[str, str]) -> list[ProjectAnalysis]:
    """Extract project analyses from resume sections."""
    projects: list[ProjectAnalysis] = []

    # Look in project sections or the full text
    project_text = sections.get("projects", "")
    if not project_text:
        project_text = text

    # Try to split project text into individual projects
    # Look for lines that seem like project titles (short, possibly with technologies in parens)
    project_blocks: list[tuple[str, str]] = []
    current_name = ""
    current_lines: list[str] = []

    for line in project_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this line looks like a project title
        # Heuristic: short line, not starting with -, may contain parens
        is_title = (
            len(stripped) < 100
            and not stripped.startswith("-")
            and not stripped.startswith("•")
            and not stripped.startswith("*")
            and (
                "(" in stripped
                or len(stripped.split()) <= 8
            )
            and not stripped.startswith("Technologies:")
            and not stripped.startswith("Tech:")
        )

        if is_title and current_lines:
            # Save previous project
            project_blocks.append((current_name, "\n".join(current_lines)))
            current_name = stripped
            current_lines = [stripped]
        elif is_title and not current_lines:
            current_name = stripped
            current_lines = [stripped]
        else:
            current_lines.append(stripped)

    if current_lines:
        project_blocks.append((current_name, "\n".join(current_lines)))

    # If we couldn't split projects, treat the whole section as one
    if not project_blocks:
        project_blocks = [("Project", project_text)]

    for name, block in project_blocks:
        block_lower = block.lower()

        # Classify
        classification = _classify_project(block)

        # Check features
        features: dict[str, bool] = {}
        for feature_name, keywords in _FEATURE_KEYWORDS.items():
            features[feature_name] = _check_skill_in_text(block, keywords)

        # Ownership evidence
        ownership = _check_skill_in_text(block, _OWNERSHIP_KEYWORDS)

        # Evidence quote
        evidence_quote = ""
        for feat, present in features.items():
            if present:
                quote = _find_quote(block, _FEATURE_KEYWORDS[feat])
                if quote:
                    evidence_quote = quote
                    break
        if not evidence_quote:
            # Use first meaningful line
            for line in block.split("\n"):
                line = line.strip()
                if line and len(line) > 20:
                    evidence_quote = line[:120]
                    break

        projects.append(ProjectAnalysis(
            name=name[:100] if name else "Unnamed Project",
            summary=block[:200].replace("\n", " "),
            classification=classification,
            features=features,
            ownership_evidence=ownership,
            evidence_quote=evidence_quote,
        ))

    return projects


def _build_skill_evidence(
    text: str,
    sections: dict[str, str],
    category: str,
) -> dict[str, SkillEvidence]:
    """Build SkillEvidence dict for a skill category."""
    result: dict[str, SkillEvidence] = {}
    patterns = _SKILL_PATTERNS.get(category, {})

    # Text in project/work context
    project_work_text = ""
    for section_name in ["projects", "experience", "internship"]:
        if section_name in sections:
            project_work_text += " " + sections[section_name]

    for skill_name, keywords in patterns.items():
        present = _check_skill_in_text(text, keywords)
        in_project = _check_skill_in_text(project_work_text, keywords) if project_work_text else False
        quote = _find_quote(text, keywords) if present else None

        result[skill_name] = SkillEvidence(
            present=present,
            in_project_or_work=in_project,
            quote=quote,
        )

    return result


def _extract_strengths_concerns(
    projects: list[ProjectAnalysis],
    skills: dict[str, dict[str, SkillEvidence]],
) -> tuple[list[str], list[str]]:
    """Generate strengths and concerns lists."""
    strengths: list[str] = []
    concerns: list[str] = []

    # Check project quality
    best = None
    for p in projects:
        if best is None or _tier_rank(p.classification) > _tier_rank(best.classification):
            best = p

    if best:
        if best.classification == "substantive":
            strengths.append(f"Strong AI project: {best.name}")
        elif best.classification == "basic":
            strengths.append(f"Has working AI/data project: {best.name}")
        elif best.classification == "thin_wrapper":
            concerns.append("AI project is a thin API wrapper with limited depth")
        elif best.classification == "tutorial_style":
            concerns.append("AI project appears tutorial-style with limited originality")

    # Check backend skills
    pb = skills.get("python_backend", {})
    if pb.get("python", SkillEvidence()).in_project_or_work:
        strengths.append("Python used in project/work context")
    if pb.get("fastapi", SkillEvidence()).in_project_or_work:
        strengths.append("FastAPI experience in projects")
    if pb.get("async", SkillEvidence()).present:
        strengths.append("Async programming experience")

    # Check cloud
    cloud = skills.get("cloud", {})
    if cloud.get("docker", SkillEvidence()).in_project_or_work:
        strengths.append("Docker deployment experience")

    # Concerns
    if not pb.get("postgresql", SkillEvidence()).present and not pb.get("redis", SkillEvidence()).present:
        concerns.append("No database/caching experience evident")

    eng = skills.get("engineering_signals", {})
    if not eng.get("testing", SkillEvidence()).present:
        concerns.append("No testing evidence")

    return strengths[:5], concerns[:5]  # Cap at 5 each


def _tier_rank(classification: str) -> int:
    return {"substantive": 4, "basic": 3, "thin_wrapper": 2, "tutorial_style": 1}.get(classification, 0)


def analyze_resume_fallback(raw_text: str, settings: Settings) -> ResumeAnalysis:
    """
    Deterministic resume analysis using keyword/context heuristics.

    Returns the same ResumeAnalysis structure as the LLM, with
    analysis_source="fallback".
    """
    sections = split_sections(raw_text)

    # Extract projects
    projects = _extract_projects(raw_text, sections)

    # Build skill evidence
    python_backend = _build_skill_evidence(raw_text, sections, "python_backend")
    cloud = _build_skill_evidence(raw_text, sections, "cloud")
    engineering_signals = _build_skill_evidence(raw_text, sections, "engineering_signals")

    all_skills = {
        "python_backend": python_backend,
        "cloud": cloud,
        "engineering_signals": engineering_signals,
    }

    strengths, concerns = _extract_strengths_concerns(projects, all_skills)

    return ResumeAnalysis(
        projects=projects,
        python_backend=python_backend,
        cloud=cloud,
        engineering_signals=engineering_signals,
        strengths=strengths,
        concerns=concerns,
        analysis_source="fallback",
    )
