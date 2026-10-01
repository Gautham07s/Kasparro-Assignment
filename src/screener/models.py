"""
Pydantic v2 data models for every data structure crossing module boundaries.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Ingestion / Parsing ────────────────────────────────────────


class ParsedResume(BaseModel):
    """Successfully extracted resume content."""
    file_name: str
    file_path: str
    content_hash: str
    raw_text: str
    links: list[str] = Field(default_factory=list)
    candidate_name: str = ""
    email: Optional[str] = None
    github_username: Optional[str] = None
    github_url: Optional[str] = None


class FailedResume(BaseModel):
    """A resume that could not be processed."""
    file_name: str
    reason: str
    stage: str  # "read" | "parse" | ...


class DuplicateInfo(BaseModel):
    """A duplicate resume detected by content hash or email."""
    file_name: str
    duplicate_of: str


# ── Eligibility ────────────────────────────────────────────────


class EvidenceSpan(BaseModel):
    """A matched term with surrounding context."""
    term: str
    snippet: str = Field(max_length=200)  # <=160 chars around match, slight buffer
    kind: Literal["direct", "implied"]


class EligibilityResult(BaseModel):
    """Hard-filter result for a single candidate."""
    candidate: str  # file_name
    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    python_evidence: list[EvidenceSpan] = Field(default_factory=list)
    ai_evidence: list[EvidenceSpan] = Field(default_factory=list)


# ── LLM / Fallback Analysis ───────────────────────────────────


class ProjectAnalysis(BaseModel):
    """Analysis of a single project from the resume."""
    name: str = ""
    summary: str = ""
    classification: Literal["substantive", "basic", "thin_wrapper", "tutorial_style"] = "tutorial_style"
    features: dict[str, bool] = Field(default_factory=lambda: {
        "retrieval": False,
        "tool_calling": False,
        "state_orchestration": False,
        "evaluation": False,
        "data_processing": False,
        "business_logic": False,
        "backend_api": False,
        "deployment": False,
    })
    ownership_evidence: bool = False
    evidence_quote: str = ""  # MUST be verbatim from resume


class SkillEvidence(BaseModel):
    """Whether a skill is present and in what context."""
    present: bool = False
    in_project_or_work: bool = False
    quote: Optional[str] = None


class ResumeAnalysis(BaseModel):
    """Structured analysis of an eligible resume (from LLM or fallback)."""
    projects: list[ProjectAnalysis] = Field(default_factory=list)

    python_backend: dict[str, SkillEvidence] = Field(default_factory=lambda: {
        "python": SkillEvidence(),
        "fastapi": SkillEvidence(),
        "async": SkillEvidence(),
        "postgresql": SkillEvidence(),
        "redis": SkillEvidence(),
        "flask_django": SkillEvidence(),
    })

    cloud: dict[str, SkillEvidence] = Field(default_factory=lambda: {
        "gcp": SkillEvidence(),
        "other_cloud": SkillEvidence(),
        "docker": SkillEvidence(),
        "deployment": SkillEvidence(),
        "react_nextjs": SkillEvidence(),
    })

    engineering_signals: dict[str, SkillEvidence] = Field(default_factory=lambda: {
        "testing": SkillEvidence(),
        "architecture": SkillEvidence(),
        "caching": SkillEvidence(),
        "queues": SkillEvidence(),
        "observability": SkillEvidence(),
        "concurrency": SkillEvidence(),
        "failure_handling": SkillEvidence(),
    })

    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    analysis_source: Literal["llm", "fallback"] = "fallback"


# ── GitHub ─────────────────────────────────────────────────────


class GitHubSummary(BaseModel):
    """GitHub enrichment result."""
    status: Literal["ok", "no_github", "not_found", "rate_limited", "error"] = "no_github"
    username: Optional[str] = None
    activity_score: int = 0  # 0-5
    repo_score: int = 0      # 0-5
    total: int = 0           # 0-10
    recent_push_events: int = 0
    maintained_repos: int = 0
    relevant_repos: int = 0
    summary: str = ""
    error: Optional[str] = None


# ── Scoring ────────────────────────────────────────────────────


class ScoreBreakdown(BaseModel):
    """Point breakdown by category."""
    ai_project_depth: int = 0   # <= 40
    python_backend: int = 0     # <= 30
    cloud_fullstack: int = 0    # <= 15
    github: int = 0             # <= 10
    engineering_depth: int = 0  # <= 5


# ── Final Output ───────────────────────────────────────────────


class CandidateResult(BaseModel):
    """Final result for one candidate (eligible or rejected)."""
    rank: Optional[int] = None
    candidate_name: str = ""
    email: Optional[str] = None
    source_file: str = ""
    eligible: bool = False
    total_score: int = 0
    score_breakdown: Optional[ScoreBreakdown] = None
    matched_skills: list[str] = Field(default_factory=list)
    project_summary: str = ""
    github_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    analysis_source: Optional[Literal["llm", "fallback"]] = None
    penalties_applied: list[str] = Field(default_factory=list)


class BatchSummary(BaseModel):
    """Aggregate statistics for the screening run."""
    total_resumes: int = 0
    successfully_parsed: int = 0
    eligible: int = 0
    rejected: int = 0
    failed_unreadable: int = 0
    duplicates: int = 0
    llm_fallbacks: int = 0
    github_failures: int = 0


class ScreeningRun(BaseModel):
    """Complete output of a screening pipeline run."""
    batch_summary: BatchSummary = Field(default_factory=BatchSummary)
    ranked_candidates: list[CandidateResult] = Field(default_factory=list)
    rejected_candidates: list[CandidateResult] = Field(default_factory=list)
    failed_files: list[FailedResume] = Field(default_factory=list)
    duplicates: list[DuplicateInfo] = Field(default_factory=list)
