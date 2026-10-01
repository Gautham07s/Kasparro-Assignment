"""
Scoring module: converts ResumeAnalysis + GitHubSummary into points.

Pure functions, fully unit-testable, no I/O.
All point values come from Settings (config/settings.yaml).
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import Settings
from .models import (
    GitHubSummary,
    ProjectAnalysis,
    ResumeAnalysis,
    ScoreBreakdown,
    SkillEvidence,
)

logger = logging.getLogger(__name__)


def _tier_rank(classification: str) -> int:
    """Rank tiers for comparison."""
    return {"substantive": 4, "basic": 3, "thin_wrapper": 2, "tutorial_style": 1}.get(classification, 0)


def score_ai_project_depth(
    analysis: ResumeAnalysis,
    settings: Settings,
) -> tuple[int, list[str]]:
    """
    Score AI / Agentic / RAG Project Depth (max 40).

    Returns (score, penalties_applied).
    """
    cfg = settings.ai_project
    max_score = settings.score_weights.ai_project_depth
    penalties: list[str] = []

    if not analysis.projects:
        return 0, ["no AI projects found"]

    # Find best project by tier
    sorted_projects = sorted(
        analysis.projects,
        key=lambda p: _tier_rank(p.classification),
        reverse=True,
    )
    best = sorted_projects[0]
    second = sorted_projects[1] if len(sorted_projects) > 1 else None

    # Check if AI framework is only in skills list (no project usage)
    # This is detected when no project is classified as substantive/basic
    # AND the analysis has AI-related skills but no project features
    has_project_with_ai_features = any(
        p.classification in ("substantive", "basic")
        or any(p.features.get(f, False) for f in ["retrieval", "tool_calling", "state_orchestration", "evaluation"])
        for p in analysis.projects
    )

    if not has_project_with_ai_features:
        # Cap at skills_only_cap
        cap = cfg.skills_only_cap
        penalties.append(f"AI framework appears only in skills list, capped at {cap}")
        return min(cap, max_score), penalties

    # Tier points
    score = cfg.tier_points.get(best.classification, 0)

    # Feature checklist on best project (max 16)
    feature_score = 0
    for feature_name, points in cfg.feature_points.items():
        if best.features.get(feature_name, False):
            feature_score += points
    score += feature_score

    # Ownership bonus
    if best.ownership_evidence:
        score += cfg.ownership_bonus

    # Second project bonus
    if second and second.classification in ("substantive", "basic"):
        score += cfg.second_project_bonus

    # Penalties
    if best.classification == "thin_wrapper":
        # Count features on best project
        feature_count = sum(1 for f in best.features.values() if f)
        has_other_basic = second and second.classification in ("substantive", "basic")

        if feature_count == 0 and not has_other_basic:
            penalty = cfg.penalties.get("thin_wrapper_no_features", -15)
            score += penalty
            penalties.append(f"thin_wrapper with no features: {penalty}")
        elif feature_count >= 1 or has_other_basic:
            penalty = cfg.penalties.get("thin_wrapper_has_features", -5)
            score += penalty
            penalties.append(f"thin_wrapper (has features or other project): {penalty}")
        else:
            penalty = cfg.penalties.get("thin_wrapper_default", -10)
            score += penalty
            penalties.append(f"thin_wrapper: {penalty}")

    elif best.classification == "tutorial_style":
        if not best.ownership_evidence:
            penalty = cfg.penalties.get("tutorial_no_ownership", -8)
            score += penalty
            penalties.append(f"tutorial_style, no ownership evidence: {penalty}")
        else:
            penalty = cfg.penalties.get("tutorial_no_detail", -5)
            score += penalty
            penalties.append(f"tutorial_style: {penalty}")

    # Floor at 0
    score = max(0, score)
    # Cap at category max
    score = min(score, max_score)

    return score, penalties


def score_python_backend(
    analysis: ResumeAnalysis,
    settings: Settings,
) -> int:
    """Score Python & Backend Engineering (max 30)."""
    cfg = settings.python_backend
    max_score = settings.score_weights.python_backend
    score = 0.0

    pb = analysis.python_backend

    # Python
    python_ev = pb.get("python", SkillEvidence())
    if python_ev.present:
        score += cfg.python if python_ev.in_project_or_work else cfg.python * cfg.skills_only_ratio

    # FastAPI
    fastapi_ev = pb.get("fastapi", SkillEvidence())
    flask_django_ev = pb.get("flask_django", SkillEvidence())

    if fastapi_ev.present:
        score += cfg.fastapi if fastapi_ev.in_project_or_work else cfg.fastapi * cfg.skills_only_ratio
    elif flask_django_ev.present:
        # Other Python web framework when FastAPI absent
        val = cfg.flask_django_fallback
        score += val if flask_django_ev.in_project_or_work else val * cfg.skills_only_ratio

    # Async
    async_ev = pb.get("async", SkillEvidence())
    if async_ev.present:
        score += cfg.async_programming if async_ev.in_project_or_work else cfg.async_programming * cfg.skills_only_ratio

    # PostgreSQL
    pg_ev = pb.get("postgresql", SkillEvidence())
    if pg_ev.present:
        score += cfg.postgresql if pg_ev.in_project_or_work else cfg.postgresql * cfg.skills_only_ratio

    # Redis
    redis_ev = pb.get("redis", SkillEvidence())
    if redis_ev.present:
        score += cfg.redis if redis_ev.in_project_or_work else cfg.redis * cfg.skills_only_ratio

    return min(round(score), max_score)


def score_cloud_fullstack(
    analysis: ResumeAnalysis,
    settings: Settings,
) -> int:
    """Score Cloud / Deployment / Full Stack (max 15)."""
    cfg = settings.cloud_fullstack
    max_score = settings.score_weights.cloud_fullstack
    score = 0.0

    cloud = analysis.cloud

    # GCP
    gcp_ev = cloud.get("gcp", SkillEvidence())
    other_cloud_ev = cloud.get("other_cloud", SkillEvidence())
    if gcp_ev.present:
        score += cfg.gcp if gcp_ev.in_project_or_work else cfg.gcp * cfg.skills_only_ratio
    elif other_cloud_ev.present:
        val = cfg.other_cloud
        score += val if other_cloud_ev.in_project_or_work else val * cfg.skills_only_ratio

    # Docker
    docker_ev = cloud.get("docker", SkillEvidence())
    if docker_ev.present:
        score += cfg.docker if docker_ev.in_project_or_work else cfg.docker * cfg.skills_only_ratio

    # Deployment evidence
    deploy_ev = cloud.get("deployment", SkillEvidence())
    if deploy_ev.present:
        score += cfg.deployment_evidence if deploy_ev.in_project_or_work else cfg.deployment_evidence * cfg.skills_only_ratio

    # React/Next.js
    react_ev = cloud.get("react_nextjs", SkillEvidence())
    if react_ev.present:
        if react_ev.in_project_or_work:
            score += cfg.react_nextjs_project
        else:
            score += cfg.react_nextjs_skills_only

    return min(round(score), max_score)


def score_github(
    github_summary: GitHubSummary,
    settings: Settings,
) -> int:
    """Score GitHub Activity (max 10)."""
    return min(github_summary.total, settings.score_weights.github)


def score_engineering_depth(
    analysis: ResumeAnalysis,
    settings: Settings,
) -> int:
    """Score Engineering Depth Signals (max 5, 1 point each)."""
    max_score = settings.score_weights.engineering_depth
    score = 0

    for signal in settings.engineering_depth.signals:
        ev = analysis.engineering_signals.get(signal, SkillEvidence())
        if ev.in_project_or_work:
            score += 1

    return min(score, max_score)


def compute_score(
    analysis: ResumeAnalysis,
    github_summary: GitHubSummary,
    settings: Settings,
) -> tuple[ScoreBreakdown, int, list[str]]:
    """
    Compute complete score from analysis and GitHub data.

    Returns (breakdown, total_score, penalties_applied).

    Enforces:
    - Each category <= its max
    - No-meaningful-AI-project cap (55)
    - total <= 100
    - total == sum(breakdown)
    """
    ai_score, penalties = score_ai_project_depth(analysis, settings)
    pb_score = score_python_backend(analysis, settings)
    cloud_score = score_cloud_fullstack(analysis, settings)
    gh_score = score_github(github_summary, settings)
    eng_score = score_engineering_depth(analysis, settings)

    breakdown = ScoreBreakdown(
        ai_project_depth=ai_score,
        python_backend=pb_score,
        cloud_fullstack=cloud_score,
        github=gh_score,
        engineering_depth=eng_score,
    )

    total = ai_score + pb_score + cloud_score + gh_score + eng_score

    # Guardrail cap: no meaningful AI project
    has_meaningful_ai = any(
        p.classification in ("substantive", "basic")
        for p in analysis.projects
    )
    cap = settings.caps.no_meaningful_ai_project
    if not has_meaningful_ai and total > cap:
        total = cap
        penalties.append(f"capped: no meaningful AI project (max {cap})")

    # Final cap at 100
    total = min(total, 100)

    # Assert invariants
    assert breakdown.ai_project_depth <= settings.score_weights.ai_project_depth, \
        f"AI score {breakdown.ai_project_depth} > max {settings.score_weights.ai_project_depth}"
    assert breakdown.python_backend <= settings.score_weights.python_backend
    assert breakdown.cloud_fullstack <= settings.score_weights.cloud_fullstack
    assert breakdown.github <= settings.score_weights.github
    assert breakdown.engineering_depth <= settings.score_weights.engineering_depth
    assert total <= 100

    return breakdown, total, penalties


def rank_candidates(
    candidates: list[dict],
) -> list[dict]:
    """
    Sort eligible candidates by total_score desc, with deterministic tie-breaks.

    Tie-break: ai_project_depth desc, then python_backend desc, then candidate_name asc.
    Assigns rank starting at 1.
    """
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            -c.get("total_score", 0),
            -c.get("score_breakdown", {}).get("ai_project_depth", 0) if isinstance(c.get("score_breakdown"), dict)
            else -(c.get("score_breakdown").ai_project_depth if c.get("score_breakdown") else 0),
            -c.get("score_breakdown", {}).get("python_backend", 0) if isinstance(c.get("score_breakdown"), dict)
            else -(c.get("score_breakdown").python_backend if c.get("score_breakdown") else 0),
            c.get("candidate_name", ""),
        ),
    )

    for i, candidate in enumerate(sorted_candidates, 1):
        candidate["rank"] = i

    return sorted_candidates
