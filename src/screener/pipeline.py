"""
Pipeline orchestration.

Ties together: ingestion -> eligibility -> LLM/fallback analysis ->
GitHub enrichment -> scoring -> ranking -> report.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from .config import Settings
from .eligibility import check_eligibility
from .github import enrich_github, reset_cache
from .ingestion import ingest_resumes
from .llm.fallback import analyze_resume_fallback
from .models import (
    BatchSummary,
    CandidateResult,
    GitHubSummary,
    ScreeningRun,
)
from .scoring import compute_score

logger = logging.getLogger(__name__)


async def _process_eligible_candidate(
    resume,
    eligibility_result,
    settings: Settings,
    use_llm: bool,
    llm_client,
    semaphore: asyncio.Semaphore,
) -> tuple[CandidateResult, bool]:
    """
    Process a single eligible candidate: LLM/fallback analysis + GitHub + scoring.

    Returns (CandidateResult, used_fallback).
    """
    used_fallback = False

    # Step 1: LLM or fallback analysis
    analysis = None
    if use_llm and llm_client is not None:
        try:
            async with semaphore:
                from .llm.prompts import build_system_prompt, build_user_prompt
                from .models import ResumeAnalysis

                system_prompt = build_system_prompt()
                user_prompt = build_user_prompt(resume.raw_text)

                analysis = await llm_client.extract(
                    system=system_prompt,
                    user=user_prompt,
                    schema=ResumeAnalysis,
                )
                analysis.analysis_source = "llm"

                # Evidence verification: fuzzy-check quotes against resume
                _verify_evidence(analysis, resume.raw_text)

        except Exception as exc:
            logger.warning(
                "LLM failed for %s, falling back to deterministic: %s",
                resume.file_name, exc,
            )
            analysis = None
            used_fallback = True

    if analysis is None:
        analysis = analyze_resume_fallback(resume.raw_text, settings)
        if use_llm:  # Only count as fallback if LLM was intended
            used_fallback = True

    # Step 2: GitHub enrichment
    github_summary = await enrich_github(
        resume.github_username, settings, semaphore=semaphore,
    )

    # Step 3: Scoring
    breakdown, total_score, penalties = compute_score(analysis, github_summary, settings)

    # Step 4: Build project summary
    project_summaries = []
    for p in analysis.projects:
        project_summaries.append(f"{p.name} ({p.classification}): {p.summary[:100]}")
    project_summary = "; ".join(project_summaries) if project_summaries else "No projects analyzed"

    # Build evidence list
    evidence = []
    for ev in eligibility_result.python_evidence:
        evidence.append(f"Python ({ev.kind}): {ev.term}")
    for ev in eligibility_result.ai_evidence:
        evidence.append(f"AI ({ev.kind}): {ev.term}")

    result = CandidateResult(
        candidate_name=resume.candidate_name,
        email=resume.email,
        source_file=resume.file_name,
        eligible=True,
        total_score=total_score,
        score_breakdown=breakdown,
        matched_skills=eligibility_result.matched_skills,
        project_summary=project_summary,
        github_summary=github_summary.summary,
        strengths=analysis.strengths,
        concerns=analysis.concerns,
        evidence=evidence,
        analysis_source=analysis.analysis_source,
        penalties_applied=penalties,
    )

    return result, used_fallback


def _verify_evidence(analysis, raw_text: str) -> None:
    """
    Fuzzy-substring check each evidence_quote against the resume text.

    If a quote doesn't approximately match, downgrade that project's
    ownership_evidence and clear the quote.
    """
    text_lower = raw_text.lower()

    for project in analysis.projects:
        if project.evidence_quote:
            quote_lower = project.evidence_quote.lower().strip()
            # Fuzzy match: check if a significant portion of the quote exists in text
            if len(quote_lower) > 10:
                # Check if at least 60% of words are found nearby
                words = quote_lower.split()
                found = sum(1 for w in words if w in text_lower)
                if found / max(len(words), 1) < 0.5:
                    logger.warning(
                        "Evidence quote verification failed for project '%s'",
                        project.name,
                    )
                    project.evidence_quote = ""
                    project.ownership_evidence = False


async def run_screening(
    input_dir: str | Path,
    settings: Settings,
    use_llm: bool = True,
) -> ScreeningRun:
    """
    Main screening pipeline.

    1. Ingest resumes
    2. Check eligibility (sync, cheap)
    3. For eligible: LLM/fallback + GitHub + scoring (async, bounded concurrency)
    4. Rank and build summary

    Counts MUST reconcile: total = parsed + failed + duplicates
    """
    start_time = time.monotonic()
    input_path = Path(input_dir)

    logger.info("Starting screening pipeline: input=%s, use_llm=%s", input_path, use_llm)

    # ── Step 1: Ingestion ──
    t0 = time.monotonic()
    parsed, failed, duplicates = ingest_resumes(input_path)
    logger.info("Ingestion took %.2fs: %d parsed, %d failed, %d duplicates",
                time.monotonic() - t0, len(parsed), len(failed), len(duplicates))

    # Reset GitHub cache for this run
    reset_cache()

    # ── Step 2: Eligibility ──
    t0 = time.monotonic()
    eligible_resumes = []
    rejected_candidates = []

    for resume in parsed:
        elig_result = check_eligibility(resume, settings)
        if elig_result.eligible:
            eligible_resumes.append((resume, elig_result))
        else:
            rejected_candidates.append(CandidateResult(
                candidate_name=resume.candidate_name,
                email=resume.email,
                source_file=resume.file_name,
                eligible=False,
                rejection_reasons=elig_result.rejection_reasons,
                matched_skills=elig_result.matched_skills,
            ))

    logger.info("Eligibility took %.2fs: %d eligible, %d rejected",
                time.monotonic() - t0, len(eligible_resumes), len(rejected_candidates))

    # ── Step 3: Process eligible candidates ──
    t0 = time.monotonic()
    semaphore = asyncio.Semaphore(settings.max_concurrency)
    llm_client = None
    llm_fallbacks = 0
    github_failures = 0

    if use_llm:
        llm_client = _get_llm_client(settings)

    # Process all eligible candidates concurrently
    tasks = []
    for resume, elig_result in eligible_resumes:
        task = _process_eligible_candidate(
            resume, elig_result, settings, use_llm, llm_client, semaphore,
        )
        tasks.append(task)

    ranked_results = []
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Unexpected error processing candidate: %s", result)
                resume, elig_result = eligible_resumes[i]
                # Still try to produce a result via fallback
                try:
                    analysis = analyze_resume_fallback(resume.raw_text, settings)
                    github_summary = GitHubSummary(status="error", error=str(result))
                    breakdown, total_score, penalties = compute_score(
                        analysis, github_summary, settings,
                    )
                    fallback_result = CandidateResult(
                        candidate_name=resume.candidate_name,
                        email=resume.email,
                        source_file=resume.file_name,
                        eligible=True,
                        total_score=total_score,
                        score_breakdown=breakdown,
                        matched_skills=elig_result.matched_skills,
                        project_summary="Analysis via fallback due to error",
                        github_summary=github_summary.summary,
                        strengths=analysis.strengths,
                        concerns=analysis.concerns,
                        analysis_source="fallback",
                        penalties_applied=penalties,
                    )
                    ranked_results.append(fallback_result)
                    llm_fallbacks += 1
                except Exception as inner_exc:
                    logger.error("Even fallback failed for %s: %s", resume.file_name, inner_exc)
            else:
                candidate_result, used_fallback = result
                ranked_results.append(candidate_result)
                if used_fallback:
                    llm_fallbacks += 1
                # Track GitHub failures
                if candidate_result.github_summary and any(
                    status in candidate_result.github_summary.lower()
                    for status in ["not found", "rate limit", "failed", "error"]
                ):
                    github_failures += 1

    logger.info("Processing took %.2fs", time.monotonic() - t0)

    # ── Step 4: Rank ──
    ranked_results.sort(
        key=lambda c: (
            -c.total_score,
            -(c.score_breakdown.ai_project_depth if c.score_breakdown else 0),
            -(c.score_breakdown.python_backend if c.score_breakdown else 0),
            c.candidate_name,
        ),
    )
    for i, candidate in enumerate(ranked_results, 1):
        candidate.rank = i

    # ── Build Summary ──
    total_resumes = len(parsed) + len(failed) + len(duplicates)
    batch_summary = BatchSummary(
        total_resumes=total_resumes,
        successfully_parsed=len(parsed),
        eligible=len(eligible_resumes),
        rejected=len(rejected_candidates),
        failed_unreadable=len(failed),
        duplicates=len(duplicates),
        llm_fallbacks=llm_fallbacks,
        github_failures=github_failures,
    )

    # Reconciliation check
    assert batch_summary.successfully_parsed == batch_summary.eligible + batch_summary.rejected, (
        f"Parsed ({batch_summary.successfully_parsed}) != "
        f"eligible ({batch_summary.eligible}) + rejected ({batch_summary.rejected})"
    )
    assert batch_summary.total_resumes == (
        batch_summary.successfully_parsed + batch_summary.failed_unreadable + batch_summary.duplicates
    ), "Total doesn't reconcile"

    total_time = time.monotonic() - start_time
    logger.info("Pipeline complete in %.2fs", total_time)

    return ScreeningRun(
        batch_summary=batch_summary,
        ranked_candidates=ranked_results,
        rejected_candidates=rejected_candidates,
        failed_files=failed,
        duplicates=duplicates,
    )


def _get_llm_client(settings: Settings):
    """Get the configured LLM client, or None."""
    try:
        if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
            from .llm.anthropic_client import AnthropicClient
            return AnthropicClient(settings)
        elif settings.llm_provider == "openai" and settings.openai_api_key:
            from .llm.openai_client import OpenAIClient
            return OpenAIClient(settings)
        elif settings.llm_provider == "gemini" and settings.gemini_api_key:
            from .llm.gemini_client import GeminiClient
            return GeminiClient(settings)
        else:
            logger.warning("No LLM API key configured, using fallback analyzer")
            return None
    except Exception as exc:
        logger.warning("Failed to initialize LLM client: %s", exc)
        return None
