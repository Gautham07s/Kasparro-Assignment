"""
GitHub enrichment module.

Fetches public GitHub activity for eligible candidates with a github_username.
Uses async httpx with bounded concurrency and run-level caching.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from .config import Settings
from .models import GitHubSummary

logger = logging.getLogger(__name__)

# Run-level cache: prevents repeated enrichment for the same username
_github_cache: dict[str, GitHubSummary] = {}
_rate_limited: bool = False
_warned_no_token: bool = False


def reset_cache() -> None:
    """Reset the run-level cache (for testing)."""
    global _github_cache, _rate_limited, _warned_no_token
    _github_cache.clear()
    _rate_limited = False
    _warned_no_token = False


async def enrich_github(
    username: Optional[str],
    settings: Settings,
    client: Optional[httpx.AsyncClient] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> GitHubSummary:
    """
    Fetch GitHub activity for a username.

    Returns GitHubSummary with status, scores, and summary text.
    Never raises to the caller.
    """
    global _rate_limited

    if not username:
        return GitHubSummary(status="no_github", summary="No GitHub profile provided")

    username_lower = username.lower()

    # Check cache
    if username_lower in _github_cache:
        logger.debug("GitHub cache hit for %s", username_lower)
        return _github_cache[username_lower]

    # If rate-limited, don't make new calls
    if _rate_limited:
        result = GitHubSummary(
            status="rate_limited",
            username=username,
            summary="Skipped: GitHub API rate limit reached",
            error="Rate limit reached for this run",
        )
        _github_cache[username_lower] = result
        return result

    # Create client if not provided
    own_client = client is None
    if own_client:
        client = _create_client(settings)

    try:
        if semaphore:
            async with semaphore:
                return await _fetch_github_data(username, username_lower, settings, client)
        else:
            return await _fetch_github_data(username, username_lower, settings, client)
    except Exception as exc:
        logger.error("GitHub enrichment error for %s: %s", username, exc)
        result = GitHubSummary(
            status="error",
            username=username,
            summary=f"GitHub enrichment failed: {exc}",
            error=str(exc),
        )
        _github_cache[username_lower] = result
        return result
    finally:
        if own_client and client:
            await client.aclose()



def _create_client(settings: Settings) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient with proper headers."""
    global _warned_no_token

    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    elif not _warned_no_token:
        _warned_no_token = True
        logger.warning(
            "No GITHUB_TOKEN set. Unauthenticated rate limit is 60 req/hr. "
            "50 candidates × 2 calls = 100 calls may exceed this."
        )
    return httpx.AsyncClient(
        timeout=settings.github_timeout_seconds,
        headers=headers,
    )


async def _fetch_github_data(
    username: str,
    username_lower: str,
    settings: Settings,
    client: httpx.AsyncClient,
) -> GitHubSummary:
    """Fetch events + repos for a GitHub user."""
    global _rate_limited

    base_url = "https://api.github.com"

    # Fetch events
    try:
        events_resp = await client.get(f"{base_url}/users/{username}/events/public?per_page=100")
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        result = GitHubSummary(
            status="error", username=username,
            summary=f"GitHub request failed: {exc}", error=str(exc),
        )
        _github_cache[username_lower] = result
        return result

    if events_resp.status_code == 404:
        result = GitHubSummary(
            status="not_found", username=username,
            summary=f"GitHub user '{username}' not found",
        )
        _github_cache[username_lower] = result
        return result

    if events_resp.status_code in (403, 429):
        remaining = events_resp.headers.get("X-RateLimit-Remaining", "0")
        if remaining == "0" or events_resp.status_code == 429:
            _rate_limited = True
            result = GitHubSummary(
                status="rate_limited", username=username,
                summary="GitHub API rate limit reached",
                error="Rate limited",
            )
            _github_cache[username_lower] = result
            return result

    if events_resp.status_code != 200:
        result = GitHubSummary(
            status="error", username=username,
            summary=f"GitHub API error: {events_resp.status_code}",
            error=f"HTTP {events_resp.status_code}",
        )
        _github_cache[username_lower] = result
        return result

    # Fetch repos
    try:
        repos_resp = await client.get(
            f"{base_url}/users/{username}/repos?sort=pushed&per_page=100&type=owner"
        )
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        result = GitHubSummary(
            status="error", username=username,
            summary=f"GitHub repos request failed: {exc}", error=str(exc),
        )
        _github_cache[username_lower] = result
        return result

    if repos_resp.status_code in (403, 429):
        remaining = repos_resp.headers.get("X-RateLimit-Remaining", "0")
        if remaining == "0" or repos_resp.status_code == 429:
            _rate_limited = True

    # Parse events
    events = events_resp.json() if events_resp.status_code == 200 else []
    repos = repos_resp.json() if repos_resp.status_code == 200 else []

    # Handle non-list responses (e.g., error objects)
    if not isinstance(events, list):
        events = []
    if not isinstance(repos, list):
        repos = []

    # Count PushEvents in last 90 days
    cutoff = datetime.now(timezone.utc) - timedelta(days=90)
    recent_pushes = 0
    for event in events:
        if event.get("type") == "PushEvent":
            created = event.get("created_at", "")
            try:
                event_time = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if event_time > cutoff:
                    recent_pushes += 1
            except (ValueError, AttributeError):
                continue

    # Activity score
    activity_score = _score_activity(recent_pushes, settings)

    # Repo analysis
    now = datetime.now(timezone.utc)
    twelve_months_ago = now - timedelta(days=365)

    maintained_repos = 0
    relevant_repos = 0
    ai_keywords = set(kw.lower() for kw in settings.github_config.ai_keywords)

    for repo in repos:
        if repo.get("fork", False):
            continue

        # Maintained: pushed within last 12 months
        pushed_at = repo.get("pushed_at", "")
        try:
            push_time = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
            if push_time > twelve_months_ago:
                maintained_repos += 1
        except (ValueError, AttributeError):
            continue

        # Relevant: Python language OR AI keywords
        language = (repo.get("language") or "").lower()
        name = (repo.get("name") or "").lower()
        description = (repo.get("description") or "").lower()
        topics = [t.lower() for t in repo.get("topics", [])]

        is_relevant = language == "python"
        if not is_relevant:
            searchable = f"{name} {description} {' '.join(topics)}"
            is_relevant = any(kw in searchable for kw in ai_keywords)

        if is_relevant:
            relevant_repos += 1

    # Repo score
    repo_score = _score_repos(maintained_repos, relevant_repos, settings)

    total = activity_score + repo_score

    # Build summary
    summary_parts = []
    if recent_pushes > 0:
        summary_parts.append(f"Recently active ({recent_pushes} push events in 90 days)")
    else:
        summary_parts.append("No recent push activity")

    summary_parts.append(f"{maintained_repos} maintained non-fork repos")
    if relevant_repos > 0:
        summary_parts.append(f"{relevant_repos} Python/AI relevant")

    result = GitHubSummary(
        status="ok",
        username=username,
        activity_score=activity_score,
        repo_score=repo_score,
        total=total,
        recent_push_events=recent_pushes,
        maintained_repos=maintained_repos,
        relevant_repos=relevant_repos,
        summary="; ".join(summary_parts) + ".",
    )

    _github_cache[username_lower] = result
    return result


def _score_activity(push_count: int, settings: Settings) -> int:
    """Score activity from PushEvents count."""
    for threshold in settings.github_config.activity_thresholds:
        if threshold.min <= push_count <= threshold.max:
            return threshold.score
    return 0


def _score_repos(maintained: int, relevant: int, settings: Settings) -> int:
    """Score repos from maintained + relevant counts."""
    maintained_score = 0
    for threshold in settings.github_config.repo_maintained_points:
        if maintained >= threshold.count:
            maintained_score = threshold.score

    relevant_score = 0
    for threshold in settings.github_config.repo_relevant_points:
        if relevant >= threshold.count:
            relevant_score = threshold.score

    return maintained_score + relevant_score
