"""Tests for GitHub enrichment module using respx/httpx mocking."""

import sys
from pathlib import Path

import httpx
import pytest
import respx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from screener.config import load_settings
from screener.github import enrich_github, reset_cache


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset the GitHub cache before each test."""
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def settings():
    s = load_settings()
    s.github_token = ""  # Ensure no token for tests
    s.github_timeout_seconds = 5
    return s


class TestGitHubEnrichment:
    """Test GitHub enrichment with mocked HTTP."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_ok_path_scoring(self, settings):
        """Successful GitHub fetch should return ok status and scores."""
        respx.get("https://api.github.com/users/testuser/events/public?per_page=100").mock(
            return_value=httpx.Response(200, json=[
                {"type": "PushEvent", "created_at": "2026-09-15T10:00:00Z"},
                {"type": "PushEvent", "created_at": "2026-09-14T10:00:00Z"},
                {"type": "PushEvent", "created_at": "2026-09-13T10:00:00Z"},
            ])
        )
        respx.get("https://api.github.com/users/testuser/repos?sort=pushed&per_page=100&type=owner").mock(
            return_value=httpx.Response(200, json=[
                {
                    "name": "ai-rag-project",
                    "fork": False,
                    "pushed_at": "2026-09-01T10:00:00Z",
                    "language": "Python",
                    "description": "RAG pipeline with LangChain",
                    "topics": ["ai", "rag"],
                },
                {
                    "name": "web-app",
                    "fork": False,
                    "pushed_at": "2026-08-01T10:00:00Z",
                    "language": "JavaScript",
                    "description": "A web application",
                    "topics": [],
                },
            ])
        )

        result = await enrich_github("testuser", settings)

        assert result.status == "ok"
        assert result.username == "testuser"
        assert result.recent_push_events == 3
        assert result.activity_score == 2  # 1-4 pushes -> 2
        assert result.maintained_repos == 2
        assert result.relevant_repos >= 1  # Python + AI keywords
        assert result.total > 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_404_not_found(self, settings):
        """404 response should return not_found status."""
        respx.get("https://api.github.com/users/nonexistent/events/public?per_page=100").mock(
            return_value=httpx.Response(404, json={"message": "Not Found"})
        )

        result = await enrich_github("nonexistent", settings)
        assert result.status == "not_found"

    @pytest.mark.asyncio
    @respx.mock
    async def test_429_rate_limited(self, settings):
        """429 response should set rate_limited and prevent further calls."""
        respx.get("https://api.github.com/users/user1/events/public?per_page=100").mock(
            return_value=httpx.Response(429, headers={"X-RateLimit-Remaining": "0"})
        )

        result1 = await enrich_github("user1", settings)
        assert result1.status == "rate_limited"

        # Second call should also be rate_limited WITHOUT making a request
        result2 = await enrich_github("user2", settings)
        assert result2.status == "rate_limited"

    @pytest.mark.asyncio
    @respx.mock
    async def test_timeout_error(self, settings):
        """Timeout should return error status."""
        respx.get("https://api.github.com/users/slowuser/events/public?per_page=100").mock(
            side_effect=httpx.TimeoutException("Connection timed out")
        )

        result = await enrich_github("slowuser", settings)
        assert result.status == "error"

    @pytest.mark.asyncio
    @respx.mock
    async def test_cache_prevents_second_request(self, settings):
        """Same username should only fetch once (cache hit)."""
        route = respx.get("https://api.github.com/users/cacheduser/events/public?per_page=100").mock(
            return_value=httpx.Response(200, json=[])
        )
        respx.get("https://api.github.com/users/cacheduser/repos?sort=pushed&per_page=100&type=owner").mock(
            return_value=httpx.Response(200, json=[])
        )

        result1 = await enrich_github("cacheduser", settings)
        result2 = await enrich_github("cacheduser", settings)

        assert result1.status == result2.status
        assert route.call_count == 1  # Only one API call

    @pytest.mark.asyncio
    async def test_no_username_returns_no_github(self, settings):
        """None username should return no_github status."""
        result = await enrich_github(None, settings)
        assert result.status == "no_github"
