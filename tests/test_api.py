"""Tests for FastAPI endpoints."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from screener.api import app, _screening_lock
    # Reset state
    import screener.api as api_module
    api_module._latest_run = None
    return TestClient(app)


@pytest.fixture
def fixtures_dir():
    return Path(__file__).resolve().parent / "fixtures"


class TestAPI:
    """Test FastAPI endpoints."""

    def test_health_check(self, client):
        """GET /health should return ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_results_404_before_any_run(self, client):
        """GET /results should return 404 before any screening run."""
        response = client.get("/results")
        assert response.status_code == 404

    def test_screen_with_fixtures(self, client, fixtures_dir):
        """POST /screen on fixtures dir with use_llm=false should work."""
        response = client.post("/screen", json={
            "input_dir": str(fixtures_dir),
            "use_llm": False,
            "output_path": str(fixtures_dir.parent.parent / "output" / "test_results.json"),
        })
        assert response.status_code == 200
        data = response.json()
        assert "batch_summary" in data
        assert "ranked_candidates" in data
        assert data["batch_summary"]["total_resumes"] > 0

    def test_results_200_after_run(self, client, fixtures_dir):
        """GET /results should return 200 after a screening run."""
        # First, run a screening
        client.post("/screen", json={
            "input_dir": str(fixtures_dir),
            "use_llm": False,
            "output_path": str(fixtures_dir.parent.parent / "output" / "test_results2.json"),
        })

        response = client.get("/results")
        assert response.status_code == 200
        data = response.json()
        assert "batch_summary" in data

    def test_bad_input_dir_400(self, client):
        """POST /screen with non-existent dir should return 400."""
        response = client.post("/screen", json={
            "input_dir": "/nonexistent/path/to/resumes",
            "use_llm": False,
        })
        assert response.status_code == 400
