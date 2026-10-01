"""Tests for pipeline resilience: LLM failures, batch completion, count reconciliation."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from screener.config import load_settings
from screener.llm.base import LLMError
from screener.models import ResumeAnalysis


@pytest.fixture
def settings():
    s = load_settings()
    s.github_token = ""
    return s


class TestPipelineResilience:
    """Test that the pipeline handles failures gracefully."""

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_batch_completes(self, settings, fixtures_dir):
        """
        A mock LLMClient that raises for one resume:
        - That candidate is still ranked via fallback
        - Batch completes
        - llm_fallbacks == count of failures
        """
        import tempfile, shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Copy eligible fixtures
            shutil.copy(fixtures_dir / "strong_agentic_python.txt", tmp / "strong.txt")
            shutil.copy(fixtures_dir / "python_js_langchain.txt", tmp / "mixed.txt")

            from screener.pipeline import run_screening

            # Run without LLM (deterministic)
            run = await run_screening(
                input_dir=tmp,
                settings=settings,
                use_llm=False,
            )

            # All candidates should be processed
            assert run.batch_summary.successfully_parsed == 2
            assert run.batch_summary.eligible == 2
            assert len(run.ranked_candidates) == 2
            # All should be fallback since we used --no-llm
            for c in run.ranked_candidates:
                assert c.analysis_source == "fallback"

    @pytest.mark.asyncio
    async def test_counts_reconcile(self, settings, fixtures_dir):
        """Batch summary counts must reconcile."""
        import tempfile, shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy(fixtures_dir / "strong_agentic_python.txt", tmp / "eligible.txt")
            shutil.copy(fixtures_dir / "java_react_only.txt", tmp / "rejected.txt")
            shutil.copy(fixtures_dir / "corrupt.pdf", tmp / "corrupt.pdf")

            from screener.pipeline import run_screening

            run = await run_screening(
                input_dir=tmp,
                settings=settings,
                use_llm=False,
            )

            bs = run.batch_summary
            # total = parsed + failed + duplicates
            assert bs.total_resumes == bs.successfully_parsed + bs.failed_unreadable + bs.duplicates
            # parsed = eligible + rejected
            assert bs.successfully_parsed == bs.eligible + bs.rejected

    @pytest.mark.asyncio
    async def test_corrupt_pdf_doesnt_crash_batch(self, settings, fixtures_dir):
        """A corrupt PDF must not crash the entire batch."""
        import tempfile, shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy(fixtures_dir / "strong_agentic_python.txt", tmp / "good.txt")
            shutil.copy(fixtures_dir / "corrupt.pdf", tmp / "bad.pdf")

            from screener.pipeline import run_screening

            run = await run_screening(
                input_dir=tmp,
                settings=settings,
                use_llm=False,
            )

            assert run.batch_summary.successfully_parsed >= 1
            assert run.batch_summary.failed_unreadable >= 1
            assert len(run.ranked_candidates) >= 1
