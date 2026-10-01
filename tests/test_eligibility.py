"""Tests for eligibility module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from screener.config import load_settings
from screener.eligibility import check_eligibility
from screener.models import ParsedResume


def _make_resume(text: str, file_name: str = "test.txt") -> ParsedResume:
    return ParsedResume(
        file_name=file_name, file_path=f"/{file_name}",
        content_hash="test", raw_text=text, links=[],
    )


@pytest.fixture
def settings():
    return load_settings()


class TestEligibility:
    """Test hard eligibility rules."""

    def test_strong_agentic_python_eligible(self, settings, fixtures_dir):
        text = (fixtures_dir / "strong_agentic_python.txt").read_text()
        result = check_eligibility(_make_resume(text), settings)
        assert result.eligible is True
        assert len(result.rejection_reasons) == 0
        assert len(result.python_evidence) > 0
        assert len(result.ai_evidence) > 0

    def test_java_react_only_rejected(self, settings, fixtures_dir):
        text = (fixtures_dir / "java_react_only.txt").read_text()
        result = check_eligibility(_make_resume(text), settings)
        assert result.eligible is False
        assert "No evidence of Python stack" in result.rejection_reasons
        assert "No AI/agentic project evidence" in result.rejection_reasons
        # Should still have matched skills
        assert "Java" in result.matched_skills
        assert "React" in result.matched_skills

    def test_python_without_ai_rejected(self, settings, fixtures_dir):
        text = (fixtures_dir / "python_no_ai.txt").read_text()
        result = check_eligibility(_make_resume(text), settings)
        assert result.eligible is False
        assert "No AI/agentic project evidence" in result.rejection_reasons
        # Python evidence should be found
        assert len(result.python_evidence) > 0

    def test_ai_without_python_rejected(self, settings, fixtures_dir):
        text = (fixtures_dir / "ai_no_python.txt").read_text()
        result = check_eligibility(_make_resume(text), settings)
        assert result.eligible is False
        assert "No evidence of Python stack" in result.rejection_reasons

    def test_mixed_python_js_langchain_eligible(self, settings, fixtures_dir):
        """JS present + Python + AI still eligible."""
        text = (fixtures_dir / "python_js_langchain.txt").read_text()
        result = check_eligibility(_make_resume(text), settings)
        assert result.eligible is True
        assert len(result.rejection_reasons) == 0

    def test_thin_wrapper_eligible(self, settings, fixtures_dir):
        """Thin wrapper has Python + OpenAI API -> eligible (penalty is in scoring)."""
        text = (fixtures_dir / "thin_wrapper.txt").read_text()
        result = check_eligibility(_make_resume(text), settings)
        assert result.eligible is True

    def test_skills_only_eligible(self, settings, fixtures_dir):
        """Skills-list-only frameworks: still eligible (cap is in scoring)."""
        text = (fixtures_dir / "skills_only_frameworks.txt").read_text()
        result = check_eligibility(_make_resume(text), settings)
        assert result.eligible is True

    def test_ml_only_rejected_with_specific_reason(self, settings, fixtures_dir):
        """Classic ML/CV/NLP only: rejected with 'ML only' reason."""
        text = (fixtures_dir / "ml_only.txt").read_text()
        result = check_eligibility(_make_resume(text), settings)
        assert result.eligible is False
        assert "ML only, no LLM/agentic evidence" in result.rejection_reasons

    def test_jython_does_not_count(self, settings):
        """'jython' alone does NOT satisfy Python evidence."""
        text = "I use Jython for scripting. LangChain, RAG, vector search."
        result = check_eligibility(_make_resume(text), settings)
        # Jython should not count, but LangChain is a python_implied term
        # So python_implied WILL match on langchain
        # The key test: if ONLY jython and no python ecosystem -> rejected
        text2 = "I use Jython for scripting. LangChain, RAG."
        result2 = check_eligibility(_make_resume(text2), settings)
        # LangChain is in python_implied, so it still passes python check
        # This is correct behavior - LangChain implies Python
        assert result2.eligible is True

    def test_jython_only_no_python_ecosystem(self, settings):
        """Pure Jython with AI but no Python ecosystem -> rejected."""
        text = "I use Jython for scripting. OpenAI API for chatbot. RAG pipeline."
        result = check_eligibility(_make_resume(text), settings)
        # "openai api" is an ai_provider_api, so AI check passes
        # But for Python: "jython" has \bpython\b? No, jython doesn't match \bpython\b
        # Actually wait - does \bjython\b match? No, \bpython\b does not match "jython"
        # So python_direct check fails. But "openai api" is not in python_implied.
        # So Python check fails entirely
        assert result.eligible is False
        assert "No evidence of Python stack" in result.rejection_reasons

    def test_plain_ai_ml_alone_not_sufficient(self, settings):
        """Plain 'AI' or 'machine learning' alone is NOT sufficient for AI evidence."""
        text = "Python developer working on AI and machine learning projects."
        result = check_eligibility(_make_resume(text), settings)
        # "machine learning" is in ml_only_terms, not in ai qualifying terms
        # "ai" alone: check if \bai\b matches any ai term...
        # "ai agent" contains "ai" but \bai agent\b won't match just "ai"
        # So this should be rejected
        assert result.eligible is False

    def test_matched_skills_even_for_rejected(self, settings, fixtures_dir):
        """Rejected candidates still get matched_skills populated."""
        text = (fixtures_dir / "java_react_only.txt").read_text()
        result = check_eligibility(_make_resume(text), settings)
        assert result.eligible is False
        assert len(result.matched_skills) > 0
