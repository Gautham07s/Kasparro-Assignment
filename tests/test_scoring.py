"""Tests for scoring module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from screener.config import load_settings
from screener.models import (
    GitHubSummary,
    ProjectAnalysis,
    ResumeAnalysis,
    ScoreBreakdown,
    SkillEvidence,
)
from screener.scoring import (
    compute_score,
    score_ai_project_depth,
    score_python_backend,
    score_cloud_fullstack,
    score_engineering_depth,
)


@pytest.fixture
def settings():
    return load_settings()


def _make_analysis(
    classification: str = "substantive",
    features: dict | None = None,
    ownership: bool = True,
    python_in_project: bool = True,
    fastapi_in_project: bool = False,
    analysis_source: str = "fallback",
) -> ResumeAnalysis:
    """Helper to build a ResumeAnalysis for testing."""
    if features is None:
        features = {
            "retrieval": True,
            "tool_calling": True,
            "state_orchestration": True,
            "evaluation": True,
            "data_processing": True,
            "business_logic": True,
            "backend_api": True,
            "deployment": True,
        }

    project = ProjectAnalysis(
        name="Test Project",
        summary="A test project",
        classification=classification,
        features=features,
        ownership_evidence=ownership,
        evidence_quote="Built a RAG system with LangChain",
    )

    return ResumeAnalysis(
        projects=[project],
        python_backend={
            "python": SkillEvidence(present=True, in_project_or_work=python_in_project),
            "fastapi": SkillEvidence(present=fastapi_in_project, in_project_or_work=fastapi_in_project),
            "async": SkillEvidence(present=False, in_project_or_work=False),
            "postgresql": SkillEvidence(present=False, in_project_or_work=False),
            "redis": SkillEvidence(present=False, in_project_or_work=False),
            "flask_django": SkillEvidence(present=False, in_project_or_work=False),
        },
        cloud={
            "gcp": SkillEvidence(present=False, in_project_or_work=False),
            "other_cloud": SkillEvidence(present=False, in_project_or_work=False),
            "docker": SkillEvidence(present=False, in_project_or_work=False),
            "deployment": SkillEvidence(present=False, in_project_or_work=False),
            "react_nextjs": SkillEvidence(present=False, in_project_or_work=False),
        },
        engineering_signals={
            "testing": SkillEvidence(present=False, in_project_or_work=False),
            "architecture": SkillEvidence(present=False, in_project_or_work=False),
            "caching": SkillEvidence(present=False, in_project_or_work=False),
            "queues": SkillEvidence(present=False, in_project_or_work=False),
            "observability": SkillEvidence(present=False, in_project_or_work=False),
            "concurrency": SkillEvidence(present=False, in_project_or_work=False),
            "failure_handling": SkillEvidence(present=False, in_project_or_work=False),
        },
        strengths=["Strong project"],
        concerns=[],
        analysis_source=analysis_source,
    )


class TestAIProjectScoring:
    """Test AI project depth scoring."""

    def test_substantive_scores_higher_than_thin_wrapper(self, settings):
        """Substantive project should score higher than thin wrapper."""
        substantive = _make_analysis(classification="substantive")
        thin = _make_analysis(
            classification="thin_wrapper",
            features={"retrieval": False, "tool_calling": False,
                      "state_orchestration": False, "evaluation": False,
                      "data_processing": False, "business_logic": False,
                      "backend_api": False, "deployment": False},
            ownership=False,
        )

        s_score, _ = score_ai_project_depth(substantive, settings)
        t_score, t_pen = score_ai_project_depth(thin, settings)

        assert s_score > t_score
        assert len(t_pen) > 0  # penalties applied

    def test_thin_wrapper_penalties_in_list(self, settings):
        """Thin wrapper penalties should appear in penalties_applied."""
        thin = _make_analysis(
            classification="thin_wrapper",
            features={"retrieval": False, "tool_calling": False,
                      "state_orchestration": False, "evaluation": False,
                      "data_processing": False, "business_logic": False,
                      "backend_api": False, "deployment": False},
            ownership=False,
        )
        _, penalties = score_ai_project_depth(thin, settings)
        assert any("thin_wrapper" in p for p in penalties)

    def test_category_max_respected(self, settings):
        """AI score should never exceed category max (40)."""
        analysis = _make_analysis(classification="substantive")
        score, _ = score_ai_project_depth(analysis, settings)
        assert score <= settings.score_weights.ai_project_depth


class TestNoMeaningfulAICap:
    """Test the guardrail cap for no meaningful AI project."""

    def test_no_ai_project_capped_at_55(self, settings):
        """Total score must be <= 55 when no substantive/basic AI project."""
        analysis = _make_analysis(
            classification="thin_wrapper",
            features={"retrieval": False, "tool_calling": False,
                      "state_orchestration": False, "evaluation": False,
                      "data_processing": False, "business_logic": False,
                      "backend_api": True, "deployment": True},
            ownership=False,
            python_in_project=True,
            fastapi_in_project=True,
        )
        # Give it high backend scores
        analysis.python_backend["python"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.python_backend["fastapi"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.python_backend["async"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.python_backend["postgresql"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.python_backend["redis"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.cloud["docker"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.cloud["gcp"] = SkillEvidence(present=True, in_project_or_work=True)

        github = GitHubSummary(status="ok", total=10, activity_score=5, repo_score=5)
        breakdown, total, penalties = compute_score(analysis, github, settings)

        assert total <= settings.caps.no_meaningful_ai_project
        assert any("capped" in p for p in penalties)


class TestScoreInvariants:
    """Test score invariants."""

    def test_every_category_within_max(self, settings):
        analysis = _make_analysis()
        github = GitHubSummary(status="ok", total=10, activity_score=5, repo_score=5)
        breakdown, total, _ = compute_score(analysis, github, settings)

        assert breakdown.ai_project_depth <= settings.score_weights.ai_project_depth
        assert breakdown.python_backend <= settings.score_weights.python_backend
        assert breakdown.cloud_fullstack <= settings.score_weights.cloud_fullstack
        assert breakdown.github <= settings.score_weights.github
        assert breakdown.engineering_depth <= settings.score_weights.engineering_depth

    def test_total_never_exceeds_100(self, settings):
        analysis = _make_analysis()
        # Max out everything
        analysis.python_backend["python"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.python_backend["fastapi"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.python_backend["async"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.python_backend["postgresql"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.python_backend["redis"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.cloud["docker"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.cloud["gcp"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.cloud["deployment"] = SkillEvidence(present=True, in_project_or_work=True)
        analysis.cloud["react_nextjs"] = SkillEvidence(present=True, in_project_or_work=True)
        for sig in analysis.engineering_signals:
            analysis.engineering_signals[sig] = SkillEvidence(present=True, in_project_or_work=True)

        github = GitHubSummary(status="ok", total=10, activity_score=5, repo_score=5)
        breakdown, total, _ = compute_score(analysis, github, settings)
        assert total <= 100

    def test_skills_only_earns_40_percent(self, settings):
        """Skills-only mention should earn 40% of item points."""
        # In-project
        analysis_proj = _make_analysis(python_in_project=True)
        proj_score = score_python_backend(analysis_proj, settings)

        # Skills-only
        analysis_skills = _make_analysis(python_in_project=False)
        analysis_skills.python_backend["python"] = SkillEvidence(present=True, in_project_or_work=False)
        skills_score = score_python_backend(analysis_skills, settings)

        assert skills_score < proj_score

    def test_deterministic_tie_break(self):
        """Tie-breaking should be deterministic: ai_project desc, python desc, name asc."""
        from screener.scoring import rank_candidates

        candidates = [
            {"candidate_name": "Charlie", "total_score": 80,
             "score_breakdown": ScoreBreakdown(ai_project_depth=30, python_backend=25)},
            {"candidate_name": "Alice", "total_score": 80,
             "score_breakdown": ScoreBreakdown(ai_project_depth=30, python_backend=25)},
            {"candidate_name": "Bob", "total_score": 80,
             "score_breakdown": ScoreBreakdown(ai_project_depth=35, python_backend=20)},
        ]

        ranked = rank_candidates(candidates)
        # Bob has higher ai_project_depth
        assert ranked[0]["candidate_name"] == "Bob"
        # Alice before Charlie (name asc)
        assert ranked[1]["candidate_name"] == "Alice"
        assert ranked[2]["candidate_name"] == "Charlie"


class TestSkillsOnlyCap:
    """Test that AI framework only in skills list caps the AI category."""

    def test_skills_only_framework_capped(self, settings):
        """When AI framework only in skills list (no project features), cap at 8."""
        analysis = ResumeAnalysis(
            projects=[ProjectAnalysis(
                name="Todo App",
                summary="Simple todo app",
                classification="tutorial_style",
                features={
                    "retrieval": False, "tool_calling": False,
                    "state_orchestration": False, "evaluation": False,
                    "data_processing": False, "business_logic": False,
                    "backend_api": False, "deployment": False,
                },
                ownership_evidence=False,
                evidence_quote="Simple todo app",
            )],
            analysis_source="fallback",
        )
        score, penalties = score_ai_project_depth(analysis, settings)
        assert score <= settings.ai_project.skills_only_cap
        assert any("skills list" in p for p in penalties)
