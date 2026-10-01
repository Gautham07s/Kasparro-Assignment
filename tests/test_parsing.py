"""Tests for parsing module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from screener.models import ParsedResume
from screener.parsing import parse_candidate_info, split_sections, _extract_github_username


class TestEmailExtraction:
    """Test email regex extraction."""

    def test_standard_email(self):
        resume = ParsedResume(
            file_name="test.txt", file_path="/test.txt",
            content_hash="abc", raw_text="Contact: arjun.mehta@gmail.com for inquiries",
            links=[],
        )
        result = parse_candidate_info(resume)
        assert result.email == "arjun.mehta@gmail.com"

    def test_no_email(self):
        resume = ParsedResume(
            file_name="test.txt", file_path="/test.txt",
            content_hash="abc", raw_text="No email here at all",
            links=[],
        )
        result = parse_candidate_info(resume)
        assert result.email is None


class TestGitHubExtraction:
    """Test GitHub username extraction from text and links."""

    def test_github_url_in_text(self):
        username, url = _extract_github_username(
            "Visit my GitHub: https://github.com/johndoe/my-project", []
        )
        assert username == "johndoe"

    def test_github_repo_url_extracts_user_only(self):
        """github.com/user/repo -> user (only first path segment)."""
        username, url = _extract_github_username(
            "https://github.com/alice/awesome-rag-project", []
        )
        assert username == "alice"

    def test_github_from_link_annotation(self):
        """Link annotation should be preferred over text."""
        username, url = _extract_github_username(
            "GitHub profile",  # text has no URL
            ["https://github.com/linkuser/repo"],
        )
        assert username == "linkuser"

    def test_link_annotation_preferred_over_text(self):
        """When both exist, link annotation wins."""
        username, url = _extract_github_username(
            "Check https://github.com/textuser",
            ["https://github.com/linkuser"],
        )
        assert username == "linkuser"

    def test_reserved_paths_ignored(self):
        """Reserved GitHub paths should not be treated as usernames."""
        username, url = _extract_github_username(
            "https://github.com/features", []
        )
        assert username is None

    def test_no_github(self):
        username, url = _extract_github_username("No GitHub here", [])
        assert username is None

    def test_name_fallback_to_filename(self):
        resume = ParsedResume(
            file_name="candidate_42.txt", file_path="/candidate_42.txt",
            content_hash="abc", raw_text="12345\nhttps://example.com\n\n",
            links=[],
        )
        result = parse_candidate_info(resume)
        assert "Candidate" in result.candidate_name


class TestNameExtraction:
    """Test candidate name extraction."""

    def test_name_from_first_line(self):
        resume = ParsedResume(
            file_name="test.txt", file_path="/test.txt",
            content_hash="abc",
            raw_text="Arjun Mehta\narjun.mehta@gmail.com\nSKILLS\nPython",
            links=[],
        )
        result = parse_candidate_info(resume)
        assert result.candidate_name == "Arjun Mehta"


class TestSectionSplitting:
    """Test section detection."""

    def test_basic_sections(self):
        text = """SKILLS
Python, FastAPI, Docker

PROJECTS
Built a cool RAG system

EXPERIENCE
Worked at TechCorp

EDUCATION
B.Tech CS"""
        sections = split_sections(text)
        assert "skills" in sections
        assert "projects" in sections
        assert "experience" in sections
        assert "education" in sections

    def test_no_sections_returns_full(self):
        text = "Just some random text without any section headings at all."
        sections = split_sections(text)
        assert "full" in sections
