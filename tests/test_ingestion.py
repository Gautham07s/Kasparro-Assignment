"""Tests for ingestion module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from screener.ingestion import ingest_resumes


class TestIngestion:
    """Test resume ingestion pipeline."""

    def test_corrupt_pdf_becomes_failed_resume(self, fixtures_dir: Path):
        """A corrupt PDF must produce a FailedResume, never raise."""
        # Create a temp dir with just the corrupt PDF
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy(fixtures_dir / "corrupt.pdf", tmp / "corrupt.pdf")
            parsed, failed, dupes = ingest_resumes(tmp)
            assert len(parsed) == 0
            assert len(failed) == 1
            assert failed[0].file_name == "corrupt.pdf"
            assert failed[0].stage == "read"

    def test_unsupported_extension_recorded(self, fixtures_dir: Path):
        """Unsupported file types are recorded in failed, not raised."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "resume.xlsx").write_text("data")
            parsed, failed, dupes = ingest_resumes(tmp)
            assert len(parsed) == 0
            assert len(failed) == 1
            assert "unsupported file type" in failed[0].reason

    def test_txt_files_ingested(self, fixtures_dir: Path):
        """TXT files should be successfully ingested."""
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy(fixtures_dir / "strong_agentic_python.txt", tmp / "strong.txt")
            parsed, failed, dupes = ingest_resumes(tmp)
            assert len(parsed) == 1
            assert parsed[0].file_name == "strong.txt"
            assert parsed[0].candidate_name  # Should have extracted a name

    def test_duplicate_detection_by_content(self, fixtures_dir: Path):
        """Identical files should be flagged as duplicates."""
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy(fixtures_dir / "strong_agentic_python.txt", tmp / "copy_a.txt")
            shutil.copy(fixtures_dir / "strong_agentic_python.txt", tmp / "copy_b.txt")
            parsed, failed, dupes = ingest_resumes(tmp)
            assert len(parsed) == 1
            assert len(dupes) == 1
            assert dupes[0].duplicate_of == "copy_a.txt"
            assert dupes[0].file_name == "copy_b.txt"

    def test_hidden_files_skipped(self, fixtures_dir: Path):
        """Files starting with . should be skipped."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / ".hidden.txt").write_text("secret stuff " * 20)
            parsed, failed, dupes = ingest_resumes(tmp)
            assert len(parsed) == 0
            assert len(failed) == 0

    def test_empty_directory(self):
        """Empty directory should return empty lists."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            parsed, failed, dupes = ingest_resumes(Path(tmpdir))
            assert len(parsed) == 0
            assert len(failed) == 0
            assert len(dupes) == 0
