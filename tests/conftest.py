"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Ensure src/ is on sys.path for test imports
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def settings():
    """Load the real Settings object for tests."""
    from screener.config import load_settings
    return load_settings()
