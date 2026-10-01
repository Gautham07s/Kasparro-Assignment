"""
Configuration loader.

Loads config/settings.yaml and merges with environment variables.
Env vars override YAML for: model, provider, concurrency, timeouts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/screener -> src -> project


def _load_yaml() -> dict[str, Any]:
    """Load the settings.yaml file from the config directory."""
    yaml_path = _PROJECT_ROOT / "config" / "settings.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Settings file not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ScoreWeights(BaseModel):
    ai_project_depth: int = 40
    python_backend: int = 30
    cloud_fullstack: int = 15
    github: int = 10
    engineering_depth: int = 5


class AIProjectConfig(BaseModel):
    tier_points: dict[str, int] = Field(default_factory=dict)
    feature_points: dict[str, int] = Field(default_factory=dict)
    ownership_bonus: int = 2
    second_project_bonus: int = 2
    penalties: dict[str, int] = Field(default_factory=dict)
    skills_only_cap: int = 8


class PythonBackendConfig(BaseModel):
    python: int = 8
    fastapi: int = 6
    async_programming: int = 5
    postgresql: int = 6
    redis: int = 5
    flask_django_fallback: int = 3
    skills_only_ratio: float = 0.4


class CloudFullstackConfig(BaseModel):
    gcp: int = 5
    other_cloud: int = 3
    docker: int = 5
    deployment_evidence: int = 3
    react_nextjs_project: int = 2
    react_nextjs_skills_only: int = 1
    skills_only_ratio: float = 0.4


class GitHubThreshold(BaseModel):
    min: int
    max: int
    score: int


class GitHubRepoThreshold(BaseModel):
    count: int
    score: int


class GitHubConfig(BaseModel):
    activity_thresholds: list[GitHubThreshold] = Field(default_factory=list)
    repo_maintained_points: list[GitHubRepoThreshold] = Field(default_factory=list)
    repo_relevant_points: list[GitHubRepoThreshold] = Field(default_factory=list)
    ai_keywords: list[str] = Field(default_factory=list)


class EngineeringDepthConfig(BaseModel):
    signals: list[str] = Field(default_factory=list)


class CapsConfig(BaseModel):
    no_meaningful_ai_project: int = 55


class EligibilityConfig(BaseModel):
    python_direct: list[str] = Field(default_factory=list)
    python_exclude: list[str] = Field(default_factory=list)
    python_implied: list[str] = Field(default_factory=list)
    ai_frameworks: list[str] = Field(default_factory=list)
    ai_concepts: list[str] = Field(default_factory=list)
    ai_provider_apis: list[str] = Field(default_factory=list)
    ai_extended: list[str] = Field(default_factory=list)
    ml_only_terms: list[str] = Field(default_factory=list)


class Settings(BaseModel):
    """Typed, validated application settings."""

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # GitHub
    github_token: str = ""

    # Concurrency / timeouts
    max_concurrency: int = 5
    llm_timeout_seconds: int = 60
    github_timeout_seconds: int = 10

    # Paths
    default_input_dir: str = "./resumes"
    default_output_path: str = "./output/results.json"

    # Loaded from YAML
    score_weights: ScoreWeights = Field(default_factory=ScoreWeights)
    ai_project: AIProjectConfig = Field(default_factory=AIProjectConfig)
    python_backend: PythonBackendConfig = Field(default_factory=PythonBackendConfig)
    cloud_fullstack: CloudFullstackConfig = Field(default_factory=CloudFullstackConfig)
    github_config: GitHubConfig = Field(default_factory=GitHubConfig)
    engineering_depth: EngineeringDepthConfig = Field(default_factory=EngineeringDepthConfig)
    caps: CapsConfig = Field(default_factory=CapsConfig)
    eligibility: EligibilityConfig = Field(default_factory=EligibilityConfig)
    skill_aliases: dict[str, list[str]] = Field(default_factory=dict)
    github_reserved_paths: list[str] = Field(default_factory=list)


def load_settings() -> Settings:
    """
    Build a Settings object from YAML + environment variables.

    Env vars override YAML for: LLM_PROVIDER, LLM_MODEL, MAX_CONCURRENCY,
    LLM_TIMEOUT_SECONDS, GITHUB_TIMEOUT_SECONDS, API keys, paths.
    """
    # Try loading .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    yaml_data = _load_yaml()

    # Map YAML sections to Settings fields
    kwargs: dict[str, Any] = {}

    # Direct YAML sections
    if "score_weights" in yaml_data:
        kwargs["score_weights"] = yaml_data["score_weights"]
    if "ai_project" in yaml_data:
        kwargs["ai_project"] = yaml_data["ai_project"]
    if "python_backend" in yaml_data:
        kwargs["python_backend"] = yaml_data["python_backend"]
    if "cloud_fullstack" in yaml_data:
        kwargs["cloud_fullstack"] = yaml_data["cloud_fullstack"]
    if "github" in yaml_data:
        kwargs["github_config"] = yaml_data["github"]
    if "engineering_depth" in yaml_data:
        kwargs["engineering_depth"] = yaml_data["engineering_depth"]
    if "caps" in yaml_data:
        kwargs["caps"] = yaml_data["caps"]
    if "eligibility" in yaml_data:
        kwargs["eligibility"] = yaml_data["eligibility"]
    if "skill_aliases" in yaml_data:
        kwargs["skill_aliases"] = yaml_data["skill_aliases"]
    if "github_reserved_paths" in yaml_data:
        kwargs["github_reserved_paths"] = yaml_data["github_reserved_paths"]

    # Env-var overrides (these take priority over YAML)
    env_map = {
        "LLM_PROVIDER": "llm_provider",
        "LLM_MODEL": "llm_model",
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "OPENAI_API_KEY": "openai_api_key",
        "GEMINI_API_KEY": "gemini_api_key",
        "GITHUB_TOKEN": "github_token",
        "MAX_CONCURRENCY": "max_concurrency",
        "LLM_TIMEOUT_SECONDS": "llm_timeout_seconds",
        "GITHUB_TIMEOUT_SECONDS": "github_timeout_seconds",
        "DEFAULT_INPUT_DIR": "default_input_dir",
        "DEFAULT_OUTPUT_PATH": "default_output_path",
    }
    for env_key, field_name in env_map.items():
        val = os.environ.get(env_key)
        if val is not None and val != "":
            # Convert numeric fields
            if field_name in ("max_concurrency", "llm_timeout_seconds", "github_timeout_seconds"):
                kwargs[field_name] = int(val)
            else:
                kwargs[field_name] = val

    return Settings(**kwargs)
