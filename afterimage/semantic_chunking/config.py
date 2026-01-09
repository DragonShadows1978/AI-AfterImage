"""
Semantic Chunking Configuration.

Provides YAML/environment-based configuration for the semantic chunking system.
Integrates with AfterImage's existing config system.

Part of AfterImage Semantic Chunking v0.3.0.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

import yaml


@dataclass
class SemanticChunkingConfig:
    """Configuration for semantic chunking system."""
    # Master enable/disable
    enabled: bool = True

    # Token budget settings
    max_tokens: int = 2000

    # Chunking settings
    chunk_enabled: bool = True
    max_chunk_tokens: int = 500

    # Relevance scoring weights (should sum to 1.0)
    recency_weight: float = 0.20
    proximity_weight: float = 0.25
    semantic_weight: float = 0.35
    project_weight: float = 0.20
    min_relevance_score: float = 0.3

    # Summarization settings
    summary_enabled: bool = True
    similarity_threshold: float = 0.7
    max_individual_snippets: int = 3
    max_results: int = 5
    summary_mode_threshold: int = 3

    # Cache settings
    cache_enabled: bool = True
    cache_max_entries: int = 100
    cache_ttl_seconds: float = 3600

    # Graceful degradation
    fallback_on_error: bool = True
    log_errors: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enabled": self.enabled,
            "max_tokens": self.max_tokens,
            "chunking": {
                "enabled": self.chunk_enabled,
                "max_chunk_tokens": self.max_chunk_tokens,
            },
            "scoring": {
                "recency_weight": self.recency_weight,
                "proximity_weight": self.proximity_weight,
                "semantic_weight": self.semantic_weight,
                "project_weight": self.project_weight,
                "min_relevance_score": self.min_relevance_score,
            },
            "summarization": {
                "enabled": self.summary_enabled,
                "similarity_threshold": self.similarity_threshold,
                "max_individual_snippets": self.max_individual_snippets,
                "max_results": self.max_results,
                "summary_mode_threshold": self.summary_mode_threshold,
            },
            "cache": {
                "enabled": self.cache_enabled,
                "max_entries": self.cache_max_entries,
                "ttl_seconds": self.cache_ttl_seconds,
            },
            "fallback_on_error": self.fallback_on_error,
            "log_errors": self.log_errors,
        }


def get_config_path() -> Path:
    """Get path to AfterImage config file."""
    return Path.home() / ".afterimage" / "config.yaml"


def get_default_config() -> SemanticChunkingConfig:
    """Get default semantic chunking configuration."""
    return SemanticChunkingConfig()


def load_semantic_config(config_path: Optional[Path] = None) -> SemanticChunkingConfig:
    """
    Load semantic chunking configuration from YAML file.

    Priority (highest to lowest):
    1. Environment variables (AFTERIMAGE_SEMANTIC_*)
    2. Config file values (semantic_chunking section)
    3. Default values

    Args:
        config_path: Path to config file. Defaults to ~/.afterimage/config.yaml

    Returns:
        Loaded configuration
    """
    if config_path is None:
        config_path = get_config_path()

    config = SemanticChunkingConfig()

    # Load from file if exists
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            config = _merge_config(config, data)
        except Exception:
            pass  # Use defaults on error

    # Apply environment variable overrides
    config = apply_env_overrides(config)

    return config


def _merge_config(config: SemanticChunkingConfig, data: Dict[str, Any]) -> SemanticChunkingConfig:
    """Merge YAML data into config dataclass."""
    sc = data.get("semantic_chunking", {})
    if not sc:
        return config

    # Master enable
    if "enabled" in sc:
        config.enabled = bool(sc["enabled"])

    # Token budget
    if "max_tokens" in sc:
        config.max_tokens = int(sc["max_tokens"])

    # Chunking
    chunking = sc.get("chunking", {})
    if "enabled" in chunking:
        config.chunk_enabled = bool(chunking["enabled"])
    if "max_chunk_tokens" in chunking:
        config.max_chunk_tokens = int(chunking["max_chunk_tokens"])

    # Scoring
    scoring = sc.get("scoring", {})
    if "recency_weight" in scoring:
        config.recency_weight = float(scoring["recency_weight"])
    if "proximity_weight" in scoring:
        config.proximity_weight = float(scoring["proximity_weight"])
    if "semantic_weight" in scoring:
        config.semantic_weight = float(scoring["semantic_weight"])
    if "project_weight" in scoring:
        config.project_weight = float(scoring["project_weight"])
    if "min_relevance_score" in scoring:
        config.min_relevance_score = float(scoring["min_relevance_score"])

    # Summarization
    summarization = sc.get("summarization", {})
    if "enabled" in summarization:
        config.summary_enabled = bool(summarization["enabled"])
    if "similarity_threshold" in summarization:
        config.similarity_threshold = float(summarization["similarity_threshold"])
    if "max_individual_snippets" in summarization:
        config.max_individual_snippets = int(summarization["max_individual_snippets"])
    if "max_results" in summarization:
        config.max_results = int(summarization["max_results"])
    if "summary_mode_threshold" in summarization:
        config.summary_mode_threshold = int(summarization["summary_mode_threshold"])

    # Cache
    cache = sc.get("cache", {})
    if "enabled" in cache:
        config.cache_enabled = bool(cache["enabled"])
    if "max_entries" in cache:
        config.cache_max_entries = int(cache["max_entries"])
    if "ttl_seconds" in cache:
        config.cache_ttl_seconds = float(cache["ttl_seconds"])

    # Fallback
    if "fallback_on_error" in sc:
        config.fallback_on_error = bool(sc["fallback_on_error"])
    if "log_errors" in sc:
        config.log_errors = bool(sc["log_errors"])

    return config


def apply_env_overrides(config: SemanticChunkingConfig) -> SemanticChunkingConfig:
    """Apply environment variable overrides to config."""
    # Master enable
    if os.environ.get("AFTERIMAGE_SEMANTIC_ENABLED"):
        config.enabled = os.environ["AFTERIMAGE_SEMANTIC_ENABLED"].lower() in ("1", "true", "yes")

    # Token budget
    if os.environ.get("AFTERIMAGE_SEMANTIC_MAX_TOKENS"):
        config.max_tokens = int(os.environ["AFTERIMAGE_SEMANTIC_MAX_TOKENS"])

    # Chunking
    if os.environ.get("AFTERIMAGE_SEMANTIC_CHUNK_ENABLED"):
        config.chunk_enabled = os.environ["AFTERIMAGE_SEMANTIC_CHUNK_ENABLED"].lower() in ("1", "true", "yes")
    if os.environ.get("AFTERIMAGE_SEMANTIC_MAX_CHUNK_TOKENS"):
        config.max_chunk_tokens = int(os.environ["AFTERIMAGE_SEMANTIC_MAX_CHUNK_TOKENS"])

    # Scoring weights
    if os.environ.get("AFTERIMAGE_SEMANTIC_RECENCY_WEIGHT"):
        config.recency_weight = float(os.environ["AFTERIMAGE_SEMANTIC_RECENCY_WEIGHT"])
    if os.environ.get("AFTERIMAGE_SEMANTIC_PROXIMITY_WEIGHT"):
        config.proximity_weight = float(os.environ["AFTERIMAGE_SEMANTIC_PROXIMITY_WEIGHT"])
    if os.environ.get("AFTERIMAGE_SEMANTIC_SEMANTIC_WEIGHT"):
        config.semantic_weight = float(os.environ["AFTERIMAGE_SEMANTIC_SEMANTIC_WEIGHT"])
    if os.environ.get("AFTERIMAGE_SEMANTIC_PROJECT_WEIGHT"):
        config.project_weight = float(os.environ["AFTERIMAGE_SEMANTIC_PROJECT_WEIGHT"])
    if os.environ.get("AFTERIMAGE_SEMANTIC_MIN_SCORE"):
        config.min_relevance_score = float(os.environ["AFTERIMAGE_SEMANTIC_MIN_SCORE"])

    # Summarization
    if os.environ.get("AFTERIMAGE_SEMANTIC_SUMMARY_ENABLED"):
        config.summary_enabled = os.environ["AFTERIMAGE_SEMANTIC_SUMMARY_ENABLED"].lower() in ("1", "true", "yes")
    if os.environ.get("AFTERIMAGE_SEMANTIC_SIMILARITY_THRESHOLD"):
        config.similarity_threshold = float(os.environ["AFTERIMAGE_SEMANTIC_SIMILARITY_THRESHOLD"])
    if os.environ.get("AFTERIMAGE_SEMANTIC_MAX_INDIVIDUAL"):
        config.max_individual_snippets = int(os.environ["AFTERIMAGE_SEMANTIC_MAX_INDIVIDUAL"])
    if os.environ.get("AFTERIMAGE_SEMANTIC_MAX_RESULTS"):
        config.max_results = int(os.environ["AFTERIMAGE_SEMANTIC_MAX_RESULTS"])

    # Cache
    if os.environ.get("AFTERIMAGE_SEMANTIC_CACHE_ENABLED"):
        config.cache_enabled = os.environ["AFTERIMAGE_SEMANTIC_CACHE_ENABLED"].lower() in ("1", "true", "yes")
    if os.environ.get("AFTERIMAGE_SEMANTIC_CACHE_TTL"):
        config.cache_ttl_seconds = float(os.environ["AFTERIMAGE_SEMANTIC_CACHE_TTL"])
    if os.environ.get("AFTERIMAGE_SEMANTIC_CACHE_MAX_ENTRIES"):
        config.cache_max_entries = int(os.environ["AFTERIMAGE_SEMANTIC_CACHE_MAX_ENTRIES"])

    return config


# Template for config file section
CONFIG_TEMPLATE = """
# Semantic Chunking Configuration
# Add this section to ~/.afterimage/config.yaml
semantic_chunking:
  enabled: true

  # Token budget (total injection size)
  max_tokens: 2000

  # Chunking settings
  chunking:
    enabled: true
    max_chunk_tokens: 500

  # Relevance scoring weights (must sum to 1.0)
  scoring:
    recency_weight: 0.20
    proximity_weight: 0.25
    semantic_weight: 0.35
    project_weight: 0.20
    min_relevance_score: 0.3

  # Summarization (for similar snippets)
  summarization:
    enabled: true
    similarity_threshold: 0.7
    max_individual_snippets: 3
    max_results: 5
    summary_mode_threshold: 3  # Min group size for summary mode

  # Caching
  cache:
    enabled: true
    max_entries: 100
    ttl_seconds: 3600

  # Error handling
  fallback_on_error: true
  log_errors: true
"""


def get_config_template() -> str:
    """Get the config template for documentation."""
    return CONFIG_TEMPLATE.strip()
