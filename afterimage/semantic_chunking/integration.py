"""
AfterImage Integration: Connect semantic chunking to AfterImage hook system.

This module provides the bridge between:
1. SmartContextInjector (semantic chunking, relevance scoring, summarization)
2. AfterImage hook (the Claude Code hook that injects context)
3. AfterImage embeddings (for semantic similarity scoring)

Part of AfterImage Semantic Chunking v0.3.0.
"""

import sys
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import math

from .config import SemanticChunkingConfig, load_semantic_config

# Configure logging
logger = logging.getLogger("afterimage.semantic")


@dataclass
class IntegrationConfig:
    """Configuration for AfterImage integration."""
    # Chunking settings
    max_chunk_tokens: int = 500
    chunk_enabled: bool = True

    # Token budget
    max_tokens: int = 2000

    # Relevance scoring weights
    recency_weight: float = 0.20
    proximity_weight: float = 0.25
    semantic_weight: float = 0.35
    project_weight: float = 0.20
    min_relevance_score: float = 0.3

    # Summarization
    summary_enabled: bool = True
    similarity_threshold: float = 0.7
    max_individual_snippets: int = 3
    max_results: int = 5

    # Caching
    cache_enabled: bool = True
    cache_max_entries: int = 100
    cache_ttl_seconds: float = 3600

    # Graceful degradation
    fallback_on_error: bool = True
    log_errors: bool = True

    @classmethod
    def from_semantic_config(cls, config: SemanticChunkingConfig) -> "IntegrationConfig":
        """Create IntegrationConfig from SemanticChunkingConfig."""
        return cls(
            max_chunk_tokens=config.max_chunk_tokens,
            chunk_enabled=config.chunk_enabled,
            max_tokens=config.max_tokens,
            recency_weight=config.recency_weight,
            proximity_weight=config.proximity_weight,
            semantic_weight=config.semantic_weight,
            project_weight=config.project_weight,
            min_relevance_score=config.min_relevance_score,
            summary_enabled=config.summary_enabled,
            similarity_threshold=config.similarity_threshold,
            max_individual_snippets=config.max_individual_snippets,
            max_results=config.max_results,
            cache_enabled=config.cache_enabled,
            cache_max_entries=config.cache_max_entries,
            cache_ttl_seconds=config.cache_ttl_seconds,
            fallback_on_error=config.fallback_on_error,
            log_errors=config.log_errors,
        )


# Track AfterImage availability
_afterimage_available: Optional[bool] = None
_embedder_class = None


def _check_afterimage_available() -> bool:
    """Check if AfterImage embeddings are available."""
    global _afterimage_available, _embedder_class

    if _afterimage_available is not None:
        return _afterimage_available

    try:
        from ..embeddings import EmbeddingGenerator
        _embedder_class = EmbeddingGenerator
        _afterimage_available = True
        logger.info("AfterImage embeddings available")
        return True
    except ImportError as e:
        logger.warning(f"AfterImage embeddings not available: {e}")
        _afterimage_available = False
        return False


class AfterImageEmbeddingAdapter:
    """
    Adapter to use AfterImage's embedding system with our RelevanceScorer.

    This class wraps AfterImage's EmbeddingGenerator to provide embeddings
    for semantic similarity scoring in the relevance scorer.
    """

    def __init__(self):
        """Initialize the embedding adapter."""
        self._embedder = None
        self._available = _check_afterimage_available()

    @property
    def available(self) -> bool:
        """Check if embeddings are available."""
        return self._available

    @property
    def embedder(self):
        """Lazy load the embedder."""
        if self._embedder is None and self._available:
            try:
                self._embedder = _embedder_class()
                logger.info("AfterImage embedder loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load AfterImage embedder: {e}")
                self._available = False
        return self._embedder

    def embed_query(self, query: str) -> Optional[List[float]]:
        """
        Generate embedding for a query string.

        Args:
            query: Text to embed

        Returns:
            384-dimensional embedding vector, or None if unavailable
        """
        if not self._available or self.embedder is None:
            return None

        try:
            return self.embedder.embed(query)
        except Exception as e:
            logger.warning(f"Embedding query failed: {e}")
            return None

    def embed_code(
        self,
        code: str,
        file_path: Optional[str] = None
    ) -> Optional[List[float]]:
        """
        Generate embedding for code.

        Args:
            code: Code to embed
            file_path: Optional file path for context

        Returns:
            384-dimensional embedding vector, or None if unavailable
        """
        if not self._available or self.embedder is None:
            return None

        try:
            return self.embedder.embed_code(code, file_path)
        except Exception as e:
            logger.warning(f"Embedding code failed: {e}")
            return None

    def compute_similarity(
        self,
        query_embedding: List[float],
        code_embedding: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            query_embedding: Query embedding vector
            code_embedding: Code embedding vector

        Returns:
            Similarity score in range [0, 1]
        """
        if not query_embedding or not code_embedding:
            return 0.0

        try:
            dot_product = sum(a * b for a, b in zip(query_embedding, code_embedding))
            norm_a = math.sqrt(sum(a * a for a in query_embedding))
            norm_b = math.sqrt(sum(b * b for b in code_embedding))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            sim = dot_product / (norm_a * norm_b)
            return (sim + 1) / 2
        except Exception:
            return 0.0


class SemanticContextInjector:
    """
    Enhanced context injector with AfterImage integration.

    Combines:
    - SmartContextInjector (chunking, budget, scoring, summarization)
    - AfterImage embeddings (semantic similarity)
    - Chunk caching (performance)
    - Graceful degradation (reliability)
    """

    def __init__(self, config: Optional[IntegrationConfig] = None):
        """
        Initialize the semantic context injector.

        Args:
            config: Integration configuration (uses defaults if None)
        """
        if config is None:
            # Try to load from YAML config
            semantic_config = load_semantic_config()
            config = IntegrationConfig.from_semantic_config(semantic_config)

        self.config = config

        # Initialize components lazily
        self._smart_injector = None
        self._embedding_adapter = None
        self._chunk_cache = None

        # Statistics for monitoring
        self._stats = {
            "injections_total": 0,
            "injections_with_embeddings": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "fallback_activations": 0,
            "errors": 0,
        }

    @property
    def smart_injector(self):
        """Lazy load SmartContextInjector."""
        if self._smart_injector is None:
            try:
                from .smart_injector import SmartContextInjector, SmartInjectionConfig

                injection_config = SmartInjectionConfig(
                    max_tokens=self.config.max_tokens,
                    max_chunk_tokens=self.config.max_chunk_tokens,
                    chunk_enabled=self.config.chunk_enabled,
                    min_relevance_score=self.config.min_relevance_score,
                    recency_weight=self.config.recency_weight,
                    proximity_weight=self.config.proximity_weight,
                    semantic_weight=self.config.semantic_weight,
                    project_weight=self.config.project_weight,
                    summary_enabled=self.config.summary_enabled,
                    similarity_threshold=self.config.similarity_threshold,
                    max_individual_snippets=self.config.max_individual_snippets,
                    max_results=self.config.max_results,
                )
                self._smart_injector = SmartContextInjector(injection_config)
                logger.info("SmartContextInjector loaded")
            except Exception as e:
                logger.error(f"Failed to load SmartContextInjector: {e}")
                self._stats["errors"] += 1
        return self._smart_injector

    @property
    def embedding_adapter(self) -> AfterImageEmbeddingAdapter:
        """Lazy load embedding adapter."""
        if self._embedding_adapter is None:
            self._embedding_adapter = AfterImageEmbeddingAdapter()
        return self._embedding_adapter

    @property
    def chunk_cache(self):
        """Lazy load chunk cache."""
        if self._chunk_cache is None and self.config.cache_enabled:
            try:
                from .chunk_cache import ChunkCache
                self._chunk_cache = ChunkCache(
                    max_entries=self.config.cache_max_entries,
                    ttl_seconds=self.config.cache_ttl_seconds,
                    enabled=True
                )
                logger.info("Chunk cache initialized")
            except Exception as e:
                logger.warning(f"Chunk cache not available: {e}")
        return self._chunk_cache

    def inject_context(
        self,
        search_results: List[Dict[str, Any]],
        file_path: str,
        tool_type: str = "Write"
    ) -> Optional[str]:
        """
        Create optimized context injection from AfterImage search results.

        This is the main entry point for the AfterImage hook.

        Args:
            search_results: Results from AfterImage's HybridSearch
            file_path: Path of file being written/edited
            tool_type: "Write" or "Edit"

        Returns:
            Formatted injection string, or None if no relevant context
        """
        self._stats["injections_total"] += 1

        if not search_results:
            return None

        try:
            # Try embedding-enhanced injection first
            if self.embedding_adapter.available:
                result = self._inject_with_embeddings(search_results, file_path, tool_type)
                if result:
                    self._stats["injections_with_embeddings"] += 1
                    return result

            # Fall back to basic injection
            if self.config.fallback_on_error:
                return self._inject_basic(search_results, file_path, tool_type)

        except Exception as e:
            self._stats["errors"] += 1
            if self.config.log_errors:
                logger.error(f"Injection failed: {e}")

            # Graceful degradation
            if self.config.fallback_on_error:
                self._stats["fallback_activations"] += 1
                return self._inject_basic(search_results, file_path, tool_type)

        return None

    def _inject_with_embeddings(
        self,
        search_results: List[Dict[str, Any]],
        file_path: str,
        tool_type: str
    ) -> Optional[str]:
        """Inject with semantic embedding enhancement."""
        if self.smart_injector is None:
            return None

        # Convert search results to format expected by SmartContextInjector
        raw_results = []
        for r in search_results:
            # Handle both SearchResult objects and dicts
            if hasattr(r, 'to_dict'):
                result_dict = r.to_dict()
            else:
                result_dict = r

            raw_results.append({
                "code": result_dict.get("new_code", result_dict.get("code", "")),
                "new_code": result_dict.get("new_code", result_dict.get("code", "")),
                "file_path": result_dict.get("file_path", ""),
                "timestamp": result_dict.get("timestamp", ""),
                "context": result_dict.get("context", ""),
                "semantic_score": result_dict.get("semantic_score", 0.0),
            })

        # Use the hook-optimized injection method
        return self.smart_injector.inject_for_hook(raw_results, file_path, tool_type)

    def _inject_basic(
        self,
        search_results: List[Dict[str, Any]],
        file_path: str,
        tool_type: str
    ) -> Optional[str]:
        """Basic injection without semantic enhancement (fallback)."""
        if not search_results:
            return None

        # Simple formatting similar to the original hook
        action = "creating" if tool_type == "Write" else "editing"
        file_name = Path(file_path).name

        lines = [
            f'<memory context="You are {action} {file_name}">',
            "Previously written similar code:",
            ""
        ]

        seen_paths = set()
        for r in search_results[:3]:  # Limit to 3
            if hasattr(r, 'file_path'):
                result_path = r.file_path
                code = r.new_code if hasattr(r, 'new_code') else ""
            else:
                result_path = r.get("file_path", "")
                code = r.get("new_code", r.get("code", ""))

            short_path = "/".join(Path(result_path).parts[-3:]) if result_path else "unknown"
            if short_path in seen_paths:
                continue
            seen_paths.add(short_path)

            preview = code[:400] if code else ""
            lines.append(f"**From:** `{short_path}`")
            lines.append("```")
            lines.append(preview)
            if len(code) > 400:
                lines.append("... (truncated)")
            lines.append("```")
            lines.append("")

        lines.append("</memory>")
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get injection statistics for monitoring."""
        stats = dict(self._stats)

        # Add cache stats if available
        if self.chunk_cache:
            stats["cache"] = self.chunk_cache.get_summary()

        # Add embedding status
        stats["embeddings_available"] = self.embedding_adapter.available

        return stats


def create_semantic_injector(
    max_tokens: int = 2000,
    cache_enabled: bool = True
) -> SemanticContextInjector:
    """
    Create a configured SemanticContextInjector.

    Args:
        max_tokens: Maximum tokens for injection
        cache_enabled: Whether to enable chunk caching

    Returns:
        Configured SemanticContextInjector instance
    """
    config = IntegrationConfig(
        max_tokens=max_tokens,
        cache_enabled=cache_enabled
    )
    return SemanticContextInjector(config)


# Singleton instance for hook integration
_global_injector: Optional[SemanticContextInjector] = None


def get_semantic_injector() -> SemanticContextInjector:
    """Get or create the global semantic injector instance."""
    global _global_injector
    if _global_injector is None:
        _global_injector = create_semantic_injector()
    return _global_injector


def inject_semantic_context(
    search_results: List[Dict[str, Any]],
    file_path: str,
    tool_type: str = "Write"
) -> Optional[str]:
    """
    Convenience function for hook integration.

    Args:
        search_results: AfterImage search results
        file_path: File being written/edited
        tool_type: "Write" or "Edit"

    Returns:
        Formatted injection string or None
    """
    injector = get_semantic_injector()
    return injector.inject_context(search_results, file_path, tool_type)
