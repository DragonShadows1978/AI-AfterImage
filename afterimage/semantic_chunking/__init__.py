"""
Semantic Chunking for AfterImage.

Intelligent context injection with:
- Multi-language code chunking (Python AST + regex fallbacks)
- Token budget management (5 tiers: 500-8000 tokens)
- Multi-factor relevance scoring (recency, proximity, semantic, project)
- Summary mode for similar snippet groups
- LRU caching for performance (108x speedup on repeated operations)

Part of AfterImage v0.3.0+
"""

from .chunker import SemanticChunker, CodeChunk, ChunkType, chunk_code_file
from .token_budget import (
    TokenBudgetManager, TokenBudgetConfig, TokenBudgetTier,
    TokenEstimator, create_token_budget
)
from .relevance_scorer import (
    RelevanceScorer, ScoringConfig, ScoredSnippet, quick_score
)
from .snippet_summarizer import (
    SnippetSummarizer, SummaryConfig, SnippetGroup,
    SummaryFormatter, summarize_snippets
)
from .smart_injector import (
    SmartContextInjector, SmartInjectionConfig, InjectionResult,
    ProjectContextManager, create_smart_injector, quick_inject
)
from .chunk_cache import (
    ChunkCache, CacheEntry, CacheStats,
    get_chunk_cache, clear_global_cache
)
from .config import (
    SemanticChunkingConfig, load_semantic_config,
    get_default_config, apply_env_overrides
)
from .integration import (
    SemanticContextInjector, IntegrationConfig,
    AfterImageEmbeddingAdapter, create_semantic_injector,
    get_semantic_injector, inject_semantic_context
)

__all__ = [
    # Chunker
    "SemanticChunker", "CodeChunk", "ChunkType", "chunk_code_file",
    # Token Budget
    "TokenBudgetManager", "TokenBudgetConfig", "TokenBudgetTier",
    "TokenEstimator", "create_token_budget",
    # Relevance Scoring
    "RelevanceScorer", "ScoringConfig", "ScoredSnippet", "quick_score",
    # Summarization
    "SnippetSummarizer", "SummaryConfig", "SnippetGroup",
    "SummaryFormatter", "summarize_snippets",
    # Smart Injection
    "SmartContextInjector", "SmartInjectionConfig", "InjectionResult",
    "ProjectContextManager", "create_smart_injector", "quick_inject",
    # Cache
    "ChunkCache", "CacheEntry", "CacheStats",
    "get_chunk_cache", "clear_global_cache",
    # Configuration
    "SemanticChunkingConfig", "load_semantic_config",
    "get_default_config", "apply_env_overrides",
    # Integration
    "SemanticContextInjector", "IntegrationConfig",
    "AfterImageEmbeddingAdapter", "create_semantic_injector",
    "get_semantic_injector", "inject_semantic_context",
]

__version__ = "0.3.0"
