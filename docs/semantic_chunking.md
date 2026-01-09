# AfterImage Semantic Chunking

## Overview

The Semantic Chunking module enhances AfterImage's context injection by breaking code into meaningful semantic units instead of raw file contents. This results in more relevant and compact context injections for Claude Code hooks.

**Version:** 0.3.0

## Features

- **Semantic Code Parsing**: Uses Python AST for Python files, regex patterns for JavaScript/TypeScript/Rust/Go/C/C++
- **Multi-Factor Relevance Scoring**: Combines recency (20%), proximity (25%), semantic similarity (35%), and project awareness (20%)
- **Token Budget Management**: 5 tiers from 500 to 8000 tokens with automatic truncation
- **Snippet Summarization**: Groups similar snippets and uses summary mode for 3+ similar items
- **LRU Caching**: Content-hash invalidation with configurable TTL and max entries
- **Graceful Degradation**: Falls back to basic injection on errors

## Quick Start

```python
from afterimage.semantic_chunking import inject_semantic_context

# In your hook code:
results = kb.search_fts("authentication", limit=10)
injection = inject_semantic_context(
    results,
    file_path="/project/src/auth.py",
    tool_type="Write"
)
```

## Configuration

### YAML Configuration

Add to `~/.afterimage/config.yaml`:

```yaml
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
    summary_mode_threshold: 3

  # Caching
  cache:
    enabled: true
    max_entries: 100
    ttl_seconds: 3600

  # Error handling
  fallback_on_error: true
  log_errors: true
```

### Environment Variable Overrides

All settings can be overridden via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `AFTERIMAGE_SEMANTIC_ENABLED` | Enable/disable semantic chunking | `true` |
| `AFTERIMAGE_SEMANTIC_MAX_TOKENS` | Maximum injection tokens | `2000` |
| `AFTERIMAGE_SEMANTIC_CHUNK_ENABLED` | Enable code chunking | `true` |
| `AFTERIMAGE_SEMANTIC_MAX_CHUNK_TOKENS` | Max tokens per chunk | `500` |
| `AFTERIMAGE_SEMANTIC_RECENCY_WEIGHT` | Recency score weight | `0.20` |
| `AFTERIMAGE_SEMANTIC_PROXIMITY_WEIGHT` | Proximity score weight | `0.25` |
| `AFTERIMAGE_SEMANTIC_SEMANTIC_WEIGHT` | Semantic score weight | `0.35` |
| `AFTERIMAGE_SEMANTIC_PROJECT_WEIGHT` | Project score weight | `0.20` |
| `AFTERIMAGE_SEMANTIC_MIN_SCORE` | Minimum relevance score | `0.3` |
| `AFTERIMAGE_SEMANTIC_SUMMARY_ENABLED` | Enable summarization | `true` |
| `AFTERIMAGE_SEMANTIC_SIMILARITY_THRESHOLD` | Similarity threshold | `0.7` |
| `AFTERIMAGE_SEMANTIC_CACHE_ENABLED` | Enable caching | `true` |
| `AFTERIMAGE_SEMANTIC_CACHE_TTL` | Cache TTL in seconds | `3600` |
| `AFTERIMAGE_SEMANTIC_CACHE_MAX_ENTRIES` | Max cache entries | `100` |

## Components

### SemanticChunker

Parses source code into semantic units (functions, classes, methods, blocks).

```python
from afterimage.semantic_chunking import SemanticChunker

chunker = SemanticChunker(max_chunk_tokens=500)
chunks = chunker.chunk_code(code, "example.py")

for chunk in chunks:
    print(f"{chunk.chunk_type.value}: {chunk.name}")
    print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
    print(f"  Tokens: {chunk.token_count}")
```

Supported chunk types:
- `function` - Top-level functions
- `method` - Class methods
- `class` - Class definitions
- `imports` - Import blocks
- `constants` - Module-level constants
- `block` - Generic code blocks

### TokenBudgetManager

Manages token budgets for context injection.

```python
from afterimage.semantic_chunking import TokenBudgetManager, TokenBudgetTier

manager = TokenBudgetManager()

# Check if content fits
if manager.fits_in_budget(content):
    # Use content
    pass

# Token tiers
# MINIMAL   = 500 tokens
# COMPACT   = 1000 tokens
# STANDARD  = 2000 tokens (default)
# GENEROUS  = 4000 tokens
# EXTENSIVE = 8000 tokens
```

### RelevanceScorer

Scores code snippets using multiple factors.

```python
from afterimage.semantic_chunking import RelevanceScorer, ScoringConfig

config = ScoringConfig(
    recency_weight=0.20,
    proximity_weight=0.25,
    semantic_weight=0.35,
    project_weight=0.20
)

scorer = RelevanceScorer(config)
scorer.set_context(
    current_file="/project/src/auth.py",
    current_project="/project"
)

scored = scorer.score_snippets(snippets)
```

### SnippetSummarizer

Groups similar snippets and generates summaries.

```python
from afterimage.semantic_chunking import SnippetSummarizer, SummaryConfig

config = SummaryConfig(
    similarity_threshold=0.7,
    summary_mode_threshold=3  # Use summary mode for 3+ similar
)

summarizer = SnippetSummarizer(config)
individual, groups = summarizer.summarize(snippets, max_output=5)
```

### SmartContextInjector

Main orchestrator combining all components.

```python
from afterimage.semantic_chunking import SmartContextInjector, SmartInjectionConfig

config = SmartInjectionConfig(
    max_tokens=2000,
    summary_enabled=True
)

injector = SmartContextInjector(config)
injector.set_context(current_file="/project/src/auth.py")

result = injector.inject(search_results)
print(f"Tokens used: {result.tokens_used}")
print(f"Snippets included: {result.snippets_included}")
print(result.injection_text)
```

### ChunkCache

LRU cache for chunking results.

```python
from afterimage.semantic_chunking import ChunkCache, get_chunk_cache

# Get global cache
cache = get_chunk_cache()

# Check cache
cached = cache.get(file_path, content, max_chunk_tokens=500)
if cached:
    chunks = cached
else:
    chunks = chunker.chunk_code(content, file_path)
    cache.put(file_path, content, chunks, max_chunk_tokens=500)

# View stats
print(cache.get_summary())
```

## Hook Integration

The semantic chunking system integrates with the AfterImage hook via `SemanticContextInjector`:

```python
from afterimage.semantic_chunking import get_semantic_injector, inject_semantic_context

# Option 1: Use the singleton injector
injector = get_semantic_injector()
output = injector.inject_context(results, file_path, tool_type)

# Option 2: Use the convenience function
output = inject_semantic_context(results, file_path, tool_type)
```

The hook automatically:
1. Tries semantic injection first
2. Falls back to basic injection if semantic fails
3. Can be disabled via `AFTERIMAGE_SEMANTIC_ENABLED=0`

## Performance

- **Chunking latency**: < 100ms for typical files
- **Injection latency**: < 50ms for up to 10 search results
- **Memory usage**: < 100MB peak
- **Cache hit rate**: 60-80% typical (108x speedup on repeated operations)

## API Reference

### Main Functions

```python
# Inject context using semantic chunking
inject_semantic_context(
    search_results: List[Dict],
    file_path: str,
    tool_type: str = "Write"
) -> Optional[str]

# Get the singleton injector
get_semantic_injector() -> SemanticContextInjector

# Quick injection utility
quick_inject(
    raw_results: List[Dict],
    current_file: Optional[str] = None,
    max_tokens: int = 2000
) -> str

# Quick scoring utility
quick_score(
    snippets: List[Dict],
    current_file: Optional[str] = None,
    current_project: Optional[str] = None
) -> List[ScoredSnippet]
```

### Classes

- `SemanticChunker` - Code parsing
- `TokenBudgetManager` - Budget management
- `RelevanceScorer` - Multi-factor scoring
- `SnippetSummarizer` - Grouping and summarization
- `SmartContextInjector` - Main orchestrator
- `ChunkCache` - LRU caching
- `SemanticContextInjector` - Hook integration
- `SemanticChunkingConfig` - Configuration

### Configuration Classes

- `SmartInjectionConfig` - Injector settings
- `ScoringConfig` - Scoring weights
- `SummaryConfig` - Summarization settings
- `TokenBudgetConfig` - Budget settings

## Troubleshooting

### Semantic injection not working

1. Check if enabled: `echo $AFTERIMAGE_SEMANTIC_ENABLED`
2. Verify imports work: `python3 -c "from afterimage.semantic_chunking import inject_semantic_context; print('OK')"`
3. Check hook logs for errors

### High memory usage

1. Reduce cache size: `AFTERIMAGE_SEMANTIC_CACHE_MAX_ENTRIES=50`
2. Reduce token budget: `AFTERIMAGE_SEMANTIC_MAX_TOKENS=1000`

### Slow performance

1. Enable caching: `AFTERIMAGE_SEMANTIC_CACHE_ENABLED=1`
2. Reduce max results: `AFTERIMAGE_SEMANTIC_MAX_RESULTS=3`

### Context not relevant

1. Increase semantic weight: `AFTERIMAGE_SEMANTIC_SEMANTIC_WEIGHT=0.5`
2. Lower min score: `AFTERIMAGE_SEMANTIC_MIN_SCORE=0.2`
3. Increase results limit in search
