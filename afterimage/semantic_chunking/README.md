# Semantic Chunking Module

Part of AfterImage v0.3.0.

## Module Structure

```
semantic_chunking/
├── __init__.py           # Package exports
├── chunker.py            # SemanticChunker - AST/regex code parsing
├── token_budget.py       # TokenBudgetManager - token limit management
├── relevance_scorer.py   # RelevanceScorer - multi-factor scoring
├── snippet_summarizer.py # SnippetSummarizer - grouping & summary mode
├── smart_injector.py     # SmartContextInjector - main orchestrator
├── chunk_cache.py        # ChunkCache - LRU caching layer
├── config.py             # YAML configuration & env var overrides
├── integration.py        # SemanticContextInjector - hook integration
└── README.md             # This file
```

## Quick Usage

```python
# Simple injection
from afterimage.semantic_chunking import inject_semantic_context

injection = inject_semantic_context(
    search_results,  # From AfterImage search
    file_path="/project/src/new_file.py",
    tool_type="Write"
)
```

## Component Dependencies

```
SemanticContextInjector (integration.py)
        │
        └── SmartContextInjector (smart_injector.py)
                │
                ├── SemanticChunker (chunker.py)
                ├── TokenBudgetManager (token_budget.py)
                ├── RelevanceScorer (relevance_scorer.py)
                └── SnippetSummarizer (snippet_summarizer.py)
                        │
                        └── SummaryFormatter

ChunkCache (chunk_cache.py) - Global singleton, used by chunker
SemanticChunkingConfig (config.py) - YAML + env var configuration
```

## Configuration Priority

1. Environment variables (AFTERIMAGE_SEMANTIC_*)
2. YAML file (~/.afterimage/config.yaml)
3. Default values

## Performance Characteristics

| Operation | Target | Actual |
|-----------|--------|--------|
| Chunking (50 functions) | < 100ms | ~5ms |
| Full injection (10 results) | < 50ms | ~15ms |
| Cache hit | - | ~0.1ms |
| Memory usage | < 100MB | ~20MB |

## Integration Points

1. **Hook Integration**: `inject_semantic_context()` called from `afterimage_hook.py`
2. **Configuration**: Reads from AfterImage's config system
3. **Embeddings**: Uses AfterImage's `EmbeddingGenerator` when available
4. **Fallback**: Falls back to basic injection if semantic fails

## Tests

Run tests with:

```bash
cd /home/vader/mini-mind-v2/workspace/AI-AfterImage
python3 -m pytest tests/test_semantic_chunking_integration.py -v
```

All 41 tests pass, including:
- Import tests
- Chunker tests (Python, JS, TS, generic)
- Token budget tests
- Relevance scoring tests
- Summarization tests
- Smart injector tests
- Hook integration tests
- Configuration tests
- Performance tests
- Real database tests
- Graceful degradation tests
