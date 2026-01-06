"""
Tests for the Context Injection module.

Tests cover:
- InjectionConfig defaults and custom settings
- ContextInjector formatting and output
- Token limits and truncation
- Relevance filtering
- Hook-specific formatting
"""

import pytest
from afterimage.inject import (
    ContextInjector,
    InjectionConfig,
    create_injection_message,
)
from afterimage.search import SearchResult


def make_result(
    id: str = "test_id",
    file_path: str = "/home/user/project/src/utils.py",
    new_code: str = "def hello():\n    return 'world'",
    old_code: str = None,
    context: str = "Added hello function for greeting",
    timestamp: str = "2026-01-06T12:00:00Z",
    session_id: str = "session_123",
    relevance_score: float = 0.8,
    fts_score: float = 0.5,
    semantic_score: float = 0.9
) -> SearchResult:
    """Factory for creating test SearchResult objects."""
    return SearchResult(
        id=id,
        file_path=file_path,
        new_code=new_code,
        old_code=old_code,
        context=context,
        timestamp=timestamp,
        session_id=session_id,
        relevance_score=relevance_score,
        fts_score=fts_score,
        semantic_score=semantic_score,
    )


class TestInjectionConfig:
    """Tests for InjectionConfig defaults and settings."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = InjectionConfig()
        assert config.max_results == 5
        assert config.max_tokens == 2000
        assert config.relevance_threshold == 0.4
        assert config.include_context is True
        assert config.include_file_path is True
        assert config.include_timestamp is False
        assert config.code_fence_language == ""

    def test_custom_values(self):
        """Config should accept custom values."""
        config = InjectionConfig(
            max_results=10,
            max_tokens=3000,
            relevance_threshold=0.6,
            include_context=False,
            include_file_path=False,
            include_timestamp=True,
            code_fence_language="python"
        )
        assert config.max_results == 10
        assert config.max_tokens == 3000
        assert config.relevance_threshold == 0.6
        assert config.include_context is False
        assert config.include_file_path is False
        assert config.include_timestamp is True
        assert config.code_fence_language == "python"


class TestContextInjector:
    """Tests for the ContextInjector class."""

    def test_format_injection_single_result(self):
        """Should format a single result properly."""
        injector = ContextInjector()
        results = [make_result()]

        output = injector.format_injection(results)

        assert output is not None
        assert "You have written similar code before:" in output
        assert "def hello():" in output
        assert "utils.py" in output
        assert "80%" in output  # Relevance percentage

    def test_format_injection_multiple_results(self):
        """Should format multiple results with count."""
        injector = ContextInjector()
        results = [
            make_result(id="1", file_path="/a/b/one.py", new_code="# one"),
            make_result(id="2", file_path="/a/b/two.py", new_code="# two"),
            make_result(id="3", file_path="/a/b/three.py", new_code="# three"),
        ]

        output = injector.format_injection(results)

        assert output is not None
        assert "(3 matches)" in output
        assert "Match 1" in output
        assert "Match 2" in output
        assert "Match 3" in output

    def test_format_injection_filters_by_relevance(self):
        """Should filter out low-relevance results."""
        config = InjectionConfig(relevance_threshold=0.5)
        injector = ContextInjector(config)
        results = [
            make_result(id="1", relevance_score=0.8),  # Above threshold
            make_result(id="2", relevance_score=0.3),  # Below threshold
            make_result(id="3", relevance_score=0.6),  # Above threshold
        ]

        output = injector.format_injection(results)

        assert output is not None
        assert "(2 matches)" in output

    def test_format_injection_respects_max_results(self):
        """Should limit results to max_results."""
        config = InjectionConfig(max_results=2)
        injector = ContextInjector(config)
        results = [
            make_result(id="1", file_path="/a/one.py"),
            make_result(id="2", file_path="/a/two.py"),
            make_result(id="3", file_path="/a/three.py"),
        ]

        output = injector.format_injection(results)

        assert "Match 1" in output
        assert "Match 2" in output
        assert "Match 3" not in output

    def test_format_injection_returns_none_for_no_relevant_results(self):
        """Should return None if no results meet threshold."""
        config = InjectionConfig(relevance_threshold=0.9)
        injector = ContextInjector(config)
        results = [
            make_result(id="1", relevance_score=0.5),
            make_result(id="2", relevance_score=0.6),
        ]

        output = injector.format_injection(results)

        assert output is None

    def test_format_injection_returns_none_for_empty_results(self):
        """Should return None for empty results list."""
        injector = ContextInjector()
        output = injector.format_injection([])
        assert output is None

    def test_format_injection_includes_context(self):
        """Should include context when configured."""
        config = InjectionConfig(include_context=True)
        injector = ContextInjector(config)
        results = [make_result(context="This is the context")]

        output = injector.format_injection(results)

        assert "**Context:**" in output
        assert "This is the context" in output

    def test_format_injection_excludes_context(self):
        """Should exclude context when configured."""
        config = InjectionConfig(include_context=False)
        injector = ContextInjector(config)
        results = [make_result(context="This is the context")]

        output = injector.format_injection(results)

        assert "**Context:**" not in output

    def test_format_injection_includes_timestamp(self):
        """Should include timestamp when configured."""
        config = InjectionConfig(include_timestamp=True)
        injector = ContextInjector(config)
        results = [make_result(timestamp="2026-01-06T12:00:00Z")]

        output = injector.format_injection(results)

        assert "[2026-01-06" in output

    def test_format_injection_token_limit_truncation(self):
        """Should truncate when exceeding token limit."""
        config = InjectionConfig(max_tokens=100)  # Very low limit
        injector = ContextInjector(config)
        # Create results with large code blocks
        results = [
            make_result(id="1", new_code="x = 1\n" * 100),
            make_result(id="2", new_code="y = 2\n" * 100),
            make_result(id="3", new_code="z = 3\n" * 100),
        ]

        output = injector.format_injection(results)

        # Should include truncation notice
        assert "truncated" in output.lower()


class TestContextInjectorLanguageDetection:
    """Tests for language detection in code fences."""

    def test_detects_python(self):
        """Should detect Python from .py extension."""
        injector = ContextInjector()
        results = [make_result(file_path="/path/to/file.py")]

        output = injector.format_injection(results)

        assert "```python" in output

    def test_detects_javascript(self):
        """Should detect JavaScript from .js extension."""
        injector = ContextInjector()
        results = [make_result(file_path="/path/to/file.js")]

        output = injector.format_injection(results)

        assert "```javascript" in output

    def test_detects_typescript(self):
        """Should detect TypeScript from .ts extension."""
        injector = ContextInjector()
        results = [make_result(file_path="/path/to/file.ts")]

        output = injector.format_injection(results)

        assert "```typescript" in output

    def test_detects_rust(self):
        """Should detect Rust from .rs extension."""
        injector = ContextInjector()
        results = [make_result(file_path="/path/to/file.rs")]

        output = injector.format_injection(results)

        assert "```rust" in output

    def test_respects_configured_language(self):
        """Should use configured language over detected."""
        config = InjectionConfig(code_fence_language="custom")
        injector = ContextInjector(config)
        results = [make_result(file_path="/path/to/file.py")]

        output = injector.format_injection(results)

        assert "```custom" in output

    def test_unknown_extension(self):
        """Should use empty fence for unknown extension."""
        injector = ContextInjector()
        results = [make_result(file_path="/path/to/file.xyz")]

        output = injector.format_injection(results)

        assert "```\n" in output


class TestContextInjectorTruncation:
    """Tests for code and context truncation."""

    def test_truncates_long_code(self):
        """Should truncate code that exceeds max lines."""
        injector = ContextInjector()
        long_code = "\n".join([f"line_{i} = {i}" for i in range(100)])
        results = [make_result(new_code=long_code)]

        output = injector.format_injection(results)

        assert "truncated" in output.lower()
        # Should have head and tail preserved
        assert "line_0" in output
        assert "line_99" in output

    def test_truncates_long_context(self):
        """Should truncate context that exceeds max chars."""
        injector = ContextInjector()
        long_context = "word " * 100
        results = [make_result(context=long_context)]

        output = injector.format_injection(results)

        assert "..." in output


class TestFormatForHook:
    """Tests for hook-specific formatting."""

    def test_format_for_hook_write(self):
        """Should format for Write tool correctly."""
        injector = ContextInjector()
        results = [make_result()]

        output = injector.format_for_hook(
            results=results,
            file_path="/home/user/project/src/main.py",
            tool_type="Write"
        )

        assert output is not None
        assert "<memory" in output
        assert "creating" in output
        assert "main.py" in output
        assert "</memory>" in output

    def test_format_for_hook_edit(self):
        """Should format for Edit tool correctly."""
        injector = ContextInjector()
        results = [make_result()]

        output = injector.format_for_hook(
            results=results,
            file_path="/home/user/project/src/main.py",
            tool_type="Edit"
        )

        assert output is not None
        assert "editing" in output

    def test_format_for_hook_filters_by_relevance(self):
        """Should filter results by relevance threshold."""
        config = InjectionConfig(relevance_threshold=0.7)
        injector = ContextInjector(config)
        results = [
            make_result(id="1", relevance_score=0.8),
            make_result(id="2", relevance_score=0.5),  # Below threshold
        ]

        output = injector.format_for_hook(
            results=results,
            file_path="/path/main.py",
            tool_type="Write"
        )

        # Should only include one result
        assert output is not None
        # Count occurrences of file markers
        assert output.count("---") == 2  # Opening and closing marker for 1 result

    def test_format_for_hook_returns_none_if_no_relevant(self):
        """Should return None if no relevant results."""
        config = InjectionConfig(relevance_threshold=0.9)
        injector = ContextInjector(config)
        results = [make_result(relevance_score=0.5)]

        output = injector.format_for_hook(
            results=results,
            file_path="/path/main.py",
            tool_type="Write"
        )

        assert output is None

    def test_format_for_hook_includes_why_context(self):
        """Should include 'Why' context in compact format."""
        injector = ContextInjector()
        results = [make_result(context="Added for performance")]

        output = injector.format_for_hook(
            results=results,
            file_path="/path/main.py",
            tool_type="Write"
        )

        assert "Why: Added for performance" in output


class TestFormatSingle:
    """Tests for single result formatting."""

    def test_format_single_basic(self):
        """Should format a single result without numbering."""
        injector = ContextInjector()
        result = make_result()

        output = injector.format_single(result)

        assert "### Previous Code" in output
        assert "Match" not in output
        assert "def hello():" in output


class TestConvenienceFunction:
    """Tests for create_injection_message convenience function."""

    def test_creates_injection_message(self):
        """Should create injection message using convenience function."""
        results = [make_result()]

        output = create_injection_message(results)

        assert output is not None
        assert "You have written similar code before" in output

    def test_accepts_custom_config(self):
        """Should accept custom config."""
        config = InjectionConfig(relevance_threshold=0.9)
        results = [make_result(relevance_score=0.5)]

        output = create_injection_message(results, config)

        assert output is None  # Below threshold


class TestTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens_basic(self):
        """Should estimate tokens at ~4 chars per token."""
        injector = ContextInjector()

        # 100 chars should be ~25 tokens
        tokens = injector._estimate_tokens("x" * 100)

        assert tokens == 25

    def test_estimate_tokens_empty(self):
        """Should handle empty string."""
        injector = ContextInjector()

        tokens = injector._estimate_tokens("")

        assert tokens == 0
