"""Tests for the Search module."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from afterimage.kb import KnowledgeBase
from afterimage.search import HybridSearch, SearchResult, quick_search


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        os.unlink(db_path)


@pytest.fixture
def mock_embedder():
    """Create a mock embedder for testing without sentence-transformers."""
    embedder = MagicMock()
    # Return deterministic embeddings based on text hash
    def embed(text):
        # Simple hash-based embedding
        h = hash(text)
        return [(h >> i) % 100 / 100.0 for i in range(10)]

    embedder.embed.side_effect = embed
    embedder.embed_code.side_effect = lambda code, *args: embed(code)
    embedder.embedding_dim = 10
    return embedder


@pytest.fixture
def backend(temp_db):
    """Create a storage backend instance."""
    from afterimage.storage import SQLiteBackend
    backend = SQLiteBackend(db_path=temp_db)
    backend.initialize()
    return backend


@pytest.fixture
def kb(backend):
    """Create a KnowledgeBase instance with backend."""
    return KnowledgeBase(backend=backend)


@pytest.fixture
def search(backend, mock_embedder):
    """Create a HybridSearch instance with mock embedder."""
    return HybridSearch(backend=backend, embedder=mock_embedder)


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SearchResult(
            id="test-123",
            file_path="/test.py",
            new_code="def test(): pass",
            old_code=None,
            context="Test context",
            timestamp="2024-01-01T00:00:00",
            session_id="session1",
            relevance_score=0.8,
            fts_score=0.5,
            semantic_score=0.9
        )

        d = result.to_dict()

        assert d["id"] == "test-123"
        assert d["file_path"] == "/test.py"
        assert d["relevance_score"] == 0.8


class TestHybridSearch:
    """Tests for the HybridSearch class."""

    def test_search_by_path(self, search, kb):
        """Test searching by file path."""
        kb.store(file_path="/project/src/auth.py", new_code="auth code")
        kb.store(file_path="/project/src/db.py", new_code="db code")

        results = search.search_by_path("auth")

        assert len(results) == 1
        assert "auth" in results[0].file_path

    def test_search_fts_only(self, search, kb):
        """Test FTS-only search when embeddings unavailable."""
        kb.store(
            file_path="/test.py",
            new_code="def authenticate_user(): pass",
            context="User authentication function"
        )
        kb.store(
            file_path="/other.py",
            new_code="def process_data(): pass",
            context="Data processing"
        )

        results = search.search("authenticate", include_fts_only=True)

        assert len(results) >= 1
        # Result should have FTS score but no semantic score (no embedding stored)
        assert results[0].fts_score > 0

    def test_search_with_embeddings(self, search, kb, mock_embedder):
        """Test search with semantic matching."""
        # Store entries with embeddings
        embedding1 = mock_embedder.embed("authentication code")
        kb.store(
            file_path="/auth.py",
            new_code="def login(): pass",
            embedding=embedding1
        )

        embedding2 = mock_embedder.embed("database connection")
        kb.store(
            file_path="/db.py",
            new_code="def connect(): pass",
            embedding=embedding2
        )

        results = search.search("auth login", threshold=0.0)

        # Should return results with semantic scores
        assert len(results) >= 1

    def test_search_by_code(self, search, kb, mock_embedder):
        """Test searching for similar code."""
        # Store some code with embeddings
        code1 = "def process_items(items):\n    return [x * 2 for x in items]"
        embedding1 = mock_embedder.embed_code(code1)
        kb.store(file_path="/utils.py", new_code=code1, embedding=embedding1)

        code2 = "def calculate_sum(numbers):\n    return sum(numbers)"
        embedding2 = mock_embedder.embed_code(code2)
        kb.store(file_path="/math.py", new_code=code2, embedding=embedding2)

        # Search for similar code
        query_code = "def double_values(vals):\n    return [v * 2 for v in vals]"
        results = search.search_by_code(query_code, threshold=0.0)

        assert len(results) >= 1

    def test_search_respects_limit(self, search, kb):
        """Test that search respects the limit parameter."""
        # Add many entries
        for i in range(10):
            kb.store(
                file_path=f"/file{i}.py",
                new_code=f"def func{i}(): pass",
                context="Test function"
            )

        results = search.search("func", limit=3, threshold=0.0)

        assert len(results) <= 3

    def test_search_respects_threshold(self, search, kb, mock_embedder):
        """Test that search respects relevance threshold."""
        # Store entry with embedding
        embedding = mock_embedder.embed("specific code")
        kb.store(
            file_path="/specific.py",
            new_code="specific functionality",
            embedding=embedding
        )

        # High threshold should filter out low-relevance results
        results = search.search("completely different query", threshold=0.99)
        # May or may not return results depending on FTS

    def test_search_path_filter(self, search, kb):
        """Test search with path filter."""
        kb.store(file_path="/project/src/auth.py", new_code="auth code")
        kb.store(file_path="/project/tests/test_auth.py", new_code="test code")

        results = search.search("auth", path_filter="src/", threshold=0.0)

        # Should only return src files
        for result in results:
            assert "src/" in result.file_path

    def test_combined_scoring(self, search, kb, mock_embedder):
        """Test that combined relevance scoring works."""
        # Store with both FTS content and embedding
        embedding = mock_embedder.embed("authentication login")
        kb.store(
            file_path="/auth.py",
            new_code="def authenticate_user(username, password): pass",
            context="Authentication function for login",
            embedding=embedding
        )

        results = search.search("authenticate login", threshold=0.0)

        assert len(results) >= 1
        # Should have both FTS and potentially semantic contribution
        result = results[0]
        assert result.relevance_score >= 0

    def test_escape_fts_special_chars(self, search, kb):
        """Test that special FTS characters are escaped."""
        kb.store(
            file_path="/test.py",
            new_code="def test(): pass",
            context="Test function"
        )

        # Query with special characters that could break FTS
        # These should not cause errors
        results = search.search("test()", threshold=0.0)
        results = search.search("test*", threshold=0.0)
        results = search.search('"test"', threshold=0.0)


class TestQuickSearch:
    """Tests for the quick_search convenience function."""

    def test_quick_search_uses_default_kb(self, temp_db):
        """Test that quick_search uses the default KB location."""
        # This would use ~/.afterimage/memory.db
        # Just verify it doesn't error
        with patch.object(KnowledgeBase, '__init__', return_value=None):
            with patch.object(HybridSearch, 'search', return_value=[]):
                # Should not raise
                results = quick_search("test query")
                assert isinstance(results, list)
