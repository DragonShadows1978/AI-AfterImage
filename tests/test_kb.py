"""Tests for the Knowledge Base module."""

import os
import tempfile
import pytest
from pathlib import Path

from afterimage.kb import KnowledgeBase, serialize_embedding, deserialize_embedding


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        os.unlink(db_path)


@pytest.fixture
def kb(temp_db):
    """Create a KnowledgeBase instance with temporary database."""
    return KnowledgeBase(db_path=temp_db)


class TestEmbeddingSerialization:
    """Tests for embedding serialization/deserialization."""

    def test_serialize_deserialize_roundtrip(self):
        """Test that serialization and deserialization are inverses."""
        original = [0.1, 0.2, 0.3, -0.5, 0.0]
        serialized = serialize_embedding(original)
        deserialized = deserialize_embedding(serialized)

        assert len(original) == len(deserialized)
        for a, b in zip(original, deserialized):
            assert abs(a - b) < 1e-6

    def test_serialize_empty_embedding(self):
        """Test serialization of empty embedding."""
        empty = []
        serialized = serialize_embedding(empty)
        deserialized = deserialize_embedding(serialized)
        assert deserialized == []

    def test_serialize_large_embedding(self):
        """Test serialization of large embedding (384 dim like MiniLM)."""
        large = [i / 384.0 for i in range(384)]
        serialized = serialize_embedding(large)
        deserialized = deserialize_embedding(serialized)

        assert len(deserialized) == 384
        for a, b in zip(large, deserialized):
            assert abs(a - b) < 1e-6


class TestKnowledgeBase:
    """Tests for the KnowledgeBase class."""

    def test_init_creates_tables(self, kb):
        """Test that initialization creates required tables."""
        import sqlite3
        conn = sqlite3.connect(kb.db_path)
        cursor = conn.cursor()

        # Check main table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='code_memory'"
        )
        assert cursor.fetchone() is not None

        # Check FTS table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='code_memory_fts'"
        )
        assert cursor.fetchone() is not None

        conn.close()

    def test_store_and_get(self, kb):
        """Test storing and retrieving an entry."""
        entry_id = kb.store(
            file_path="/test/file.py",
            new_code="def hello(): pass",
            context="Test context"
        )

        assert entry_id is not None

        retrieved = kb.get(entry_id)
        assert retrieved is not None
        assert retrieved["file_path"] == "/test/file.py"
        assert retrieved["new_code"] == "def hello(): pass"
        assert retrieved["context"] == "Test context"

    def test_store_with_embedding(self, kb):
        """Test storing entry with embedding."""
        embedding = [0.1] * 384

        entry_id = kb.store(
            file_path="/test/embed.py",
            new_code="code",
            embedding=embedding
        )

        retrieved = kb.get(entry_id)
        assert retrieved["embedding"] is not None
        assert len(retrieved["embedding"]) == 384

    def test_store_edit(self, kb):
        """Test storing an edit with old and new code."""
        entry_id = kb.store(
            file_path="/test/edit.py",
            new_code="def new(): pass",
            old_code="def old(): pass",
            context="Refactoring"
        )

        retrieved = kb.get(entry_id)
        assert retrieved["old_code"] == "def old(): pass"
        assert retrieved["new_code"] == "def new(): pass"

    def test_search_by_path(self, kb):
        """Test searching by file path."""
        kb.store(file_path="/project/src/auth.py", new_code="auth code")
        kb.store(file_path="/project/src/db.py", new_code="db code")
        kb.store(file_path="/project/tests/test_auth.py", new_code="test code")

        results = kb.search_by_path("auth")
        assert len(results) == 2

        results = kb.search_by_path("src/auth")
        assert len(results) == 1

    def test_search_fts(self, kb):
        """Test full-text search."""
        kb.store(file_path="/test.py", new_code="def authenticate_user(): pass")
        kb.store(file_path="/other.py", new_code="def process_data(): pass")

        results = kb.search_fts("authenticate")
        assert len(results) >= 1
        assert "authenticate" in results[0]["new_code"]

    def test_get_all_with_embeddings(self, kb):
        """Test getting entries with embeddings."""
        # Store some with embeddings, some without
        kb.store(file_path="/a.py", new_code="a", embedding=[0.1] * 10)
        kb.store(file_path="/b.py", new_code="b")  # No embedding
        kb.store(file_path="/c.py", new_code="c", embedding=[0.2] * 10)

        results = kb.get_all_with_embeddings()
        assert len(results) == 2

    def test_get_recent(self, kb):
        """Test getting recent entries."""
        for i in range(5):
            kb.store(file_path=f"/file{i}.py", new_code=f"code {i}")

        results = kb.get_recent(limit=3)
        assert len(results) == 3

    def test_get_by_session(self, kb):
        """Test getting entries by session ID."""
        kb.store(file_path="/a.py", new_code="a", session_id="session1")
        kb.store(file_path="/b.py", new_code="b", session_id="session1")
        kb.store(file_path="/c.py", new_code="c", session_id="session2")

        results = kb.get_by_session("session1")
        assert len(results) == 2

    def test_update_embedding(self, kb):
        """Test updating an entry's embedding."""
        entry_id = kb.store(file_path="/test.py", new_code="code")

        # Initially no embedding
        entry = kb.get(entry_id)
        assert entry["embedding"] is None

        # Update embedding
        new_embedding = [0.5] * 100
        success = kb.update_embedding(entry_id, new_embedding)
        assert success

        # Verify update
        entry = kb.get(entry_id)
        assert entry["embedding"] is not None
        assert len(entry["embedding"]) == 100

    def test_delete(self, kb):
        """Test deleting an entry."""
        entry_id = kb.store(file_path="/test.py", new_code="code")

        assert kb.get(entry_id) is not None

        deleted = kb.delete(entry_id)
        assert deleted

        assert kb.get(entry_id) is None

    def test_clear(self, kb):
        """Test clearing the database."""
        for i in range(3):
            kb.store(file_path=f"/file{i}.py", new_code=f"code {i}")

        stats = kb.stats()
        assert stats["total_entries"] == 3

        cleared = kb.clear()
        assert cleared == 3

        stats = kb.stats()
        assert stats["total_entries"] == 0

    def test_stats(self, kb):
        """Test statistics gathering."""
        kb.store(
            file_path="/a.py", new_code="a",
            session_id="s1", embedding=[0.1] * 10
        )
        kb.store(file_path="/b.py", new_code="b", session_id="s1")
        kb.store(file_path="/a.py", new_code="a2", session_id="s2")

        stats = kb.stats()
        assert stats["total_entries"] == 3
        assert stats["entries_with_embeddings"] == 1
        assert stats["unique_files"] == 2
        assert stats["unique_sessions"] == 2

    def test_export(self, kb):
        """Test exporting entries."""
        kb.store(file_path="/a.py", new_code="a", embedding=[0.1] * 10)
        kb.store(file_path="/b.py", new_code="b")

        exported = kb.export()
        assert len(exported) == 2

        # Export should not include embeddings
        for entry in exported:
            assert entry["embedding"] is None
