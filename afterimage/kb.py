"""
Knowledge Base: High-level API for AI-AfterImage storage.

Updated to use the storage backend abstraction, supporting both SQLite
and PostgreSQL backends transparently.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

from .storage import StorageBackend, StorageEntry, StorageStats
from .config import load_config, get_storage_backend


class KnowledgeBase:
    """
    High-level Knowledge Base interface for AI-AfterImage.

    Provides a consistent API regardless of backend (SQLite or PostgreSQL).
    Handles embedding generation, storage, and retrieval.

    This class maintains backward compatibility with the original SQLite-only
    implementation while adding support for PostgreSQL.
    """

    def __init__(
        self,
        backend: Optional[StorageBackend] = None,
        db_path: Optional[Path] = None
    ):
        """
        Initialize the Knowledge Base.

        Args:
            backend: Storage backend to use. If None, loads from config.
            db_path: Legacy parameter for SQLite path. Ignored if backend provided.
        """
        if backend is not None:
            self._backend = backend
        elif db_path is not None:
            # Legacy mode: explicit SQLite path
            from .storage import SQLiteBackend
            self._backend = SQLiteBackend(db_path=db_path)
            self._backend.initialize()
        else:
            # Load from config
            self._backend = None  # Lazy load

    @property
    def backend(self) -> StorageBackend:
        """Get storage backend, loading from config if needed."""
        if self._backend is None:
            self._backend = get_storage_backend()
        return self._backend

    @property
    def db_path(self) -> Optional[Path]:
        """Legacy property for SQLite path. Returns None for PostgreSQL."""
        from .storage import SQLiteBackend
        if isinstance(self.backend, SQLiteBackend):
            return self.backend.db_path
        return None

    def store(
        self,
        file_path: str,
        new_code: str,
        old_code: Optional[str] = None,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        timestamp: Optional[str] = None
    ) -> str:
        """
        Store a code snippet in the knowledge base.

        Args:
            file_path: Path to the file that was written
            new_code: The code that was written
            old_code: Previous content (for edits)
            context: Surrounding conversation context
            session_id: Claude Code session identifier
            embedding: Vector embedding for semantic search
            timestamp: ISO timestamp (auto-generated if not provided)

        Returns:
            ID of the stored entry
        """
        return self.backend.store(
            file_path=file_path,
            new_code=new_code,
            old_code=old_code,
            context=context,
            session_id=session_id,
            embedding=embedding,
            timestamp=timestamp
        )

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single entry by ID.

        Args:
            entry_id: UUID of the entry

        Returns:
            Entry dictionary if found, None otherwise
        """
        entry = self.backend.get(entry_id)
        if entry is None:
            return None
        return entry.to_dict()

    def search_by_path(self, path_pattern: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search entries by file path pattern (LIKE match).

        Args:
            path_pattern: Pattern to match against file paths
            limit: Maximum results

        Returns:
            List of matching entry dictionaries
        """
        entries = self.backend.search_by_path(path_pattern, limit)
        return [e.to_dict() for e in entries]

    def search_fts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Full-text search.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching entries with fts_rank
        """
        entries = self.backend.search_fts(query, limit)
        results = []
        for entry in entries:
            d = entry.to_dict()
            d["fts_rank"] = entry.fts_rank
            results.append(d)
        return results

    def get_all_with_embeddings(self) -> List[Dict[str, Any]]:
        """
        Get all entries that have embeddings (for vector search).

        Returns:
            List of entries with embeddings
        """
        entries = self.backend.get_all_with_embeddings()
        return [e.to_dict() for e in entries]

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get most recent entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of recent entries
        """
        entries = self.backend.get_recent(limit)
        return [e.to_dict() for e in entries]

    def get_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all entries from a specific session.

        Args:
            session_id: Session identifier

        Returns:
            List of entries from that session
        """
        entries = self.backend.get_by_session(session_id)
        return [e.to_dict() for e in entries]

    def update_embedding(self, entry_id: str, embedding: List[float]) -> bool:
        """
        Update the embedding for an existing entry.

        Args:
            entry_id: UUID of the entry
            embedding: New embedding vector

        Returns:
            True if updated, False if entry not found
        """
        return self.backend.update_embedding(entry_id, embedding)

    def delete(self, entry_id: str) -> bool:
        """
        Delete an entry by ID.

        Args:
            entry_id: UUID of the entry

        Returns:
            True if deleted, False if entry not found
        """
        return self.backend.delete(entry_id)

    def clear(self) -> int:
        """
        Delete all entries.

        Returns:
            Count of deleted entries
        """
        return self.backend.clear()

    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary with statistics
        """
        s = self.backend.stats()
        return {
            "total_entries": s.total_entries,
            "entries_with_embeddings": s.entries_with_embeddings,
            "unique_files": s.unique_files,
            "unique_sessions": s.unique_sessions,
            "oldest_entry": s.oldest_entry,
            "newest_entry": s.newest_entry,
            "db_size_bytes": s.db_size_bytes,
            "backend_type": s.backend_type,
        }

    def export(self) -> List[Dict[str, Any]]:
        """
        Export all entries as dictionaries (excludes embeddings).

        Returns:
            List of entry dictionaries
        """
        return self.backend.export()

    def close(self):
        """Close backend connections."""
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Legacy helper functions for backward compatibility
def get_db_path() -> Path:
    """Get default database path."""
    config = load_config()
    return config.sqlite.path


def serialize_embedding(embedding: List[float]) -> bytes:
    """Serialize embedding to bytes (legacy SQLite format)."""
    import struct
    return struct.pack(f"{len(embedding)}f", *embedding)


def deserialize_embedding(data: bytes) -> List[float]:
    """Deserialize embedding from bytes (legacy SQLite format)."""
    import struct
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))
