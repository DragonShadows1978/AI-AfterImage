"""
Base storage backend abstraction for AI-AfterImage.

Defines the interface that all storage backends must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime


@dataclass
class StorageEntry:
    """
    Represents a single code memory entry.

    This is the canonical format for entries across all backends.
    """
    id: str
    file_path: str
    new_code: str
    old_code: Optional[str] = None
    context: Optional[str] = None
    timestamp: str = ""
    session_id: Optional[str] = None
    embedding: Optional[List[float]] = None

    # Search scoring (populated by search operations)
    fts_rank: Optional[float] = None
    semantic_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "new_code": self.new_code,
            "old_code": self.old_code,
            "context": self.context,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageEntry":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            file_path=data["file_path"],
            new_code=data["new_code"],
            old_code=data.get("old_code"),
            context=data.get("context"),
            timestamp=data.get("timestamp", ""),
            session_id=data.get("session_id"),
            embedding=data.get("embedding"),
        )


@dataclass
class StorageStats:
    """Statistics about the storage backend."""
    total_entries: int = 0
    entries_with_embeddings: int = 0
    unique_files: int = 0
    unique_sessions: int = 0
    oldest_entry: Optional[str] = None
    newest_entry: Optional[str] = None
    db_size_bytes: int = 0
    backend_type: str = "unknown"


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    All storage implementations (SQLite, PostgreSQL, etc.) must implement
    this interface to ensure consistent behavior.
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the storage backend.

        Creates necessary tables, indexes, and extensions.
        Should be idempotent - safe to call multiple times.
        """
        pass

    @abstractmethod
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
        Store a code snippet.

        Args:
            file_path: Path to the file that was written
            new_code: The code that was written
            old_code: Previous content (for edits)
            context: Surrounding conversation context
            session_id: Claude Code session identifier
            embedding: Vector embedding (384-dim for all-MiniLM-L6-v2)
            timestamp: ISO timestamp (auto-generated if not provided)

        Returns:
            ID of the stored entry (UUID string)
        """
        pass

    @abstractmethod
    def get(self, entry_id: str) -> Optional[StorageEntry]:
        """
        Get a single entry by ID.

        Args:
            entry_id: UUID of the entry

        Returns:
            StorageEntry if found, None otherwise
        """
        pass

    @abstractmethod
    def search_fts(self, query: str, limit: int = 10) -> List[StorageEntry]:
        """
        Full-text search.

        Args:
            query: Search query (backend-specific syntax may apply)
            limit: Maximum results

        Returns:
            List of matching entries with fts_rank populated
        """
        pass

    @abstractmethod
    def search_semantic(
        self,
        embedding: List[float],
        limit: int = 10
    ) -> List[Tuple[StorageEntry, float]]:
        """
        Semantic search using vector similarity.

        Args:
            embedding: Query embedding vector (384-dimensional)
            limit: Maximum results

        Returns:
            List of (entry, similarity_score) tuples, sorted by score descending
        """
        pass

    @abstractmethod
    def search_by_path(self, path_pattern: str, limit: int = 10) -> List[StorageEntry]:
        """
        Search entries by file path pattern.

        Args:
            path_pattern: Pattern to match against file paths (LIKE syntax)
            limit: Maximum results

        Returns:
            List of matching entries, sorted by timestamp descending
        """
        pass

    @abstractmethod
    def get_all_with_embeddings(self) -> List[StorageEntry]:
        """
        Get all entries that have embeddings.

        Used for in-memory semantic search in SQLite backend.
        PostgreSQL backend may not need this.

        Returns:
            List of all entries with embeddings
        """
        pass

    @abstractmethod
    def get_recent(self, limit: int = 20) -> List[StorageEntry]:
        """
        Get most recent entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of entries, sorted by timestamp descending
        """
        pass

    @abstractmethod
    def get_by_session(self, session_id: str) -> List[StorageEntry]:
        """
        Get all entries from a specific session.

        Args:
            session_id: Session identifier

        Returns:
            List of entries from that session, sorted by timestamp ascending
        """
        pass

    @abstractmethod
    def update_embedding(self, entry_id: str, embedding: List[float]) -> bool:
        """
        Update the embedding for an existing entry.

        Args:
            entry_id: UUID of the entry
            embedding: New embedding vector

        Returns:
            True if updated, False if entry not found
        """
        pass

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """
        Delete an entry by ID.

        Args:
            entry_id: UUID of the entry

        Returns:
            True if deleted, False if entry not found
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """
        Delete all entries.

        Returns:
            Count of deleted entries
        """
        pass

    @abstractmethod
    def stats(self) -> StorageStats:
        """
        Get statistics about the storage.

        Returns:
            StorageStats object with storage metrics
        """
        pass

    @abstractmethod
    def export(self) -> List[Dict[str, Any]]:
        """
        Export all entries as dictionaries.

        Note: Embeddings are typically excluded for size reasons.

        Returns:
            List of entry dictionaries
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close any open connections.

        Should be called when done using the backend.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes connections."""
        self.close()
        return False
