"""
SQLite storage backend for AI-AfterImage.

Refactored from the original kb.py implementation. Maintains FTS5 for
full-text search and stores embeddings as BLOBs for in-memory similarity.
"""

import sqlite3
import os
import uuid
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .base import StorageBackend, StorageEntry, StorageStats


def serialize_embedding(embedding: List[float]) -> bytes:
    """Serialize embedding to bytes for SQLite storage."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def deserialize_embedding(data: bytes) -> List[float]:
    """Deserialize embedding from bytes."""
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import math
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


class SQLiteBackend(StorageBackend):
    """
    SQLite-based storage backend.

    Features:
    - FTS5 full-text search with BM25 ranking
    - Embeddings stored as packed floats (BLOB)
    - In-memory cosine similarity for semantic search
    - Single-file storage (~/.afterimage/memory.db by default)

    Limitations:
    - Single-writer concurrency (SQLITE_BUSY under load)
    - In-memory semantic search (loads all embeddings)
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize SQLite backend.

        Args:
            db_path: Path to database file. Defaults to ~/.afterimage/memory.db
        """
        if db_path is None:
            afterimage_dir = Path.home() / ".afterimage"
            afterimage_dir.mkdir(exist_ok=True)
            db_path = afterimage_dir / "memory.db"
        self.db_path = Path(db_path)
        self._initialized = False

    def initialize(self) -> None:
        """Initialize database schema with FTS5 and triggers."""
        if self._initialized:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_memory (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                old_code TEXT,
                new_code TEXT NOT NULL,
                context TEXT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                embedding BLOB,
                UNIQUE(file_path, timestamp)
            )
        """)

        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_path
            ON code_memory(file_path)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON code_memory(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session
            ON code_memory(session_id)
        """)

        # FTS5 virtual table
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS code_memory_fts USING fts5(
                id,
                file_path,
                new_code,
                context,
                content='code_memory',
                content_rowid='rowid'
            )
        """)

        # Sync triggers
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS code_memory_ai
            AFTER INSERT ON code_memory BEGIN
                INSERT INTO code_memory_fts(rowid, id, file_path, new_code, context)
                VALUES (new.rowid, new.id, new.file_path, new.new_code, new.context);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS code_memory_ad
            AFTER DELETE ON code_memory BEGIN
                INSERT INTO code_memory_fts(code_memory_fts, rowid, id, file_path, new_code, context)
                VALUES('delete', old.rowid, old.id, old.file_path, old.new_code, old.context);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS code_memory_au
            AFTER UPDATE ON code_memory BEGIN
                INSERT INTO code_memory_fts(code_memory_fts, rowid, id, file_path, new_code, context)
                VALUES('delete', old.rowid, old.id, old.file_path, old.new_code, old.context);
                INSERT INTO code_memory_fts(rowid, id, file_path, new_code, context)
                VALUES (new.rowid, new.id, new.file_path, new.new_code, new.context);
            END
        """)

        conn.commit()
        conn.close()
        self._initialized = True

    def _ensure_init(self):
        """Ensure database is initialized before operations."""
        if not self._initialized:
            self.initialize()

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
        """Store a code snippet in SQLite."""
        self._ensure_init()

        entry_id = str(uuid.uuid4())
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        embedding_blob = None
        if embedding:
            embedding_blob = serialize_embedding(embedding)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO code_memory
                (id, file_path, old_code, new_code, context, timestamp, session_id, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (entry_id, file_path, old_code, new_code, context, timestamp, session_id, embedding_blob))
            conn.commit()
        finally:
            conn.close()

        return entry_id

    def get(self, entry_id: str) -> Optional[StorageEntry]:
        """Get a single entry by ID."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM code_memory WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return self._row_to_entry(row)

    def search_fts(self, query: str, limit: int = 10) -> List[StorageEntry]:
        """Full-text search using FTS5 with BM25 ranking."""
        self._ensure_init()

        # Escape special FTS5 characters
        safe_query = self._escape_fts_query(query)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT cm.*, bm25(code_memory_fts) as rank
                FROM code_memory cm
                JOIN code_memory_fts fts ON cm.id = fts.id
                WHERE code_memory_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (safe_query, limit))
            rows = cursor.fetchall()
        except Exception:
            # FTS query failed - try simpler query
            words = query.split()
            if words:
                safe_query = " OR ".join(f'"{w}"' for w in words[:5])
                try:
                    cursor.execute("""
                        SELECT cm.*, bm25(code_memory_fts) as rank
                        FROM code_memory cm
                        JOIN code_memory_fts fts ON cm.id = fts.id
                        WHERE code_memory_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    """, (safe_query, limit))
                    rows = cursor.fetchall()
                except Exception:
                    rows = []
            else:
                rows = []

        conn.close()

        results = []
        for row in rows:
            entry = self._row_to_entry(row)
            entry.fts_rank = row["rank"]
            results.append(entry)

        return results

    def search_semantic(
        self,
        embedding: List[float],
        limit: int = 10
    ) -> List[Tuple[StorageEntry, float]]:
        """
        Semantic search using in-memory cosine similarity.

        Note: Loads all embeddings into memory for comparison.
        For large databases, consider PostgreSQL with pgvector.
        """
        self._ensure_init()

        entries = self.get_all_with_embeddings()

        results = []
        for entry in entries:
            if entry.embedding:
                similarity = cosine_similarity(embedding, entry.embedding)
                entry.semantic_score = similarity
                results.append((entry, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def search_by_path(self, path_pattern: str, limit: int = 10) -> List[StorageEntry]:
        """Search entries by file path pattern."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM code_memory
            WHERE file_path LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{path_pattern}%", limit))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_entry(row) for row in rows]

    def get_all_with_embeddings(self) -> List[StorageEntry]:
        """Get all entries that have embeddings."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM code_memory
            WHERE embedding IS NOT NULL
            ORDER BY timestamp DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_entry(row) for row in rows]

    def get_recent(self, limit: int = 20) -> List[StorageEntry]:
        """Get most recent entries."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM code_memory
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_entry(row) for row in rows]

    def get_by_session(self, session_id: str) -> List[StorageEntry]:
        """Get all entries from a specific session."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM code_memory
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_entry(row) for row in rows]

    def update_embedding(self, entry_id: str, embedding: List[float]) -> bool:
        """Update the embedding for an existing entry."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        embedding_blob = serialize_embedding(embedding)
        cursor.execute("""
            UPDATE code_memory SET embedding = ? WHERE id = ?
        """, (embedding_blob, entry_id))

        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return updated

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM code_memory WHERE id = ?", (entry_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return deleted

    def clear(self) -> int:
        """Delete all entries."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM code_memory")
        count = cursor.fetchone()[0]

        cursor.execute("DELETE FROM code_memory")
        conn.commit()
        conn.close()

        return count

    def stats(self) -> StorageStats:
        """Get statistics about the SQLite database."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM code_memory")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM code_memory WHERE embedding IS NOT NULL")
        with_embeddings = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT file_path) FROM code_memory")
        unique_files = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM code_memory WHERE session_id IS NOT NULL")
        unique_sessions = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM code_memory")
        time_range = cursor.fetchone()

        conn.close()

        db_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0

        return StorageStats(
            total_entries=total,
            entries_with_embeddings=with_embeddings,
            unique_files=unique_files,
            unique_sessions=unique_sessions,
            oldest_entry=time_range[0],
            newest_entry=time_range[1],
            db_size_bytes=db_size,
            backend_type="sqlite"
        )

    def export(self) -> List[Dict[str, Any]]:
        """Export all entries as dictionaries (excludes embeddings)."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM code_memory ORDER BY timestamp ASC")
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            entry = self._row_to_entry(row, include_embedding=False)
            results.append(entry.to_dict())
        return results

    def close(self) -> None:
        """Close connections (SQLite connections are per-operation)."""
        pass

    def _row_to_entry(self, row: sqlite3.Row, include_embedding: bool = True) -> StorageEntry:
        """Convert a database row to a StorageEntry."""
        embedding = None
        if include_embedding and row["embedding"]:
            embedding = deserialize_embedding(row["embedding"])

        return StorageEntry(
            id=row["id"],
            file_path=row["file_path"],
            new_code=row["new_code"],
            old_code=row["old_code"],
            context=row["context"],
            timestamp=row["timestamp"],
            session_id=row["session_id"],
            embedding=embedding,
        )

    def _escape_fts_query(self, query: str) -> str:
        """Escape special characters for FTS5 query."""
        special_chars = ['"', "'", "(", ")", "*", ":", "-", "^"]
        escaped = query
        for char in special_chars:
            escaped = escaped.replace(char, " ")
        escaped = " ".join(escaped.split())
        if not escaped.strip():
            return "*"
        return escaped
