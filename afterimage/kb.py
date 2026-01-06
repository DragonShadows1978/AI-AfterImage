"""
Knowledge Base: SQLite storage with FTS5 for full-text search.

Stores code snippets with metadata for later retrieval.
"""

import sqlite3
import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import struct


def get_db_path() -> Path:
    """Get path to database file (~/.afterimage/memory.db)."""
    afterimage_dir = Path.home() / ".afterimage"
    afterimage_dir.mkdir(exist_ok=True)
    return afterimage_dir / "memory.db"


def serialize_embedding(embedding: List[float]) -> bytes:
    """Serialize embedding to bytes for SQLite storage."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def deserialize_embedding(data: bytes) -> List[float]:
    """Deserialize embedding from bytes."""
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


class KnowledgeBase:
    """
    SQLite-backed knowledge base for code snippets.

    Schema:
    - id: unique identifier (UUID)
    - file_path: where the code was written
    - old_code: what was there before (NULL for new files)
    - new_code: what was written
    - context: surrounding conversation context
    - timestamp: when it was written
    - session_id: which Claude Code session
    - embedding: vector embedding (BLOB)

    Also maintains FTS5 index for full-text search.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_db_path()
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
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

        # Create indexes
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

        # FTS5 virtual table for full-text search
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

        # Triggers to keep FTS5 in sync
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

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get a single entry by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM code_memory WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return self._row_to_dict(row)

    def search_by_path(self, path_pattern: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entries by file path pattern (LIKE match)."""
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

        return [self._row_to_dict(row) for row in rows]

    def search_fts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Full-text search using FTS5.

        Args:
            query: Search query (FTS5 syntax supported)
            limit: Maximum results

        Returns:
            List of matching entries with BM25 rank
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Use BM25 ranking
        cursor.execute("""
            SELECT cm.*, bm25(code_memory_fts) as rank
            FROM code_memory cm
            JOIN code_memory_fts fts ON cm.id = fts.id
            WHERE code_memory_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            entry = self._row_to_dict(row)
            entry["fts_rank"] = row["rank"]
            results.append(entry)

        return results

    def get_all_with_embeddings(self) -> List[Dict[str, Any]]:
        """Get all entries that have embeddings (for vector search)."""
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

        return [self._row_to_dict(row) for row in rows]

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most recent entries."""
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

        return [self._row_to_dict(row) for row in rows]

    def get_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all entries from a specific session."""
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

        return [self._row_to_dict(row) for row in rows]

    def update_embedding(self, entry_id: str, embedding: List[float]) -> bool:
        """Update the embedding for an existing entry."""
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM code_memory WHERE id = ?", (entry_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return deleted

    def clear(self) -> int:
        """Delete all entries. Returns count of deleted rows."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM code_memory")
        count = cursor.fetchone()[0]

        cursor.execute("DELETE FROM code_memory")
        conn.commit()
        conn.close()

        return count

    def stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
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

        # Get file size
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

        conn.close()

        return {
            "total_entries": total,
            "entries_with_embeddings": with_embeddings,
            "unique_files": unique_files,
            "unique_sessions": unique_sessions,
            "oldest_entry": time_range[0],
            "newest_entry": time_range[1],
            "db_size_bytes": db_size,
        }

    def export(self) -> List[Dict[str, Any]]:
        """Export all entries as a list of dictionaries (excludes embeddings)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM code_memory ORDER BY timestamp ASC")
        rows = cursor.fetchall()
        conn.close()

        # Don't include embeddings in export
        return [self._row_to_dict(row, include_embedding=False) for row in rows]

    def _row_to_dict(self, row: sqlite3.Row, include_embedding: bool = True) -> Dict[str, Any]:
        """Convert a database row to a dictionary."""
        result = {
            "id": row["id"],
            "file_path": row["file_path"],
            "old_code": row["old_code"],
            "new_code": row["new_code"],
            "context": row["context"],
            "timestamp": row["timestamp"],
            "session_id": row["session_id"],
        }

        if include_embedding and row["embedding"]:
            result["embedding"] = deserialize_embedding(row["embedding"])
        else:
            result["embedding"] = None

        return result
