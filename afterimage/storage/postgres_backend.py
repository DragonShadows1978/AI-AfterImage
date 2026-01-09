"""
PostgreSQL storage backend for AI-AfterImage with pgvector support.

Provides concurrent read/write access and native vector similarity search
using the pgvector extension with HNSW indexing.
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

from .base import StorageBackend, StorageEntry, StorageStats

# Lazy imports for optional dependencies
_asyncpg = None
_pgvector = None


def _import_asyncpg():
    """Lazy import asyncpg."""
    global _asyncpg
    if _asyncpg is None:
        try:
            import asyncpg
            _asyncpg = asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for PostgreSQL backend. "
                "Install with: pip install asyncpg"
            )
    return _asyncpg


def _import_pgvector():
    """Lazy import pgvector for asyncpg."""
    global _pgvector
    if _pgvector is None:
        try:
            from pgvector.asyncpg import register_vector
            _pgvector = register_vector
        except ImportError:
            raise ImportError(
                "pgvector is required for PostgreSQL backend. "
                "Install with: pip install pgvector"
            )
    return _pgvector


class PostgreSQLBackend(StorageBackend):
    """
    PostgreSQL storage backend with pgvector for vector similarity.

    Features:
    - Concurrent read/write access (no single-writer bottleneck)
    - Native HNSW index for fast approximate nearest neighbor search
    - tsvector with GIN index for full-text search
    - Connection pooling for efficient resource usage
    - Async operations for high throughput

    Requirements:
    - PostgreSQL 14+ with pgvector extension
    - asyncpg and pgvector Python packages
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "afterimage",
        user: str = "afterimage",
        password: Optional[str] = None,
        connection_string: Optional[str] = None,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
        embedding_dim: int = 384
    ):
        """
        Initialize PostgreSQL backend.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password (also checks AFTERIMAGE_PG_PASSWORD env)
            connection_string: Full connection string (overrides other params)
            min_pool_size: Minimum connection pool size
            max_pool_size: Maximum connection pool size
            embedding_dim: Embedding vector dimension (384 for all-MiniLM-L6-v2)
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password or os.environ.get("AFTERIMAGE_PG_PASSWORD")
        self.connection_string = connection_string or os.environ.get("AFTERIMAGE_DATABASE_URL")
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.embedding_dim = embedding_dim

        self._pool = None
        self._initialized = False
        self._loop = None

    def _get_dsn(self) -> str:
        """Get connection DSN."""
        if self.connection_string:
            return self.connection_string
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            asyncpg = _import_asyncpg()
            register_vector = _import_pgvector()

            async def init_connection(conn):
                """Initialize each connection with pgvector."""
                await register_vector(conn)

            self._pool = await asyncpg.create_pool(
                self._get_dsn(),
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                init=init_connection
            )
        return self._pool

    def _get_loop(self):
        """Get or create the event loop for this backend."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def _run_async(self, coro):
        """Run async coroutine in sync context using a persistent loop."""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def initialize(self) -> None:
        """Initialize database schema with pgvector extension."""
        self._run_async(self._async_initialize())

    async def _async_initialize(self) -> None:
        """Async initialization of schema."""
        if self._initialized:
            return

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create main table with vector column
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS code_memory (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    old_code TEXT,
                    new_code TEXT NOT NULL,
                    context TEXT,
                    timestamp TIMESTAMPTZ NOT NULL,
                    session_id TEXT,
                    embedding vector({self.embedding_dim}),
                    search_vector tsvector,
                    UNIQUE(file_path, timestamp)
                )
            """)

            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cm_file_path
                ON code_memory(file_path)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cm_timestamp
                ON code_memory(timestamp)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cm_session
                ON code_memory(session_id)
            """)

            # HNSW index for vector similarity (cosine distance)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cm_embedding
                ON code_memory USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)

            # GIN index for full-text search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cm_search_vector
                ON code_memory USING gin(search_vector)
            """)

            # Trigger to auto-update search_vector
            await conn.execute("""
                CREATE OR REPLACE FUNCTION update_search_vector()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.search_vector :=
                        setweight(to_tsvector('english', COALESCE(NEW.file_path, '')), 'A') ||
                        setweight(to_tsvector('english', COALESCE(NEW.new_code, '')), 'B') ||
                        setweight(to_tsvector('english', COALESCE(NEW.context, '')), 'C');
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
            """)

            # Drop and recreate trigger to ensure it exists
            await conn.execute("""
                DROP TRIGGER IF EXISTS trig_update_search_vector ON code_memory
            """)
            await conn.execute("""
                CREATE TRIGGER trig_update_search_vector
                BEFORE INSERT OR UPDATE ON code_memory
                FOR EACH ROW EXECUTE FUNCTION update_search_vector()
            """)

        self._initialized = True

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
        """Store a code snippet in PostgreSQL."""
        return self._run_async(self._async_store(
            file_path, new_code, old_code, context, session_id, embedding, timestamp
        ))

    async def _async_store(
        self,
        file_path: str,
        new_code: str,
        old_code: Optional[str] = None,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        timestamp: Optional[str] = None
    ) -> str:
        """Async store operation."""
        if not self._initialized:
            await self._async_initialize()

        entry_id = str(uuid.uuid4())
        if timestamp is None:
            ts = datetime.now(timezone.utc)
        else:
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        pool = await self._get_pool()

        # Convert embedding to numpy array for pgvector
        embedding_val = None
        if embedding:
            import numpy as np
            embedding_val = np.array(embedding, dtype=np.float32)

        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO code_memory
                (id, file_path, old_code, new_code, context, timestamp, session_id, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, entry_id, file_path, old_code, new_code, context, ts, session_id, embedding_val)

        return entry_id

    def get(self, entry_id: str) -> Optional[StorageEntry]:
        """Get a single entry by ID."""
        return self._run_async(self._async_get(entry_id))

    async def _async_get(self, entry_id: str) -> Optional[StorageEntry]:
        """Async get operation."""
        if not self._initialized:
            await self._async_initialize()

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM code_memory WHERE id = $1", entry_id
            )

        if row is None:
            return None

        return self._row_to_entry(row)

    def search_fts(self, query: str, limit: int = 10) -> List[StorageEntry]:
        """Full-text search using PostgreSQL tsvector."""
        return self._run_async(self._async_search_fts(query, limit))

    async def _async_search_fts(self, query: str, limit: int = 10) -> List[StorageEntry]:
        """Async FTS operation."""
        if not self._initialized:
            await self._async_initialize()

        pool = await self._get_pool()

        # Convert query to tsquery format
        safe_query = self._prepare_tsquery(query)

        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT *, ts_rank(search_vector, plainto_tsquery('english', $1)) as rank
                FROM code_memory
                WHERE search_vector @@ plainto_tsquery('english', $1)
                ORDER BY rank DESC
                LIMIT $2
            """, safe_query, limit)

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
        """Semantic search using pgvector HNSW index."""
        return self._run_async(self._async_search_semantic(embedding, limit))

    async def _async_search_semantic(
        self,
        embedding: List[float],
        limit: int = 10
    ) -> List[Tuple[StorageEntry, float]]:
        """Async semantic search using pgvector."""
        if not self._initialized:
            await self._async_initialize()

        pool = await self._get_pool()

        import numpy as np
        embedding_arr = np.array(embedding, dtype=np.float32)

        async with pool.acquire() as conn:
            # Use cosine distance (<=>), convert to similarity (1 - distance)
            rows = await conn.fetch("""
                SELECT *,
                       1 - (embedding <=> $1::vector) as similarity
                FROM code_memory
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """, embedding_arr, limit)

        results = []
        for row in rows:
            entry = self._row_to_entry(row)
            similarity = row["similarity"]
            entry.semantic_score = similarity
            results.append((entry, similarity))

        return results

    def search_by_path(self, path_pattern: str, limit: int = 10) -> List[StorageEntry]:
        """Search entries by file path pattern."""
        return self._run_async(self._async_search_by_path(path_pattern, limit))

    async def _async_search_by_path(self, path_pattern: str, limit: int = 10) -> List[StorageEntry]:
        """Async path search."""
        if not self._initialized:
            await self._async_initialize()

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM code_memory
                WHERE file_path LIKE $1
                ORDER BY timestamp DESC
                LIMIT $2
            """, f"%{path_pattern}%", limit)

        return [self._row_to_entry(row) for row in rows]

    def get_all_with_embeddings(self) -> List[StorageEntry]:
        """Get all entries with embeddings."""
        return self._run_async(self._async_get_all_with_embeddings())

    async def _async_get_all_with_embeddings(self) -> List[StorageEntry]:
        """Async get all with embeddings."""
        if not self._initialized:
            await self._async_initialize()

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM code_memory
                WHERE embedding IS NOT NULL
                ORDER BY timestamp DESC
            """)

        return [self._row_to_entry(row) for row in rows]

    def get_recent(self, limit: int = 20) -> List[StorageEntry]:
        """Get most recent entries."""
        return self._run_async(self._async_get_recent(limit))

    async def _async_get_recent(self, limit: int = 20) -> List[StorageEntry]:
        """Async get recent."""
        if not self._initialized:
            await self._async_initialize()

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM code_memory
                ORDER BY timestamp DESC
                LIMIT $1
            """, limit)

        return [self._row_to_entry(row) for row in rows]

    def get_by_session(self, session_id: str) -> List[StorageEntry]:
        """Get all entries from a specific session."""
        return self._run_async(self._async_get_by_session(session_id))

    async def _async_get_by_session(self, session_id: str) -> List[StorageEntry]:
        """Async get by session."""
        if not self._initialized:
            await self._async_initialize()

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM code_memory
                WHERE session_id = $1
                ORDER BY timestamp ASC
            """, session_id)

        return [self._row_to_entry(row) for row in rows]

    def update_embedding(self, entry_id: str, embedding: List[float]) -> bool:
        """Update the embedding for an existing entry."""
        return self._run_async(self._async_update_embedding(entry_id, embedding))

    async def _async_update_embedding(self, entry_id: str, embedding: List[float]) -> bool:
        """Async update embedding."""
        if not self._initialized:
            await self._async_initialize()

        pool = await self._get_pool()

        import numpy as np
        embedding_arr = np.array(embedding, dtype=np.float32)

        async with pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE code_memory SET embedding = $1 WHERE id = $2
            """, embedding_arr, entry_id)

        return "UPDATE 1" in result

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        return self._run_async(self._async_delete(entry_id))

    async def _async_delete(self, entry_id: str) -> bool:
        """Async delete."""
        if not self._initialized:
            await self._async_initialize()

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM code_memory WHERE id = $1", entry_id
            )

        return "DELETE 1" in result

    def clear(self) -> int:
        """Delete all entries."""
        return self._run_async(self._async_clear())

    async def _async_clear(self) -> int:
        """Async clear."""
        if not self._initialized:
            await self._async_initialize()

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM code_memory")
            await conn.execute("DELETE FROM code_memory")

        return count

    def stats(self) -> StorageStats:
        """Get statistics about the PostgreSQL database."""
        return self._run_async(self._async_stats())

    async def _async_stats(self) -> StorageStats:
        """Async stats."""
        if not self._initialized:
            await self._async_initialize()

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM code_memory")
            with_embeddings = await conn.fetchval(
                "SELECT COUNT(*) FROM code_memory WHERE embedding IS NOT NULL"
            )
            unique_files = await conn.fetchval(
                "SELECT COUNT(DISTINCT file_path) FROM code_memory"
            )
            unique_sessions = await conn.fetchval(
                "SELECT COUNT(DISTINCT session_id) FROM code_memory WHERE session_id IS NOT NULL"
            )
            time_range = await conn.fetchrow(
                "SELECT MIN(timestamp), MAX(timestamp) FROM code_memory"
            )

            # Get database size
            db_size = await conn.fetchval("""
                SELECT pg_database_size($1)
            """, self.database)

        oldest = time_range[0].isoformat() if time_range[0] else None
        newest = time_range[1].isoformat() if time_range[1] else None

        return StorageStats(
            total_entries=total,
            entries_with_embeddings=with_embeddings,
            unique_files=unique_files,
            unique_sessions=unique_sessions,
            oldest_entry=oldest,
            newest_entry=newest,
            db_size_bytes=db_size or 0,
            backend_type="postgresql"
        )

    def export(self) -> List[Dict[str, Any]]:
        """Export all entries as dictionaries."""
        return self._run_async(self._async_export())

    async def _async_export(self) -> List[Dict[str, Any]]:
        """Async export."""
        if not self._initialized:
            await self._async_initialize()

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM code_memory ORDER BY timestamp ASC"
            )

        results = []
        for row in rows:
            entry = self._row_to_entry(row, include_embedding=False)
            results.append(entry.to_dict())
        return results

    def close(self) -> None:
        """Close connection pool."""
        if self._pool is not None:
            self._run_async(self._async_close())

    async def _async_close(self) -> None:
        """Async close."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _row_to_entry(self, row, include_embedding: bool = True) -> StorageEntry:
        """Convert a database row to a StorageEntry."""
        embedding = None
        if include_embedding and row["embedding"] is not None:
            # pgvector returns numpy array
            embedding = row["embedding"].tolist()

        timestamp = row["timestamp"]
        if hasattr(timestamp, 'isoformat'):
            timestamp = timestamp.isoformat()

        return StorageEntry(
            id=row["id"],
            file_path=row["file_path"],
            new_code=row["new_code"],
            old_code=row["old_code"],
            context=row["context"],
            timestamp=timestamp,
            session_id=row["session_id"],
            embedding=embedding,
        )

    def _prepare_tsquery(self, query: str) -> str:
        """Prepare query for PostgreSQL full-text search."""
        # Simple cleanup - let PostgreSQL handle the parsing
        return " ".join(query.split())


# Async-native interface for use in async contexts
class AsyncPostgreSQLBackend:
    """
    Async-native PostgreSQL backend for use in async applications.

    Provides the same functionality as PostgreSQLBackend but with
    native async methods instead of sync wrappers.
    """

    def __init__(self, sync_backend: PostgreSQLBackend):
        """Wrap a sync backend for async use."""
        self._backend = sync_backend

    async def initialize(self) -> None:
        await self._backend._async_initialize()

    async def store(self, **kwargs) -> str:
        return await self._backend._async_store(**kwargs)

    async def get(self, entry_id: str) -> Optional[StorageEntry]:
        return await self._backend._async_get(entry_id)

    async def search_fts(self, query: str, limit: int = 10) -> List[StorageEntry]:
        return await self._backend._async_search_fts(query, limit)

    async def search_semantic(
        self,
        embedding: List[float],
        limit: int = 10
    ) -> List[Tuple[StorageEntry, float]]:
        return await self._backend._async_search_semantic(embedding, limit)

    async def search_by_path(self, path_pattern: str, limit: int = 10) -> List[StorageEntry]:
        return await self._backend._async_search_by_path(path_pattern, limit)

    async def get_all_with_embeddings(self) -> List[StorageEntry]:
        return await self._backend._async_get_all_with_embeddings()

    async def get_recent(self, limit: int = 20) -> List[StorageEntry]:
        return await self._backend._async_get_recent(limit)

    async def get_by_session(self, session_id: str) -> List[StorageEntry]:
        return await self._backend._async_get_by_session(session_id)

    async def update_embedding(self, entry_id: str, embedding: List[float]) -> bool:
        return await self._backend._async_update_embedding(entry_id, embedding)

    async def delete(self, entry_id: str) -> bool:
        return await self._backend._async_delete(entry_id)

    async def clear(self) -> int:
        return await self._backend._async_clear()

    async def stats(self) -> StorageStats:
        return await self._backend._async_stats()

    async def export(self) -> List[Dict[str, Any]]:
        return await self._backend._async_export()

    async def close(self) -> None:
        await self._backend._async_close()


# ============================================================================
# Synchronous Interface using psycopg3 for thread-safe concurrent access
# ============================================================================

import threading

# Global lock for schema initialization (prevents deadlocks during concurrent init)
_schema_init_lock = threading.Lock()
_schema_initialized = set()  # Track initialized DSNs


class SyncPostgreSQLBackend(StorageBackend):
    """
    Synchronous PostgreSQL backend using psycopg3 for thread-safe operations.

    This backend is preferred for multi-threaded environments (like RDE with
    parallel agents) because each thread can safely use its own connection
    without asyncpg event loop issues.

    Features:
    - True thread-safe concurrent access
    - Connection-per-operation (no shared pool state issues)
    - Compatible with ThreadPoolExecutor and multiprocessing
    - Uses pgvector for semantic search
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "afterimage",
        user: str = "afterimage",
        password: Optional[str] = None,
        connection_string: Optional[str] = None,
        embedding_dim: int = 384
    ):
        """Initialize sync PostgreSQL backend."""
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password or os.environ.get("AFTERIMAGE_PG_PASSWORD")
        self.connection_string = connection_string or os.environ.get("AFTERIMAGE_DATABASE_URL")
        self.embedding_dim = embedding_dim
        self._initialized = False
    
    def _get_dsn(self) -> str:
        """Get connection DSN."""
        if self.connection_string:
            return self.connection_string
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def _get_connection(self):
        """Get a new connection (thread-safe - each call creates new connection)."""
        try:
            import psycopg
            from psycopg.rows import dict_row
            return psycopg.connect(self._get_dsn(), row_factory=dict_row)
        except ImportError:
            raise ImportError(
                "psycopg is required for SyncPostgreSQLBackend. "
                "Install with: pip install 'psycopg[binary]'"
            )
    
    def initialize(self) -> None:
        """Initialize database schema with thread-safe locking."""
        if self._initialized:
            return

        dsn = self._get_dsn()

        # Use global lock to prevent concurrent schema modifications
        with _schema_init_lock:
            # Double-check after acquiring lock
            if dsn in _schema_initialized:
                self._initialized = True
                return

            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if schema already exists (fast path)
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'code_memory'
                        ) as table_exists
                    """)
                    row = cur.fetchone()
                    table_exists = row["table_exists"] if row else False

                    if not table_exists:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                        cur.execute(f"""
                            CREATE TABLE IF NOT EXISTS code_memory (
                                id TEXT PRIMARY KEY,
                                file_path TEXT NOT NULL,
                                old_code TEXT,
                                new_code TEXT NOT NULL,
                                context TEXT,
                                timestamp TIMESTAMPTZ NOT NULL,
                                session_id TEXT,
                                embedding vector({self.embedding_dim}),
                                search_vector tsvector,
                                UNIQUE(file_path, timestamp)
                            )
                        """)

                        # Create indexes
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_cm_file_path ON code_memory(file_path)")
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_cm_timestamp ON code_memory(timestamp)")
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_cm_session ON code_memory(session_id)")
                        cur.execute("""
                            CREATE INDEX IF NOT EXISTS idx_cm_embedding
                            ON code_memory USING hnsw (embedding vector_cosine_ops)
                            WITH (m = 16, ef_construction = 64)
                        """)
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_cm_search_vector ON code_memory USING gin(search_vector)")

                        # Trigger for search vector
                        cur.execute("""
                            CREATE OR REPLACE FUNCTION update_search_vector()
                            RETURNS TRIGGER AS $$
                            BEGIN
                                NEW.search_vector :=
                                    setweight(to_tsvector('english', COALESCE(NEW.file_path, '')), 'A') ||
                                    setweight(to_tsvector('english', COALESCE(NEW.new_code, '')), 'B') ||
                                    setweight(to_tsvector('english', COALESCE(NEW.context, '')), 'C');
                                RETURN NEW;
                            END;
                            $$ LANGUAGE plpgsql
                        """)
                        cur.execute("DROP TRIGGER IF EXISTS trig_update_search_vector ON code_memory")
                        cur.execute("""
                            CREATE TRIGGER trig_update_search_vector
                            BEFORE INSERT OR UPDATE ON code_memory
                            FOR EACH ROW EXECUTE FUNCTION update_search_vector()
                        """)
                conn.commit()

            _schema_initialized.add(dsn)
            self._initialized = True
    
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
        """Store a code snippet (thread-safe)."""
        if not self._initialized:
            self.initialize()
        
        entry_id = str(uuid.uuid4())
        if timestamp is None:
            ts = datetime.now(timezone.utc)
        else:
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Format embedding for pgvector
        embedding_str = None
        if embedding:
            embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO code_memory
                    (id, file_path, old_code, new_code, context, timestamp, session_id, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                """, (entry_id, file_path, old_code, new_code, context, ts, session_id, embedding_str))
            conn.commit()
        
        return entry_id
    
    def get(self, entry_id: str) -> Optional[StorageEntry]:
        """Get a single entry by ID."""
        if not self._initialized:
            self.initialize()
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM code_memory WHERE id = %s", (entry_id,))
                row = cur.fetchone()
        
        if row is None:
            return None
        return self._row_to_entry(row)
    
    def search_fts(self, query: str, limit: int = 10) -> List[StorageEntry]:
        """Full-text search using PostgreSQL tsvector."""
        if not self._initialized:
            self.initialize()
        
        safe_query = " ".join(query.split())
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT *, ts_rank(search_vector, plainto_tsquery('english', %s)) as rank
                    FROM code_memory
                    WHERE search_vector @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                """, (safe_query, safe_query, limit))
                rows = cur.fetchall()
        
        results = []
        for row in rows:
            entry = self._row_to_entry(row)
            entry.fts_rank = row.get("rank", 0)
            results.append(entry)
        return results
    
    def search_semantic(
        self,
        embedding: List[float],
        limit: int = 10
    ) -> List[Tuple[StorageEntry, float]]:
        """Semantic search using pgvector."""
        if not self._initialized:
            self.initialize()
        
        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT *,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM code_memory
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (embedding_str, embedding_str, limit))
                rows = cur.fetchall()
        
        results = []
        for row in rows:
            entry = self._row_to_entry(row)
            similarity = row.get("similarity", 0)
            entry.semantic_score = similarity
            results.append((entry, similarity))
        return results
    
    def search_by_path(self, path_pattern: str, limit: int = 10) -> List[StorageEntry]:
        """Search by file path pattern."""
        if not self._initialized:
            self.initialize()
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM code_memory
                    WHERE file_path LIKE %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (f"%{path_pattern}%", limit))
                rows = cur.fetchall()
        
        return [self._row_to_entry(row) for row in rows]
    
    def get_all_with_embeddings(self) -> List[StorageEntry]:
        """Get all entries with embeddings."""
        if not self._initialized:
            self.initialize()
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM code_memory
                    WHERE embedding IS NOT NULL
                    ORDER BY timestamp DESC
                """)
                rows = cur.fetchall()
        
        return [self._row_to_entry(row) for row in rows]
    
    def get_recent(self, limit: int = 20) -> List[StorageEntry]:
        """Get most recent entries."""
        if not self._initialized:
            self.initialize()
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM code_memory ORDER BY timestamp DESC LIMIT %s", (limit,))
                rows = cur.fetchall()
        
        return [self._row_to_entry(row) for row in rows]
    
    def get_by_session(self, session_id: str) -> List[StorageEntry]:
        """Get entries by session."""
        if not self._initialized:
            self.initialize()
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM code_memory
                    WHERE session_id = %s
                    ORDER BY timestamp ASC
                """, (session_id,))
                rows = cur.fetchall()
        
        return [self._row_to_entry(row) for row in rows]
    
    def update_embedding(self, entry_id: str, embedding: List[float]) -> bool:
        """Update embedding for an entry."""
        if not self._initialized:
            self.initialize()
        
        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE code_memory SET embedding = %s::vector WHERE id = %s
                """, (embedding_str, entry_id))
                updated = cur.rowcount > 0
            conn.commit()
        
        return updated
    
    def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        if not self._initialized:
            self.initialize()
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM code_memory WHERE id = %s", (entry_id,))
                deleted = cur.rowcount > 0
            conn.commit()
        
        return deleted
    
    def clear(self) -> int:
        """Delete all entries."""
        if not self._initialized:
            self.initialize()
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as cnt FROM code_memory")
                count = cur.fetchone()["cnt"]
                cur.execute("DELETE FROM code_memory")
            conn.commit()
        
        return count
    
    def stats(self) -> StorageStats:
        """Get database statistics."""
        if not self._initialized:
            self.initialize()
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as cnt FROM code_memory")
                total = cur.fetchone()["cnt"]
                
                cur.execute("SELECT COUNT(*) as cnt FROM code_memory WHERE embedding IS NOT NULL")
                with_embeddings = cur.fetchone()["cnt"]
                
                cur.execute("SELECT COUNT(DISTINCT file_path) as cnt FROM code_memory")
                unique_files = cur.fetchone()["cnt"]
                
                cur.execute("SELECT COUNT(DISTINCT session_id) as cnt FROM code_memory WHERE session_id IS NOT NULL")
                unique_sessions = cur.fetchone()["cnt"]
                
                cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM code_memory")
                time_range = cur.fetchone()
                
                cur.execute("SELECT pg_database_size(%s)", (self.database,))
                db_size = cur.fetchone()["pg_database_size"]
        
        oldest = time_range["min"].isoformat() if time_range["min"] else None
        newest = time_range["max"].isoformat() if time_range["max"] else None
        
        return StorageStats(
            total_entries=total,
            entries_with_embeddings=with_embeddings,
            unique_files=unique_files,
            unique_sessions=unique_sessions,
            oldest_entry=oldest,
            newest_entry=newest,
            db_size_bytes=db_size or 0,
            backend_type="postgresql-sync"
        )
    
    def export(self) -> List[Dict[str, Any]]:
        """Export all entries."""
        if not self._initialized:
            self.initialize()
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM code_memory ORDER BY timestamp ASC")
                rows = cur.fetchall()
        
        return [self._row_to_entry(row, include_embedding=False).to_dict() for row in rows]
    
    def close(self) -> None:
        """No-op for sync backend (connections are per-operation)."""
        pass
    
    def _row_to_entry(self, row: Dict, include_embedding: bool = True) -> StorageEntry:
        """Convert row to StorageEntry."""
        embedding = None
        if include_embedding and row.get("embedding") is not None:
            # pgvector returns string representation, parse it
            emb_str = str(row["embedding"])
            if emb_str.startswith('[') and emb_str.endswith(']'):
                embedding = [float(x) for x in emb_str[1:-1].split(',')]
        
        timestamp = row["timestamp"]
        if hasattr(timestamp, 'isoformat'):
            timestamp = timestamp.isoformat()
        
        return StorageEntry(
            id=row["id"],
            file_path=row["file_path"],
            new_code=row["new_code"],
            old_code=row.get("old_code"),
            context=row.get("context"),
            timestamp=timestamp,
            session_id=row.get("session_id"),
            embedding=embedding,
        )
