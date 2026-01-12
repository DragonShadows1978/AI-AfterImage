"""
Embedding Processor for AfterImage.

Core processing logic with GPU support, incremental reindexing,
priority queue, rate limiting, and model fallback.
"""

import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Event
from typing import Optional, List, Tuple, Dict, Any

from .config import DaemonConfig
from .metrics import MetricsCollector
from .retry import RetryManager
from .rate_limiter import RateLimiter

# Add AfterImage paths
AFTERIMAGE_SEARCH_PATHS = [
    str(Path.home() / "mini-mind-v2" / "workspace" / "AI-AfterImage"),
    str(Path.home() / "mini-mind-v2" / "workspace" / "afterimage"),
    str(Path.home() / "AI-AfterImage"),
]

for path in AFTERIMAGE_SEARCH_PATHS:
    if Path(path).exists() and path not in sys.path:
        sys.path.insert(0, path)


logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """
    Processes entries without embeddings using GPU-accelerated embedding generation.

    Features:
    - GPU detection and CUDA acceleration
    - Incremental processing (only entries without embeddings)
    - Priority queue (recent entries first)
    - Batch processing for efficiency
    - Rate limiting for GPU/DB operations
    - Model fallback if primary fails
    - Batch completion tracking for graceful shutdown
    """

    def __init__(self, config: DaemonConfig, metrics: MetricsCollector):
        self.config = config
        self.metrics = metrics
        self._backend = None
        self._model = None
        self._device = None
        self._initialized = False
        self._retry_manager = None
        self._rate_limiter = None
        self._active_model_name = None
        self._active_embedding_dim = None
        self._batch_in_progress = Event()
        self._model_fallback_count = 0

    def initialize(self) -> bool:
        """Initialize backend and embedding model. Returns True on success."""
        try:
            # Detect and configure device
            self._device = self.config.detect_device()
            logger.info(f"Using device: {self._device}")

            # Initialize backend
            self._backend = self._create_backend()
            if self._backend is None:
                logger.error("Failed to create storage backend")
                return False

            # Initialize embedding model with fallback support
            self._model = self._load_embedding_model()
            if self._model is None:
                logger.error("Failed to load embedding model")
                return False

            self._initialized = True

            # Initialize retry manager
            if self.config.retry_enabled:
                self._retry_manager = RetryManager(
                    state_file=self.config.retry_state_file,
                    max_attempts=self.config.retry_max_attempts,
                    base_delay=self.config.retry_base_delay,
                    max_delay=self.config.retry_max_delay,
                    jitter=self.config.retry_jitter
                )
                logger.info("Retry manager initialized")

            # Initialize rate limiter
            if self.config.rate_limit_enabled:
                self._rate_limiter = RateLimiter(
                    gpu_capacity=self.config.rate_limit_gpu_capacity,
                    gpu_refill_rate=self.config.rate_limit_gpu_refill,
                    db_capacity=self.config.rate_limit_db_capacity,
                    db_refill_rate=self.config.rate_limit_db_refill,
                    enabled=True
                )
                logger.info("Rate limiter initialized")

            # Warmup model
            if self.config.warmup_enabled:
                warmup_time = self.warmup()
                self.metrics.record_warmup(warmup_time)
                logger.info(f"Model warmup completed in {warmup_time:.1f}ms")

            self.metrics.daemon_started(self._device, self._active_model_name or self.config.model_name)
            logger.info(f"Processor initialized with {self._device} device")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _create_backend(self):
        """Create storage backend based on config."""
        try:
            from afterimage.storage import SyncPostgreSQLBackend, SQLiteBackend

            backend_type = self.config.backend

            # Auto-detect backend
            if backend_type == "auto":
                # Try PostgreSQL first if password is available
                if self.config.pg_password:
                    backend_type = "postgresql"
                else:
                    backend_type = "sqlite"

            if backend_type == "postgresql":
                try:
                    import psycopg
                    backend = SyncPostgreSQLBackend(
                        host=self.config.pg_host,
                        port=self.config.pg_port,
                        database=self.config.pg_database,
                        user=self.config.pg_user,
                        password=self.config.pg_password,
                        embedding_dim=self.config.embedding_dim
                    )
                    backend.initialize()
                    logger.info("Using PostgreSQL backend")
                    return backend
                except Exception as e:
                    logger.warning(f"PostgreSQL failed ({e}), falling back to SQLite")

            # SQLite fallback
            self.config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            backend = SQLiteBackend(db_path=self.config.sqlite_path)
            backend.initialize()
            logger.info("Using SQLite backend")
            return backend

        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            return None

    def _load_embedding_model(self):
        """Load sentence-transformers model with fallback support."""
        try:
            from sentence_transformers import SentenceTransformer

            # Configure CUDA memory if using GPU
            if self._device == "cuda":
                try:
                    import torch
                    if self.config.cuda_memory_fraction < 1.0:
                        torch.cuda.set_per_process_memory_fraction(
                            self.config.cuda_memory_fraction
                        )
                        logger.info(f"CUDA memory fraction: {self.config.cuda_memory_fraction}")
                except Exception as e:
                    logger.warning(f"Could not configure CUDA memory: {e}")

            # Build list of models to try
            models_to_try = [self.config.model_name] + self.config.fallback_models
            cache_dir = Path.home() / ".afterimage" / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)

            for model_name in models_to_try:
                try:
                    logger.info(f"Attempting to load model: {model_name}")
                    model = SentenceTransformer(
                        model_name,
                        cache_folder=str(cache_dir),
                        device=self._device
                    )
                    
                    self._active_model_name = model_name
                    self._active_embedding_dim = model.get_sentence_embedding_dimension()
                    
                    if model_name != self.config.model_name:
                        self._model_fallback_count += 1
                        logger.warning(f"Using fallback model: {model_name} (dim={self._active_embedding_dim})")
                    else:
                        logger.info(f"Loaded model: {model_name} (dim={self._active_embedding_dim})")
                    
                    return model
                    
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
                    continue

            logger.error("All models failed to load")
            return None

        except ImportError:
            logger.error("sentence-transformers not installed")
            return None

    def get_entries_without_embeddings(
        self,
        limit: int,
        priority_first: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get entries that need embeddings, with priority queue support.

        Args:
            limit: Maximum entries to return
            priority_first: If True, return recent entries first

        Returns:
            List of entry dicts with id, file_path, new_code, timestamp
        """
        if not self._backend:
            return []

        try:
            # Query entries without embeddings
            entries = self._query_entries_without_embeddings(limit, priority_first)
            return entries

        except Exception as e:
            logger.error(f"Error querying entries: {e}")
            return []

    def _query_entries_without_embeddings(
        self,
        limit: int,
        priority_first: bool
    ) -> List[Dict[str, Any]]:
        """Query backend for entries without embeddings."""
        from afterimage.storage import SyncPostgreSQLBackend, SQLiteBackend

        if isinstance(self._backend, SyncPostgreSQLBackend):
            return self._query_postgres(limit, priority_first)
        else:
            return self._query_sqlite(limit, priority_first)

    def _query_postgres(self, limit: int, priority_first: bool) -> List[Dict[str, Any]]:
        """Query PostgreSQL for entries without embeddings."""
        import psycopg

        priority_cutoff = datetime.now(timezone.utc) - timedelta(hours=self.config.priority_window_hours)

        if priority_first:
            query = """
                SELECT id, file_path, new_code, timestamp
                FROM code_memory
                WHERE embedding IS NULL
                ORDER BY
                    CASE WHEN timestamp > %s THEN 0 ELSE 1 END,
                    timestamp DESC
                LIMIT %s
            """
            params = (priority_cutoff.isoformat(), limit)
        else:
            query = """
                SELECT id, file_path, new_code, timestamp
                FROM code_memory
                WHERE embedding IS NULL
                ORDER BY timestamp DESC
                LIMIT %s
            """
            params = (limit,)

        with psycopg.connect(
            host=self.config.pg_host,
            port=self.config.pg_port,
            dbname=self.config.pg_database,
            user=self.config.pg_user,
            password=self.config.pg_password
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        entries = []
        for row in rows:
            is_priority = False
            if row[3]:
                try:
                    ts = datetime.fromisoformat(row[3].replace('Z', '+00:00'))
                    is_priority = ts > priority_cutoff
                except:
                    pass

            entries.append({
                "id": row[0],
                "file_path": row[1],
                "new_code": row[2],
                "timestamp": row[3],
                "is_priority": is_priority
            })

        return entries

    def _query_sqlite(self, limit: int, priority_first: bool) -> List[Dict[str, Any]]:
        """Query SQLite for entries without embeddings."""
        import sqlite3

        priority_cutoff = datetime.now(timezone.utc) - timedelta(hours=self.config.priority_window_hours)

        conn = sqlite3.connect(str(self.config.sqlite_path))
        try:
            if priority_first:
                query = """
                    SELECT id, file_path, new_code, timestamp
                    FROM code_entries
                    WHERE embedding IS NULL
                    ORDER BY
                        CASE WHEN timestamp > ? THEN 0 ELSE 1 END,
                        timestamp DESC
                    LIMIT ?
                """
                params = (priority_cutoff.isoformat(), limit)
            else:
                query = """
                    SELECT id, file_path, new_code, timestamp
                    FROM code_entries
                    WHERE embedding IS NULL
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                params = (limit,)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            entries = []
            for row in rows:
                is_priority = False
                if row[3]:
                    try:
                        ts = datetime.fromisoformat(row[3].replace('Z', '+00:00'))
                        is_priority = ts > priority_cutoff
                    except:
                        pass

                entries.append({
                    "id": row[0],
                    "file_path": row[1],
                    "new_code": row[2],
                    "timestamp": row[3],
                    "is_priority": is_priority
                })

            return entries
        finally:
            conn.close()

    def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> Tuple[List[List[float]], float]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of code strings to embed

        Returns:
            Tuple of (embeddings list, time in milliseconds)
        """
        if not self._model or not texts:
            return [], 0.0

        # Rate limiting for GPU operations
        if self._rate_limiter:
            if not self._rate_limiter.acquire_gpu(tokens=1, wait=True, timeout=30.0):
                logger.warning("GPU rate limit timeout - skipping batch")
                return [], 0.0

        start_time = time.time()

        # Preprocess texts
        processed = [self._preprocess_code(t) for t in texts]

        # Generate embeddings
        embeddings = self._model.encode(
            processed,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return [e.tolist() for e in embeddings], elapsed_ms

    def _preprocess_code(self, code: str) -> str:
        """Preprocess code for embedding."""
        # Normalize whitespace
        code = code.replace("\t", "    ")

        # Truncate long code (model max ~256 tokens)
        max_chars = 1500
        if len(code) > max_chars:
            head = code[:max_chars // 2]
            tail = code[-(max_chars // 2):]
            code = head + "\n...\n" + tail

        return code.strip()

    def update_embeddings_batch(
        self,
        entries: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> Tuple[int, int]:
        """
        Update embeddings in the database.

        Args:
            entries: Entry dicts with 'id' field
            embeddings: Corresponding embedding vectors

        Returns:
            Tuple of (success_count, failed_count)
        """
        if not self._backend or len(entries) != len(embeddings):
            return 0, len(entries)

        success = 0
        failed = 0

        for entry, embedding in zip(entries, embeddings):
            # Rate limiting for DB operations
            if self._rate_limiter:
                if not self._rate_limiter.acquire_db(tokens=1, wait=True, timeout=10.0):
                    logger.warning(f"DB rate limit timeout for entry {entry['id']}")
                    failed += 1
                    continue

            try:
                result = self._backend.update_embedding(entry["id"], embedding)
                if result:
                    # Clear from retry queue if was there
                    if self._retry_manager:
                        self._retry_manager.record_success(str(entry["id"]))
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.warning(f"Failed to update embedding for {entry['id']}: {e}")
                # Add to retry queue if enabled
                if self._retry_manager:
                    self._retry_manager.record_failure(
                        str(entry["id"]),
                        entry.get("file_path", ""),
                        str(e)
                    )
                failed += 1

        return success, failed

    def run_cycle(self) -> Dict[str, Any]:
        """
        Run one reindex cycle.

        Returns:
            Dict with cycle statistics
        """
        if not self._initialized:
            return {"error": "Processor not initialized"}

        cycle_id = self.metrics.start_cycle()
        logger.info(f"Starting reindex cycle #{cycle_id}")

        total_processed = 0
        total_failed = 0
        total_priority = 0
        batches = 0

        try:
            # Get entries without embeddings
            entries = self.get_entries_without_embeddings(
                limit=self.config.max_entries_per_cycle,
                priority_first=self.config.priority_batch_first
            )

            if not entries:
                logger.info("No entries need embeddings")
                stats = self._get_kb_stats()
                self.metrics.end_cycle(stats["total"], stats["with_embeddings"])
                return {
                    "cycle_id": cycle_id,
                    "processed": 0,
                    "failed": 0,
                    "priority": 0,
                    "batches": 0,
                    "coverage_percent": (stats["with_embeddings"] / stats["total"] * 100) if stats["total"] > 0 else 0,
                    "message": "No entries need embeddings"
                }

            logger.info(f"Found {len(entries)} entries needing embeddings")

            # Process in batches
            batch_size = self.config.batch_size
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i + batch_size]
                batches += 1

                # Mark batch in progress for graceful shutdown
                self._batch_in_progress.set()

                try:
                    # Count priority entries in this batch
                    priority_count = sum(1 for e in batch if e.get("is_priority"))

                    # Generate embeddings
                    texts = [e["new_code"] for e in batch]
                    embeddings, embed_time_ms = self.generate_embeddings_batch(texts)

                    if not embeddings:
                        total_failed += len(batch)
                        self.metrics.record_batch(0, 0, len(batch), priority_count, 0)
                        continue

                    # Update database
                    success, failed = self.update_embeddings_batch(batch, embeddings)

                    total_processed += success
                    total_failed += failed
                    total_priority += priority_count

                    self.metrics.record_batch(success, 0, failed, priority_count, embed_time_ms)

                    logger.debug(
                        f"Batch {batches}: {success} processed, {failed} failed, "
                        f"{embed_time_ms:.1f}ms"
                    )
                finally:
                    # Clear batch in progress
                    self._batch_in_progress.clear()

        except Exception as e:
            logger.error(f"Cycle error: {e}")
            self._batch_in_progress.clear()

        # Get final stats and end cycle
        stats = self._get_kb_stats()
        self.metrics.end_cycle(stats["total"], stats["with_embeddings"])

        result = {
            "cycle_id": cycle_id,
            "processed": total_processed,
            "failed": total_failed,
            "priority": total_priority,
            "batches": batches,
            "coverage_percent": (stats["with_embeddings"] / stats["total"] * 100) if stats["total"] > 0 else 0
        }

        logger.info(
            f"Cycle #{cycle_id} complete: {total_processed} processed, "
            f"{total_failed} failed, coverage: {result['coverage_percent']:.1f}%"
        )

        return result

    def _get_kb_stats(self) -> Dict[str, int]:
        """Get current KB statistics."""
        try:
            stats = self._backend.stats()
            return {
                "total": stats.total_entries,
                "with_embeddings": stats.entries_with_embeddings
            }
        except Exception as e:
            logger.warning(f"Could not get stats: {e}")
            return {"total": 0, "with_embeddings": 0}

    def get_status(self) -> Dict[str, Any]:
        """Get current processor status."""
        stats = self._get_kb_stats()
        status = {
            "initialized": self._initialized,
            "device": self._device,
            "model": self._active_model_name or self.config.model_name,
            "embedding_dim": self._active_embedding_dim or self.config.embedding_dim,
            "model_fallback_count": self._model_fallback_count,
            "backend": type(self._backend).__name__ if self._backend else None,
            "total_entries": stats["total"],
            "entries_with_embeddings": stats["with_embeddings"],
            "coverage_percent": (stats["with_embeddings"] / stats["total"] * 100) if stats["total"] > 0 else 0
        }
        
        if self._rate_limiter:
            status["rate_limiter"] = self._rate_limiter.get_stats()
            
        return status

    def close(self):
        """Clean up resources."""
        if self._backend:
            try:
                self._backend.close()
            except:
                pass
            self._backend = None
        self._model = None
        self._initialized = False
        self._retry_manager = None
        self._rate_limiter = None

    def warmup(self, warmup_texts: int = None) -> float:
        """
        Warmup model by generating dummy embeddings.
        
        Args:
            warmup_texts: Number of texts to use for warmup. Defaults to config.
            
        Returns:
            Warmup time in milliseconds.
        """
        if not self._model:
            return 0.0
            
        warmup_texts = warmup_texts or self.config.warmup_batch_size
        dummy_texts = ["def hello(): pass"] * warmup_texts
        
        start = time.time()
        _ = self._model.encode(dummy_texts, show_progress_bar=False)
        elapsed_ms = (time.time() - start) * 1000
        
        return elapsed_ms

    @property
    def retry_manager(self):
        """Get the retry manager instance."""
        return self._retry_manager

    @property
    def rate_limiter(self):
        """Get the rate limiter instance."""
        return self._rate_limiter

    @property
    def batch_in_progress(self) -> bool:
        """Check if a batch is currently being processed."""
        return self._batch_in_progress.is_set()

    def wait_for_batch_completion(self, timeout: float = 60.0) -> bool:
        """
        Wait for current batch to complete (for graceful shutdown).
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if batch completed, False if timed out
        """
        if not self._batch_in_progress.is_set():
            return True
            
        logger.info("Waiting for current batch to complete...")
        start = time.time()
        
        while self._batch_in_progress.is_set():
            if time.time() - start > timeout:
                logger.warning("Batch completion wait timed out")
                return False
            time.sleep(0.1)
            
        return True

    def process_retries(self) -> Dict[str, Any]:
        """
        Process entries that are due for retry.
        
        Returns:
            Dict with retry statistics.
        """
        if not self._retry_manager:
            return {"processed": 0, "succeeded": 0, "failed": 0}
            
        due_entries = self._retry_manager.get_entries_due_for_retry()
        if not due_entries:
            return {"processed": 0, "succeeded": 0, "failed": 0}
            
        logger.info(f"Processing {len(due_entries)} retry entries")
        
        succeeded = 0
        failed = 0
        
        # Get the actual entries from backend
        for entry_id in due_entries:
            try:
                # Re-fetch entry from backend
                entry = self._get_entry_by_id(entry_id)
                if not entry:
                    self._retry_manager.record_success(entry_id)  # Entry no longer exists
                    continue
                    
                # Try to generate embedding
                texts = [entry.get("new_code", "")]
                embeddings, _ = self.generate_embeddings_batch(texts)
                
                if embeddings:
                    success, fail = self.update_embeddings_batch([entry], embeddings)
                    if success > 0:
                        self._retry_manager.record_success(entry_id)
                        succeeded += 1
                    else:
                        self._retry_manager.record_failure(
                            entry_id, 
                            entry.get("file_path", ""),
                            "Update failed"
                        )
                        failed += 1
                else:
                    self._retry_manager.record_failure(
                        entry_id,
                        entry.get("file_path", ""),
                        "Embedding generation failed"
                    )
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Retry failed for {entry_id}: {e}")
                failed += 1
                
        # Update metrics
        stats = self._retry_manager.get_stats()
        self.metrics.update_retry_stats(
            retrying=stats["entries_retrying"],
            permanently_failed=stats["entries_permanently_failed"],
            retry_attempts=len(due_entries),
            retry_successes=succeeded
        )
        
        return {
            "processed": len(due_entries),
            "succeeded": succeeded,
            "failed": failed
        }

    def _get_entry_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single entry by ID from the backend."""
        from afterimage.storage import SyncPostgreSQLBackend, SQLiteBackend
        
        try:
            if isinstance(self._backend, SyncPostgreSQLBackend):
                import psycopg
                with psycopg.connect(
                    host=self.config.pg_host,
                    port=self.config.pg_port,
                    dbname=self.config.pg_database,
                    user=self.config.pg_user,
                    password=self.config.pg_password
                ) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT id, file_path, new_code, timestamp FROM code_memory WHERE id = %s",
                            (entry_id,)
                        )
                        row = cur.fetchone()
                        if row:
                            return {
                                "id": row[0],
                                "file_path": row[1],
                                "new_code": row[2],
                                "timestamp": row[3]
                            }
            else:
                import sqlite3
                conn = sqlite3.connect(str(self.config.sqlite_path))
                try:
                    cursor = conn.execute(
                        "SELECT id, file_path, new_code, timestamp FROM code_entries WHERE id = ?",
                        (entry_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        return {
                            "id": row[0],
                            "file_path": row[1],
                            "new_code": row[2],
                            "timestamp": row[3]
                        }
                finally:
                    conn.close()
        except Exception as e:
            logger.error(f"Could not fetch entry {entry_id}: {e}")
        
        return None
