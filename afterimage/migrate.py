"""
Migration script for AI-AfterImage.

Migrates data from SQLite to PostgreSQL while preserving all entries
including embeddings.
"""

import sys
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass

from .storage import SQLiteBackend, PostgreSQLBackend, StorageEntry
from .config import load_config, AfterImageConfig


@dataclass
class MigrationStats:
    """Statistics from a migration run."""
    source_entries: int = 0
    migrated_entries: int = 0
    failed_entries: int = 0
    skipped_entries: int = 0
    elapsed_seconds: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def success_rate(self) -> float:
        if self.source_entries == 0:
            return 0.0
        return (self.migrated_entries / self.source_entries) * 100


class Migrator:
    """
    Handles migration between storage backends.

    Supports SQLite to PostgreSQL migration with:
    - Batch processing for efficiency
    - Progress callbacks
    - Validation of migrated data
    - Resumable migration (skips existing entries)
    """

    def __init__(
        self,
        source: SQLiteBackend,
        target: PostgreSQLBackend,
        batch_size: int = 100,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
        """
        Initialize migrator.

        Args:
            source: Source SQLite backend
            target: Target PostgreSQL backend
            batch_size: Number of entries per batch
            progress_callback: Callback(message, current, total) for progress updates
        """
        self.source = source
        self.target = target
        self.batch_size = batch_size
        self.progress_callback = progress_callback or self._default_progress

    def _default_progress(self, message: str, current: int, total: int):
        """Default progress callback - prints to stdout."""
        pct = (current / total * 100) if total > 0 else 0
        print(f"\r{message}: {current}/{total} ({pct:.1f}%)", end="", flush=True)
        if current == total:
            print()  # Newline at end

    def migrate(self, validate: bool = True) -> MigrationStats:
        """
        Perform migration from source to target.

        Args:
            validate: Whether to validate migration after completion

        Returns:
            MigrationStats with results
        """
        return asyncio.run(self._async_migrate(validate))

    async def _async_migrate(self, validate: bool) -> MigrationStats:
        """Async migration implementation."""
        stats = MigrationStats()
        start_time = time.time()

        # Ensure backends are initialized
        self.source.initialize()
        await self.target._async_initialize()

        # Get source stats
        source_stats = self.source.stats()
        stats.source_entries = source_stats.total_entries

        self.progress_callback("Reading source data", 0, stats.source_entries)

        # Get all entries with embeddings from source
        # Use direct SQL to avoid loading all into memory at once
        import sqlite3
        conn = sqlite3.connect(self.source.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM code_memory")
        total = cursor.fetchone()[0]

        # Process in batches
        offset = 0
        processed = 0

        while offset < total:
            cursor.execute("""
                SELECT * FROM code_memory
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
            """, (self.batch_size, offset))

            rows = cursor.fetchall()
            if not rows:
                break

            batch_entries = []
            for row in rows:
                entry = self.source._row_to_entry(row)
                batch_entries.append(entry)

            # Insert batch into target
            batch_errors = await self._insert_batch(batch_entries, stats)
            stats.errors.extend(batch_errors)

            processed += len(rows)
            offset += self.batch_size
            self.progress_callback("Migrating entries", processed, total)

        conn.close()

        # Validation
        if validate:
            self.progress_callback("Validating migration", 0, 1)
            validation_errors = await self._validate_migration(stats)
            stats.errors.extend(validation_errors)
            self.progress_callback("Validating migration", 1, 1)

        stats.elapsed_seconds = time.time() - start_time
        return stats

    async def _insert_batch(
        self,
        entries: List[StorageEntry],
        stats: MigrationStats
    ) -> List[str]:
        """Insert a batch of entries into target."""
        errors = []
        pool = await self.target._get_pool()

        for entry in entries:
            try:
                # Check if entry already exists (for resumable migration)
                async with pool.acquire() as conn:
                    existing = await conn.fetchval(
                        "SELECT id FROM code_memory WHERE id = $1",
                        entry.id
                    )

                if existing:
                    stats.skipped_entries += 1
                    continue

                # Prepare embedding for pgvector
                embedding_val = None
                if entry.embedding:
                    import numpy as np
                    embedding_val = np.array(entry.embedding, dtype=np.float32)

                # Parse timestamp
                from datetime import datetime
                if isinstance(entry.timestamp, str):
                    ts = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00'))
                else:
                    ts = entry.timestamp

                # Insert
                async with pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO code_memory
                        (id, file_path, old_code, new_code, context, timestamp, session_id, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """, entry.id, entry.file_path, entry.old_code, entry.new_code,
                        entry.context, ts, entry.session_id, embedding_val)

                stats.migrated_entries += 1

            except Exception as e:
                stats.failed_entries += 1
                errors.append(f"Entry {entry.id}: {str(e)}")

        return errors

    async def _validate_migration(self, stats: MigrationStats) -> List[str]:
        """Validate migration by comparing counts and sampling entries."""
        errors = []

        # Check counts
        target_stats = await self.target._async_stats()

        expected = stats.migrated_entries + stats.skipped_entries
        actual = target_stats.total_entries

        if actual < expected:
            errors.append(
                f"Count mismatch: expected at least {expected}, got {actual}"
            )

        # Sample validation - check a few entries have matching embeddings
        import sqlite3
        conn = sqlite3.connect(self.source.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM code_memory
            WHERE embedding IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 5
        """)
        samples = cursor.fetchall()
        conn.close()

        pool = await self.target._get_pool()

        for sample in samples:
            source_entry = self.source._row_to_entry(sample)
            source_id = sample["id"]

            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT embedding FROM code_memory WHERE id = $1",
                    source_id
                )

            if row is None:
                errors.append(f"Sample {source_id}: not found in target")
                continue

            if row["embedding"] is None and source_entry.embedding is not None:
                errors.append(f"Sample {source_id}: embedding missing in target")
                continue

            if source_entry.embedding and row["embedding"] is not None:
                # Check first few values match
                target_emb = row["embedding"].tolist()
                source_emb = source_entry.embedding
                if len(target_emb) != len(source_emb):
                    errors.append(
                        f"Sample {sample['id']}: embedding dimension mismatch "
                        f"({len(target_emb)} vs {len(source_emb)})"
                    )
                elif abs(target_emb[0] - source_emb[0]) > 0.0001:
                    errors.append(
                        f"Sample {sample['id']}: embedding values differ"
                    )

        return errors


def migrate_sqlite_to_postgres(
    sqlite_path: Optional[Path] = None,
    pg_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 100,
    validate: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> MigrationStats:
    """
    Migrate from SQLite to PostgreSQL.

    Args:
        sqlite_path: Path to SQLite database. Defaults to ~/.afterimage/memory.db
        pg_config: PostgreSQL config dict (host, port, database, user, password)
        batch_size: Entries per batch
        validate: Validate after migration
        progress_callback: Progress callback function

    Returns:
        MigrationStats with results
    """
    # Load config for defaults
    config = load_config()

    # Create source backend
    if sqlite_path is None:
        sqlite_path = config.sqlite.path
    source = SQLiteBackend(db_path=sqlite_path)

    # Create target backend
    if pg_config is None:
        target = PostgreSQLBackend(
            host=config.postgresql.host,
            port=config.postgresql.port,
            database=config.postgresql.database,
            user=config.postgresql.user,
            password=config.postgresql.password,
            connection_string=config.postgresql.connection_string,
            min_pool_size=config.postgresql.min_pool_size,
            max_pool_size=config.postgresql.max_pool_size,
            embedding_dim=config.embeddings.embedding_dim
        )
    else:
        target = PostgreSQLBackend(**pg_config)

    # Run migration
    migrator = Migrator(
        source=source,
        target=target,
        batch_size=batch_size,
        progress_callback=progress_callback
    )

    return migrator.migrate(validate=validate)


def print_migration_report(stats: MigrationStats):
    """Print a formatted migration report."""
    print("\n" + "=" * 60)
    print("AI-AfterImage Migration Report")
    print("=" * 60)
    print(f"Source entries:     {stats.source_entries}")
    print(f"Migrated:           {stats.migrated_entries}")
    print(f"Skipped (existing): {stats.skipped_entries}")
    print(f"Failed:             {stats.failed_entries}")
    print(f"Success rate:       {stats.success_rate:.1f}%")
    print(f"Elapsed time:       {stats.elapsed_seconds:.1f}s")

    if stats.errors:
        print(f"\nErrors ({len(stats.errors)}):")
        for error in stats.errors[:10]:  # Show first 10
            print(f"  - {error}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more")

    print("=" * 60)


if __name__ == "__main__":
    # CLI for direct migration
    import argparse

    parser = argparse.ArgumentParser(description="Migrate AfterImage from SQLite to PostgreSQL")
    parser.add_argument("--sqlite", help="Source SQLite path")
    parser.add_argument("--pg-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--pg-port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--pg-database", default="afterimage", help="PostgreSQL database")
    parser.add_argument("--pg-user", default="afterimage", help="PostgreSQL user")
    parser.add_argument("--pg-password", help="PostgreSQL password")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")

    args = parser.parse_args()

    sqlite_path = Path(args.sqlite).expanduser() if args.sqlite else None

    pg_config = {
        "host": args.pg_host,
        "port": args.pg_port,
        "database": args.pg_database,
        "user": args.pg_user,
        "password": args.pg_password,
    }

    print("Starting migration...")
    stats = migrate_sqlite_to_postgres(
        sqlite_path=sqlite_path,
        pg_config=pg_config,
        batch_size=args.batch_size,
        validate=not args.no_validate
    )

    print_migration_report(stats)
    sys.exit(0 if stats.failed_entries == 0 else 1)
