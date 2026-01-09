"""
Storage backends for AI-AfterImage.

Provides abstract interface and concrete implementations for SQLite and PostgreSQL.
"""

from .base import StorageBackend, StorageEntry, StorageStats
from .sqlite_backend import SQLiteBackend
from .postgres_backend import PostgreSQLBackend, SyncPostgreSQLBackend

__all__ = [
    "StorageBackend",
    "StorageEntry",
    "StorageStats",
    "SQLiteBackend",
    "PostgreSQLBackend",
    "SyncPostgreSQLBackend",
]
