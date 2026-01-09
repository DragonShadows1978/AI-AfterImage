"""
AI-AfterImage: Episodic memory for Claude Code.

Provides persistent memory of code written across sessions through
a Claude Code hook system with SQLite/PostgreSQL + vector embeddings.

Version 0.2.0 adds PostgreSQL backend with pgvector for concurrent
write support in multi-agent RDE workflows.
"""

__version__ = "0.2.0"

from .kb import KnowledgeBase
from .search import HybridSearch, SearchResult
from .config import load_config, get_storage_backend, AfterImageConfig
from .storage import StorageBackend, StorageEntry, SQLiteBackend, PostgreSQLBackend

__all__ = [
    # Core classes
    "KnowledgeBase",
    "HybridSearch",
    "SearchResult",
    # Storage backends
    "StorageBackend",
    "StorageEntry",
    "SQLiteBackend",
    "PostgreSQLBackend",
    # Configuration
    "load_config",
    "get_storage_backend",
    "AfterImageConfig",
    # Version
    "__version__",
]
