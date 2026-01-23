"""
AI-AfterImage: Episodic memory for Claude Code.

Provides persistent memory of code written across sessions through
a Claude Code hook system with SQLite/PostgreSQL + vector embeddings.

Version 0.4.0 adds language detection, AST parsing, and semantic intelligence:
- Language detection with pattern-based confidence scoring (20+ languages)
- Tree-sitter AST parsing for Python, JavaScript, TypeScript, Rust, Go, C/C++
- Semantic index with go-to-definition, find-references, and hover info

Version 0.3.0 adds code churn tracking with file stability tiers
(Gold/Silver/Bronze/Red) and warnings for high-churn patterns.

Version 0.2.0 adds PostgreSQL backend with pgvector for concurrent
write support in multi-agent AtlasForge workflows.
"""

__version__ = "0.4.0"

from .kb import KnowledgeBase
from .search import HybridSearch, SearchResult
from .config import load_config, get_storage_backend, AfterImageConfig
from .storage import StorageBackend, StorageEntry, SQLiteBackend, PostgreSQLBackend

# Language detection is always available (no external dependencies)
from .language_detection import LanguageDetector, LanguageResult, ConfidenceTier, detect_language

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
    # Language detection
    "LanguageDetector",
    "LanguageResult",
    "ConfidenceTier",
    "detect_language",
    # Version
    "__version__",
]


def get_ast_parser():
    """Get the AST parser factory (requires tree-sitter dependencies)."""
    from .ast_parser import ASTParserFactory
    return ASTParserFactory


def get_semantic_index():
    """Get the SemanticIndex class (requires tree-sitter dependencies)."""
    from .semantic_index import SemanticIndex
    return SemanticIndex
