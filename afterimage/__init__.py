"""
AI-AfterImage: Episodic memory for Claude Code.

Provides persistent memory of code written across sessions through
a Claude Code hook system with SQLite + vector embeddings.
"""

__version__ = "0.1.0"

from .kb import KnowledgeBase
from .filter import CodeFilter
from .search import HybridSearch
from .inject import ContextInjector
from .extract import TranscriptExtractor

__all__ = [
    "KnowledgeBase",
    "CodeFilter",
    "HybridSearch",
    "ContextInjector",
    "TranscriptExtractor",
    "__version__",
]
