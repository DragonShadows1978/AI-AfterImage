"""
Search System: Hybrid search combining FTS and vector similarity.

Updated to use the storage backend abstraction for both SQLite and PostgreSQL.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .storage import StorageBackend, StorageEntry
from .config import load_config, get_storage_backend


@dataclass
class SearchResult:
    """A single search result with relevance scoring."""
    id: str
    file_path: str
    new_code: str
    old_code: Optional[str]
    context: Optional[str]
    timestamp: str
    session_id: Optional[str]

    # Scoring
    relevance_score: float = 0.0
    fts_score: float = 0.0
    semantic_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "new_code": self.new_code,
            "old_code": self.old_code,
            "context": self.context,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "relevance_score": self.relevance_score,
            "fts_score": self.fts_score,
            "semantic_score": self.semantic_score,
        }

    @classmethod
    def from_entry(cls, entry: StorageEntry) -> "SearchResult":
        """Create SearchResult from StorageEntry."""
        return cls(
            id=entry.id,
            file_path=entry.file_path,
            new_code=entry.new_code,
            old_code=entry.old_code,
            context=entry.context,
            timestamp=entry.timestamp,
            session_id=entry.session_id,
            fts_score=entry.fts_rank or 0.0,
            semantic_score=entry.semantic_score or 0.0,
        )


class HybridSearch:
    """
    Hybrid search combining FTS full-text search and vector similarity.

    Works with both SQLite and PostgreSQL backends via the StorageBackend
    abstraction. Each backend handles search differently:
    - SQLite: FTS5 with BM25, in-memory cosine similarity
    - PostgreSQL: tsvector with GIN index, pgvector HNSW index

    Scoring formula:
        relevance = (fts_weight * fts_score) + (semantic_weight * semantic_score)
    """

    def __init__(
        self,
        backend: Optional[StorageBackend] = None,
        embedder=None,
        fts_weight: float = 0.4,
        semantic_weight: float = 0.6
    ):
        """
        Initialize hybrid search.

        Args:
            backend: Storage backend (loads from config if None)
            embedder: Embedding generator (creates new if None)
            fts_weight: Weight for FTS scores (0-1)
            semantic_weight: Weight for semantic scores (0-1)
        """
        self._backend = backend
        self._embedder = embedder
        self.fts_weight = fts_weight
        self.semantic_weight = semantic_weight

    @property
    def backend(self) -> StorageBackend:
        """Lazy load backend."""
        if self._backend is None:
            self._backend = get_storage_backend()
        return self._backend

    @property
    def embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from .embeddings import EmbeddingGenerator
            self._embedder = EmbeddingGenerator()
        return self._embedder

    def search(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.3,
        path_filter: Optional[str] = None,
        include_fts_only: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining FTS and vector similarity.

        Args:
            query: Search query (natural language or code pattern)
            limit: Maximum number of results
            threshold: Minimum relevance score (0-1)
            path_filter: Optional file path filter pattern
            include_fts_only: Include results that only match FTS (no embedding)

        Returns:
            List of SearchResult objects, sorted by relevance
        """
        results_map: Dict[str, SearchResult] = {}

        # 1. FTS search
        fts_results = self._search_fts(query, limit * 2, path_filter)
        for entry in fts_results:
            result = SearchResult.from_entry(entry)
            result.fts_score = self._normalize_fts_score(entry.fts_rank)
            results_map[entry.id] = result

        # 2. Semantic search
        semantic_results = self._search_semantic(query, limit * 2, path_filter)
        for entry, score in semantic_results:
            if entry.id in results_map:
                results_map[entry.id].semantic_score = score
            else:
                result = SearchResult.from_entry(entry)
                result.semantic_score = score
                results_map[entry.id] = result

        # 3. Calculate combined scores
        for result in results_map.values():
            fts_normalized = min(result.fts_score, 1.0) if result.fts_score > 0 else 0
            result.relevance_score = (
                self.fts_weight * fts_normalized +
                self.semantic_weight * result.semantic_score
            )

        # 4. Filter and sort
        filtered = [
            r for r in results_map.values()
            if r.relevance_score >= threshold or
               (include_fts_only and r.fts_score > 0)
        ]
        filtered.sort(key=lambda r: r.relevance_score, reverse=True)

        return filtered[:limit]

    def search_by_code(
        self,
        code: str,
        file_path: Optional[str] = None,
        limit: int = 5,
        threshold: float = 0.4
    ) -> List[SearchResult]:
        """
        Search for similar code snippets.

        Args:
            code: Code snippet to find matches for
            file_path: Optional file path for context
            limit: Maximum results
            threshold: Minimum similarity threshold

        Returns:
            List of similar code snippets
        """
        # Generate embedding for the code
        query_embedding = self.embedder.embed_code(code, file_path)

        # Semantic search
        results = self.backend.search_semantic(query_embedding, limit * 2)

        # Filter and convert
        search_results = []
        for entry, similarity in results:
            if similarity >= threshold:
                result = SearchResult.from_entry(entry)
                result.semantic_score = similarity
                result.relevance_score = similarity
                search_results.append(result)

        search_results.sort(key=lambda r: r.relevance_score, reverse=True)
        return search_results[:limit]

    def search_by_path(
        self,
        path_pattern: str,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search by file path pattern.

        Args:
            path_pattern: Pattern to match against file paths
            limit: Maximum results

        Returns:
            List of matching entries
        """
        entries = self.backend.search_by_path(path_pattern, limit)
        return [SearchResult.from_entry(e) for e in entries]

    def _search_fts(
        self,
        query: str,
        limit: int,
        path_filter: Optional[str] = None
    ) -> List[StorageEntry]:
        """Perform FTS search."""
        results = self.backend.search_fts(query, limit)

        if path_filter:
            results = [r for r in results if path_filter in r.file_path]

        return results

    def _search_semantic(
        self,
        query: str,
        limit: int,
        path_filter: Optional[str] = None
    ) -> List[Tuple[StorageEntry, float]]:
        """Perform semantic search."""
        try:
            query_embedding = self.embedder.embed(query)
        except Exception:
            return []

        results = self.backend.search_semantic(query_embedding, limit)

        if path_filter:
            results = [(e, s) for e, s in results if path_filter in e.file_path]

        return results

    def _normalize_fts_score(self, score: Optional[float]) -> float:
        """Normalize FTS score to 0-1 range."""
        if score is None:
            return 0.0
        # SQLite BM25 returns negative values (closer to 0 is better)
        # PostgreSQL ts_rank returns positive values
        if score < 0:
            # SQLite: Convert negative to positive, normalize
            return min(1.0, 1.0 / (1.0 + abs(score)))
        else:
            # PostgreSQL: Already positive, cap at 1
            return min(1.0, score)


def quick_search(query: str, limit: int = 5) -> List[SearchResult]:
    """
    Convenience function for quick searches.

    Args:
        query: Search query
        limit: Maximum results

    Returns:
        List of search results
    """
    search = HybridSearch()
    return search.search(query, limit=limit)
