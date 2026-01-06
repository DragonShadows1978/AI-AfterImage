"""
Search System: Hybrid search combining FTS5 and vector similarity.

Provides ranked results using both keyword matching and semantic similarity.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .kb import KnowledgeBase
from .embeddings import EmbeddingGenerator, cosine_similarity


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


class HybridSearch:
    """
    Hybrid search combining FTS5 full-text search and vector similarity.

    Scoring formula:
        relevance = (fts_weight * fts_score) + (semantic_weight * semantic_score)

    Where:
        - fts_score: Normalized BM25 score from SQLite FTS5
        - semantic_score: Cosine similarity from embeddings
    """

    def __init__(
        self,
        kb: Optional[KnowledgeBase] = None,
        embedder: Optional[EmbeddingGenerator] = None,
        fts_weight: float = 0.4,
        semantic_weight: float = 0.6
    ):
        """
        Initialize hybrid search.

        Args:
            kb: Knowledge base to search (creates new if None)
            embedder: Embedding generator (creates new if None)
            fts_weight: Weight for FTS5 scores (0-1)
            semantic_weight: Weight for semantic scores (0-1)
        """
        self.kb = kb or KnowledgeBase()
        self._embedder = embedder
        self.fts_weight = fts_weight
        self.semantic_weight = semantic_weight

    @property
    def embedder(self) -> EmbeddingGenerator:
        """Lazy load embedder."""
        if self._embedder is None:
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
        Perform hybrid search combining FTS5 and vector similarity.

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

        # 1. FTS5 search
        fts_results = self._search_fts(query, limit * 2, path_filter)
        for entry, score in fts_results:
            result = self._entry_to_result(entry)
            result.fts_score = score
            results_map[entry["id"]] = result

        # 2. Semantic search
        semantic_results = self._search_semantic(query, limit * 2, path_filter)
        for entry, score in semantic_results:
            if entry["id"] in results_map:
                # Combine scores
                results_map[entry["id"]].semantic_score = score
            else:
                result = self._entry_to_result(entry)
                result.semantic_score = score
                results_map[entry["id"]] = result

        # 3. Calculate combined scores
        for result in results_map.values():
            # Normalize and combine
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

        Optimized for finding code that is functionally similar
        to the provided snippet.

        Args:
            code: Code snippet to find matches for
            file_path: Optional file path for context
            limit: Maximum results
            threshold: Minimum similarity threshold

        Returns:
            List of similar code snippets
        """
        # Generate embedding for the code
        from .embeddings import cached_embed
        query_embedding = self.embedder.embed_code(code, file_path)

        # Get all entries with embeddings
        entries = self.kb.get_all_with_embeddings()

        # Calculate similarities
        results = []
        for entry in entries:
            if entry.get("embedding"):
                similarity = cosine_similarity(query_embedding, entry["embedding"])
                if similarity >= threshold:
                    result = self._entry_to_result(entry)
                    result.semantic_score = similarity
                    result.relevance_score = similarity
                    results.append(result)

        # Sort by similarity
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

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
        entries = self.kb.search_by_path(path_pattern, limit)
        return [self._entry_to_result(e) for e in entries]

    def _search_fts(
        self,
        query: str,
        limit: int,
        path_filter: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Perform FTS5 search and return normalized scores."""
        # Escape special FTS5 characters
        safe_query = self._escape_fts_query(query)

        try:
            results = self.kb.search_fts(safe_query, limit)
        except Exception:
            # FTS query failed - try simpler query
            words = query.split()
            if words:
                safe_query = " OR ".join(f'"{w}"' for w in words[:5])
                try:
                    results = self.kb.search_fts(safe_query, limit)
                except Exception:
                    return []
            else:
                return []

        # Apply path filter if specified
        if path_filter:
            results = [r for r in results if path_filter in r.get("file_path", "")]

        # Normalize BM25 scores (they are negative, closer to 0 is better)
        if not results:
            return []

        # BM25 scores are negative; convert to 0-1 range
        scores = [abs(r.get("fts_rank", 0)) for r in results]
        max_score = max(scores) if scores else 1

        normalized = []
        for r, score in zip(results, scores):
            # Invert and normalize: higher is better
            norm_score = 1 - (score / (max_score + 1)) if max_score > 0 else 0.5
            normalized.append((r, norm_score))

        return normalized

    def _search_semantic(
        self,
        query: str,
        limit: int,
        path_filter: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Perform semantic search using embeddings."""
        # Generate query embedding
        try:
            query_embedding = self.embedder.embed(query)
        except Exception:
            # Embeddings not available
            return []

        # Get all entries with embeddings
        entries = self.kb.get_all_with_embeddings()

        # Apply path filter
        if path_filter:
            entries = [e for e in entries if path_filter in e.get("file_path", "")]

        # Calculate similarities
        results = []
        for entry in entries:
            if entry.get("embedding"):
                similarity = cosine_similarity(query_embedding, entry["embedding"])
                results.append((entry, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _escape_fts_query(self, query: str) -> str:
        """Escape special characters for FTS5 query."""
        # Remove or escape FTS5 special characters
        special_chars = ['"', "'", "(", ")", "*", ":", "-", "^"]
        escaped = query
        for char in special_chars:
            escaped = escaped.replace(char, " ")

        # Clean up whitespace
        escaped = " ".join(escaped.split())

        # If query is empty after escaping, return a catch-all
        if not escaped.strip():
            return "*"

        return escaped

    def _entry_to_result(self, entry: Dict[str, Any]) -> SearchResult:
        """Convert a KB entry to a SearchResult."""
        return SearchResult(
            id=entry["id"],
            file_path=entry["file_path"],
            new_code=entry["new_code"],
            old_code=entry.get("old_code"),
            context=entry.get("context"),
            timestamp=entry["timestamp"],
            session_id=entry.get("session_id"),
        )


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
