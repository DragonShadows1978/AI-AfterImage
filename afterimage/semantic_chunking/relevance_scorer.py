"""
Relevance Scorer: Multi-factor relevance scoring for code snippets.

Scores code snippets based on multiple factors:
- Recency: More recent code is more relevant
- File proximity: Code from related files is more relevant
- Semantic similarity: Code with similar meaning/purpose is more relevant
- Project awareness: Code from the current project is prioritized

Part of AfterImage Semantic Chunking v0.3.0.
"""

import math
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ScoringConfig:
    """Configuration for relevance scoring."""
    # Weight factors (should sum to 1.0)
    recency_weight: float = 0.20
    proximity_weight: float = 0.25
    semantic_weight: float = 0.35
    project_weight: float = 0.20

    # Recency decay parameters
    recency_half_life_days: float = 7.0
    recency_max_days: float = 90.0

    # Proximity parameters
    same_file_bonus: float = 0.5
    same_dir_bonus: float = 0.3
    same_project_bonus: float = 0.2

    # Minimum score to consider
    min_relevance_score: float = 0.3


@dataclass
class ScoredSnippet:
    """A code snippet with computed relevance scores."""
    code: str
    file_path: str
    timestamp: str
    context: Optional[str] = None

    # Individual scores
    recency_score: float = 0.0
    proximity_score: float = 0.0
    semantic_score: float = 0.0
    project_score: float = 0.0

    # Combined score
    relevance_score: float = 0.0

    # Metadata
    chunk_name: Optional[str] = None
    chunk_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "file_path": self.file_path,
            "timestamp": self.timestamp,
            "context": self.context,
            "recency_score": self.recency_score,
            "proximity_score": self.proximity_score,
            "semantic_score": self.semantic_score,
            "project_score": self.project_score,
            "relevance_score": self.relevance_score,
            "chunk_name": self.chunk_name,
            "chunk_type": self.chunk_type,
        }


class RelevanceScorer:
    """
    Computes multi-factor relevance scores for code snippets.

    Combines recency, file proximity, semantic similarity, and project
    awareness into a single relevance score.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        """
        Initialize the scorer.

        Args:
            config: Scoring configuration (uses defaults if None)
        """
        self.config = config or ScoringConfig()
        self._current_project: Optional[str] = None
        self._current_file: Optional[str] = None

    def set_context(
        self,
        current_file: Optional[str] = None,
        current_project: Optional[str] = None
    ):
        """
        Set the current context for proximity scoring.

        Args:
            current_file: Path to the file being written/edited
            current_project: Project root directory
        """
        # Sanitize inputs - remove null bytes and strip whitespace
        if current_file:
            current_file = current_file.replace('\x00', '').strip() or None
        if current_project:
            current_project = current_project.replace('\x00', '').strip() or None

        self._current_file = current_file
        self._current_project = current_project or self._detect_project(current_file)

    def _detect_project(self, file_path: Optional[str]) -> Optional[str]:
        """Detect project root from file path."""
        if not file_path:
            return None

        path = Path(file_path).resolve()
        markers = [".git", "pyproject.toml", "package.json", "Cargo.toml",
                   "go.mod", "setup.py", "requirements.txt", ".project"]

        current = path.parent
        while current != current.parent:
            for marker in markers:
                if (current / marker).exists():
                    return str(current)
            current = current.parent

        return None

    def score_snippets(
        self,
        snippets: List[Dict[str, Any]],
        query_embedding: Optional[List[float]] = None
    ) -> List[ScoredSnippet]:
        """
        Score a list of code snippets.

        Args:
            snippets: List of snippet dictionaries with code, file_path, timestamp
            query_embedding: Optional query embedding for semantic scoring

        Returns:
            List of ScoredSnippet objects, sorted by relevance (highest first)
        """
        scored = []

        for snippet in snippets:
            scored_snippet = self._score_snippet(snippet, query_embedding)
            if scored_snippet.relevance_score >= self.config.min_relevance_score:
                scored.append(scored_snippet)

        scored.sort(key=lambda s: s.relevance_score, reverse=True)
        return scored

    def _score_snippet(
        self,
        snippet: Dict[str, Any],
        query_embedding: Optional[List[float]] = None
    ) -> ScoredSnippet:
        """Score a single snippet."""
        scored = ScoredSnippet(
            code=snippet.get("code", snippet.get("new_code", "")),
            file_path=snippet.get("file_path", ""),
            timestamp=snippet.get("timestamp", ""),
            context=snippet.get("context"),
            chunk_name=snippet.get("chunk_name"),
            chunk_type=snippet.get("chunk_type"),
        )

        scored.recency_score = self._compute_recency_score(scored.timestamp)
        scored.proximity_score = self._compute_proximity_score(scored.file_path)
        scored.project_score = self._compute_project_score(scored.file_path)

        if "semantic_score" in snippet:
            scored.semantic_score = snippet["semantic_score"]
        elif query_embedding and "embedding" in snippet:
            scored.semantic_score = self._compute_semantic_score(
                query_embedding, snippet["embedding"]
            )

        scored.relevance_score = self._combine_scores(scored)
        return scored

    def _compute_recency_score(self, timestamp: str) -> float:
        """Compute recency score using exponential decay."""
        if not timestamp:
            return 0.5

        try:
            if "T" in timestamp:
                entry_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                entry_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                entry_time = entry_time.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            age_days = (now - entry_time).total_seconds() / 86400

            if age_days > self.config.recency_max_days:
                return 0.0

            decay_rate = math.log(2) / self.config.recency_half_life_days
            score = math.exp(-decay_rate * age_days)

            return min(1.0, max(0.0, score))

        except (ValueError, TypeError):
            return 0.5

    def _compute_proximity_score(self, file_path: str) -> float:
        """Compute file proximity score."""
        if not self._current_file or not file_path:
            return 0.5

        sanitized_current = self._current_file.replace('\x00', '').strip()
        sanitized_path = file_path.replace('\x00', '').strip()

        if not sanitized_current or not sanitized_path:
            return 0.5

        try:
            current_path = Path(sanitized_current).resolve()
            snippet_path = Path(sanitized_path).resolve()
        except (ValueError, OSError):
            return 0.5

        if current_path == snippet_path:
            return 1.0

        if current_path.parent == snippet_path.parent:
            return 0.8

        if current_path.parent.parent == snippet_path.parent.parent:
            return 0.6

        current_parts = current_path.parts
        snippet_parts = snippet_path.parts

        shared = 0
        for c, s in zip(current_parts, snippet_parts):
            if c == s:
                shared += 1
            else:
                break

        max_parts = max(len(current_parts), len(snippet_parts))
        if max_parts > 0:
            return 0.3 + 0.4 * (shared / max_parts)

        return 0.3

    def _compute_project_score(self, file_path: str) -> float:
        """Compute project awareness score."""
        if not self._current_project or not file_path:
            return 0.5

        try:
            snippet_path = str(Path(file_path).resolve())

            if snippet_path.startswith(self._current_project):
                return 1.0

            return 0.3

        except (ValueError, TypeError):
            return 0.5

    def _compute_semantic_score(
        self,
        query_embedding: List[float],
        snippet_embedding: List[float]
    ) -> float:
        """Compute semantic similarity score using cosine similarity."""
        if not query_embedding or not snippet_embedding:
            return 0.5

        if len(query_embedding) != len(snippet_embedding):
            return 0.5

        dot_product = sum(a * b for a, b in zip(query_embedding, snippet_embedding))
        norm_a = math.sqrt(sum(a * a for a in query_embedding))
        norm_b = math.sqrt(sum(b * b for b in snippet_embedding))

        if norm_a == 0 or norm_b == 0:
            return 0.5

        similarity = dot_product / (norm_a * norm_b)
        return (similarity + 1) / 2

    def _combine_scores(self, scored: ScoredSnippet) -> float:
        """Combine individual scores into final relevance score."""
        score = (
            self.config.recency_weight * scored.recency_score +
            self.config.proximity_weight * scored.proximity_score +
            self.config.semantic_weight * scored.semantic_score +
            self.config.project_weight * scored.project_score
        )
        return min(1.0, max(0.0, score))

    def boost_for_dependencies(
        self,
        scored_snippets: List[ScoredSnippet],
        dependencies: List[str]
    ) -> List[ScoredSnippet]:
        """
        Boost scores for snippets that define requested dependencies.

        Args:
            scored_snippets: Already scored snippets
            dependencies: List of symbol names being used

        Returns:
            Snippets with boosted scores where applicable
        """
        if not dependencies:
            return scored_snippets

        dep_set = set(dependencies)

        for snippet in scored_snippets:
            chunk_name = snippet.chunk_name or ""
            if chunk_name in dep_set:
                snippet.relevance_score = min(1.0, snippet.relevance_score * 1.2)

            code = snippet.code
            for dep in dep_set:
                if f"class {dep}" in code or f"def {dep}" in code:
                    snippet.relevance_score = min(1.0, snippet.relevance_score * 1.15)
                    break

        scored_snippets.sort(key=lambda s: s.relevance_score, reverse=True)
        return scored_snippets


def quick_score(
    snippets: List[Dict[str, Any]],
    current_file: Optional[str] = None,
    current_project: Optional[str] = None
) -> List[ScoredSnippet]:
    """
    Convenience function to quickly score snippets.

    Args:
        snippets: List of snippet dictionaries
        current_file: Current file being edited
        current_project: Current project root

    Returns:
        Sorted list of scored snippets
    """
    scorer = RelevanceScorer()
    scorer.set_context(current_file, current_project)
    return scorer.score_snippets(snippets)
