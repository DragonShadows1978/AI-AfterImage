"""
Smart Context Injector: Enhanced context injection with semantic chunking.

This is the main integration module that combines:
- Semantic chunking (break code into meaningful units)
- Token budget management (respect Claude's context limits)
- Multi-factor relevance scoring (recency, proximity, semantic, project)
- Summary mode (condense similar snippets)
- Project-aware weighting (prioritize current project code)

Part of AfterImage Semantic Chunking v0.3.0.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from .chunker import SemanticChunker, CodeChunk, ChunkType
from .token_budget import TokenBudgetManager, TokenBudgetConfig, TokenBudgetTier
from .relevance_scorer import RelevanceScorer, ScoringConfig, ScoredSnippet
from .snippet_summarizer import (
    SnippetSummarizer, SummaryConfig, SnippetGroup,
    SummaryFormatter
)


@dataclass
class SmartInjectionConfig:
    """Configuration for smart context injection."""
    # Token budget
    max_tokens: int = 2000
    token_tier: TokenBudgetTier = TokenBudgetTier.STANDARD

    # Chunking
    max_chunk_tokens: int = 500
    chunk_enabled: bool = True

    # Relevance scoring
    min_relevance_score: float = 0.3
    recency_weight: float = 0.20
    proximity_weight: float = 0.25
    semantic_weight: float = 0.35
    project_weight: float = 0.20

    # Summarization
    summary_enabled: bool = True
    similarity_threshold: float = 0.7
    max_individual_snippets: int = 3

    # Output limits
    max_results: int = 5
    include_context: bool = True
    include_file_path: bool = True


@dataclass
class InjectionResult:
    """Result of context injection."""
    injection_text: str
    tokens_used: int
    snippets_included: int
    groups_included: int
    total_candidates: int
    was_summarized: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "injection_text": self.injection_text,
            "tokens_used": self.tokens_used,
            "snippets_included": self.snippets_included,
            "groups_included": self.groups_included,
            "total_candidates": self.total_candidates,
            "was_summarized": self.was_summarized,
        }


class SmartContextInjector:
    """
    Enhanced context injector with intelligent semantic chunking.

    Provides significantly improved context injection over the basic
    ContextInjector by:
    1. Breaking code into semantic chunks instead of raw files
    2. Managing token budget to maximize useful context
    3. Scoring snippets by multiple relevance factors
    4. Summarizing similar snippets to reduce redundancy
    5. Prioritizing code from the current project
    """

    def __init__(self, config: Optional[SmartInjectionConfig] = None):
        """
        Initialize the smart injector.

        Args:
            config: Injection configuration (uses defaults if None)
        """
        self.config = config or SmartInjectionConfig()

        # Initialize components
        self.chunker = SemanticChunker(
            max_chunk_tokens=self.config.max_chunk_tokens
        )

        self.budget_manager = TokenBudgetManager(
            TokenBudgetConfig(
                max_tokens=self.config.max_tokens,
                max_snippet_tokens=self.config.max_chunk_tokens
            )
        )

        self.scorer = RelevanceScorer(
            ScoringConfig(
                recency_weight=self.config.recency_weight,
                proximity_weight=self.config.proximity_weight,
                semantic_weight=self.config.semantic_weight,
                project_weight=self.config.project_weight,
                min_relevance_score=self.config.min_relevance_score
            )
        )

        self.summarizer = SnippetSummarizer(
            SummaryConfig(
                similarity_threshold=self.config.similarity_threshold,
                max_individual_snippets=self.config.max_individual_snippets
            )
        )

        self.formatter = SummaryFormatter()

        # Context
        self._current_file: Optional[str] = None
        self._current_project: Optional[str] = None

    def set_context(
        self,
        current_file: Optional[str] = None,
        current_project: Optional[str] = None
    ):
        """
        Set the current context for relevance scoring.

        Args:
            current_file: Path to the file being written/edited
            current_project: Project root directory
        """
        self._current_file = current_file
        self._current_project = current_project
        self.scorer.set_context(current_file, current_project)

    def inject(
        self,
        raw_results: List[Dict[str, Any]],
        query_context: Optional[str] = None,
        query_embedding: Optional[List[float]] = None
    ) -> InjectionResult:
        """
        Create an optimized context injection from search results.

        Args:
            raw_results: Search results from AfterImage (list of dicts with
                        code, file_path, timestamp, etc.)
            query_context: Optional context about what triggered the search
            query_embedding: Optional query embedding for semantic scoring

        Returns:
            InjectionResult with optimized injection text
        """
        if not raw_results:
            return InjectionResult(
                injection_text="",
                tokens_used=0,
                snippets_included=0,
                groups_included=0,
                total_candidates=0,
                was_summarized=False
            )

        # Step 1: Chunk the raw results if enabled
        if self.config.chunk_enabled:
            chunked_results = self._chunk_results(raw_results)
        else:
            chunked_results = self._convert_to_snippets(raw_results)

        # Step 2: Score all snippets
        scored_snippets = self.scorer.score_snippets(
            [self._snippet_to_dict(s) for s in chunked_results],
            query_embedding
        )

        # Step 3: Summarize if enabled and many results
        if self.config.summary_enabled and len(scored_snippets) > self.config.max_individual_snippets:
            individual, groups = self.summarizer.summarize(
                scored_snippets,
                max_output=self.config.max_results
            )
            was_summarized = len(groups) > 0
        else:
            individual = scored_snippets[:self.config.max_results]
            groups = []
            was_summarized = False

        # Step 4: Format within token budget
        injection_text = self._format_injection(
            individual, groups, query_context
        )

        tokens_used = self.budget_manager.estimator.estimate(injection_text, "mixed")

        return InjectionResult(
            injection_text=injection_text,
            tokens_used=tokens_used,
            snippets_included=len(individual),
            groups_included=len(groups),
            total_candidates=len(raw_results),
            was_summarized=was_summarized
        )

    def inject_for_hook(
        self,
        raw_results: List[Dict[str, Any]],
        file_path: str,
        tool_type: str
    ) -> Optional[str]:
        """
        Format results specifically for Claude Code hook injection.

        This is a specialized method for the AfterImage hook that creates
        concise, actionable context.

        Args:
            raw_results: Search results from AfterImage
            file_path: Path of file being written
            tool_type: "Write" or "Edit"

        Returns:
            Hook-formatted injection string, or None if no relevant results
        """
        # Set context from the file being written
        self.set_context(current_file=file_path)

        # Get injection result
        result = self.inject(raw_results)

        if not result.injection_text:
            return None

        # Wrap in memory tag for hooks
        action = "creating" if tool_type == "Write" else "editing"
        file_name = Path(file_path).name

        parts = [
            f'<memory context="You are {action} {file_name}">',
            "Previously written similar code:",
            "",
            result.injection_text,
            "</memory>"
        ]

        return "\n".join(parts)

    def _chunk_results(
        self,
        raw_results: List[Dict[str, Any]]
    ) -> List[ScoredSnippet]:
        """Chunk raw results into semantic units."""
        all_chunks = []

        for result in raw_results:
            code = result.get("new_code", result.get("code", ""))
            file_path = result.get("file_path", "")
            timestamp = result.get("timestamp", "")
            context = result.get("context", "")

            # Parse into chunks
            chunks = self.chunker.chunk_code(code, file_path)

            # Convert chunks to scored snippets
            for chunk in chunks:
                snippet = ScoredSnippet(
                    code=chunk.code,
                    file_path=file_path,
                    timestamp=timestamp,
                    context=context,
                    chunk_name=chunk.name,
                    chunk_type=chunk.chunk_type.value,
                    semantic_score=result.get("semantic_score", 0.0)
                )
                all_chunks.append(snippet)

        return all_chunks

    def _convert_to_snippets(
        self,
        raw_results: List[Dict[str, Any]]
    ) -> List[ScoredSnippet]:
        """Convert raw results to scored snippets without chunking."""
        return [
            ScoredSnippet(
                code=r.get("new_code", r.get("code", "")),
                file_path=r.get("file_path", ""),
                timestamp=r.get("timestamp", ""),
                context=r.get("context", ""),
                semantic_score=r.get("semantic_score", 0.0)
            )
            for r in raw_results
        ]

    def _snippet_to_dict(self, snippet: ScoredSnippet) -> Dict[str, Any]:
        """Convert a scored snippet to a dict for scoring."""
        return {
            "code": snippet.code,
            "file_path": snippet.file_path,
            "timestamp": snippet.timestamp,
            "context": snippet.context,
            "chunk_name": snippet.chunk_name,
            "chunk_type": snippet.chunk_type,
            "semantic_score": snippet.semantic_score,
        }

    def _format_injection(
        self,
        individual: List[ScoredSnippet],
        groups: List[SnippetGroup],
        query_context: Optional[str] = None
    ) -> str:
        """Format the final injection within token budget."""
        available_tokens = self.config.max_tokens

        # Build header
        total_items = len(individual) + sum(g.size for g in groups)
        if total_items == 1:
            header = "You have written similar code before:\n\n"
        else:
            header = f"You have written similar code before ({total_items} relevant snippets):\n\n"

        header_tokens = self.budget_manager.estimator.estimate(header, "prose")
        available_tokens -= header_tokens

        # Format body using the formatter
        body = self.formatter.format_for_injection(
            individual, groups, available_tokens
        )

        return header + body


class ProjectContextManager:
    """
    Manages project-aware context for improved relevance scoring.

    Tracks the current project and provides context information
    for the relevance scorer.
    """

    def __init__(self):
        self._project_cache: Dict[str, str] = {}
        self._current_project: Optional[str] = None

    def detect_project(self, file_path: str) -> Optional[str]:
        """Detect the project root for a file path."""
        if file_path in self._project_cache:
            return self._project_cache[file_path]

        project = self._find_project_root(file_path)
        self._project_cache[file_path] = project
        return project

    def _find_project_root(self, file_path: str) -> Optional[str]:
        """Find project root by looking for marker files."""
        markers = [
            ".git",
            "pyproject.toml",
            "setup.py",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
            ".project",
            "Makefile"
        ]

        try:
            path = Path(file_path).resolve()
            current = path.parent

            while current != current.parent:
                for marker in markers:
                    if (current / marker).exists():
                        return str(current)
                current = current.parent

        except (ValueError, OSError):
            pass

        return None

    def get_project_name(self, project_path: Optional[str]) -> Optional[str]:
        """Get a human-readable project name."""
        if not project_path:
            return None
        return Path(project_path).name

    def is_same_project(self, path1: str, path2: str) -> bool:
        """Check if two files are in the same project."""
        proj1 = self.detect_project(path1)
        proj2 = self.detect_project(path2)

        if proj1 and proj2:
            return proj1 == proj2

        return False


def create_smart_injector(
    token_tier: TokenBudgetTier = TokenBudgetTier.STANDARD,
    summary_enabled: bool = True
) -> SmartContextInjector:
    """
    Create a smart context injector with common configurations.

    Args:
        token_tier: Token budget tier to use
        summary_enabled: Whether to enable summarization

    Returns:
        Configured SmartContextInjector
    """
    config = SmartInjectionConfig(
        max_tokens=token_tier.value,
        token_tier=token_tier,
        summary_enabled=summary_enabled
    )
    return SmartContextInjector(config)


def quick_inject(
    raw_results: List[Dict[str, Any]],
    current_file: Optional[str] = None,
    max_tokens: int = 2000
) -> str:
    """
    Convenience function for quick context injection.

    Args:
        raw_results: Search results from AfterImage
        current_file: Current file being edited
        max_tokens: Maximum tokens for injection

    Returns:
        Formatted injection string
    """
    config = SmartInjectionConfig(max_tokens=max_tokens)
    injector = SmartContextInjector(config)

    if current_file:
        injector.set_context(current_file=current_file)

    result = injector.inject(raw_results)
    return result.injection_text
