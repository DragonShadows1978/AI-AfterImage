"""
Snippet Summarizer: Condense multiple similar snippets into representative examples.

When multiple similar code snippets are found, this module:
1. Groups them by similarity
2. Selects the most representative example from each group
3. Generates a summary that captures the pattern without repetition
4. Implements summary mode for large groups (3+ similar snippets)

Part of AfterImage Semantic Chunking v0.3.0.
"""

import re
from typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from .relevance_scorer import ScoredSnippet


@dataclass
class SnippetGroup:
    """A group of similar snippets."""
    representative: ScoredSnippet
    members: List[ScoredSnippet] = field(default_factory=list)
    similarity_threshold: float = 0.0
    pattern_summary: Optional[str] = None
    label: str = "code snippets"

    @property
    def size(self) -> int:
        return len(self.members) + 1  # Include representative

    @property
    def similarity(self) -> float:
        """Average similarity of members to representative."""
        if not self.members:
            return 1.0
        return self.similarity_threshold

    @property
    def total_tokens(self) -> int:
        """Estimate total tokens if all members were included."""
        return sum(len(m.code) // 4 for m in self.members) + len(self.representative.code) // 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "representative": self.representative.to_dict(),
            "member_count": len(self.members),
            "pattern_summary": self.pattern_summary,
            "similarity_threshold": self.similarity_threshold,
            "label": self.label,
        }


@dataclass
class SummaryConfig:
    """Configuration for snippet summarization."""
    # Similarity threshold for grouping (0-1)
    similarity_threshold: float = 0.7

    # Minimum group size to trigger summarization
    min_group_size: int = 2

    # Minimum group size for summary mode (show count instead of all)
    summary_mode_threshold: int = 3

    # Maximum snippets to show individually before summarizing
    max_individual_snippets: int = 3

    # Whether to include member count in summary
    show_member_count: bool = True

    # Whether to generate pattern descriptions
    generate_pattern_summary: bool = True

    # Whether to show file locations in group summaries
    show_group_locations: bool = True


class SnippetSummarizer:
    """
    Groups and summarizes similar code snippets.

    Reduces token usage by showing representative examples instead of
    many similar snippets.
    """

    def __init__(self, config: Optional[SummaryConfig] = None):
        """
        Initialize the summarizer.

        Args:
            config: Summarization configuration (uses defaults if None)
        """
        self.config = config or SummaryConfig()

    def summarize(
        self,
        snippets: List[ScoredSnippet],
        max_output: int = 5
    ) -> Tuple[List[ScoredSnippet], List[SnippetGroup]]:
        """
        Summarize a list of snippets by grouping similar ones.

        Args:
            snippets: List of scored snippets to summarize
            max_output: Maximum number of output items (snippets + groups)

        Returns:
            Tuple of (individual_snippets, grouped_snippets)
        """
        if len(snippets) <= self.config.max_individual_snippets:
            return snippets[:max_output], []

        groups = self._group_similar(snippets)

        individual = []
        summarized = []

        for group in groups:
            if group.size >= self.config.min_group_size:
                if self.config.generate_pattern_summary:
                    group.pattern_summary = self._generate_pattern_summary(group)
                group.label = self._get_group_label(group.members + [group.representative])
                summarized.append(group)
            else:
                individual.append(group.representative)

        output_individual = individual[:max_output]
        remaining_slots = max_output - len(output_individual)
        output_groups = summarized[:remaining_slots]

        return output_individual, output_groups

    def _group_similar(self, snippets: List[ScoredSnippet]) -> List[SnippetGroup]:
        """Group snippets by similarity using greedy clustering."""
        if not snippets:
            return []

        groups: List[SnippetGroup] = []
        assigned: Set[int] = set()

        sorted_snippets = sorted(snippets, key=lambda s: s.relevance_score, reverse=True)

        for i, snippet in enumerate(sorted_snippets):
            if i in assigned:
                continue

            group = SnippetGroup(
                representative=snippet,
                similarity_threshold=self.config.similarity_threshold
            )
            assigned.add(i)

            for j, other in enumerate(sorted_snippets):
                if j in assigned:
                    continue

                similarity = self._compute_similarity(snippet, other)
                if similarity >= self.config.similarity_threshold:
                    group.members.append(other)
                    assigned.add(j)

            groups.append(group)

        return groups

    def _compute_similarity(
        self,
        snippet1: ScoredSnippet,
        snippet2: ScoredSnippet
    ) -> float:
        """Compute similarity between two snippets."""
        scores = []

        code_sim = self._code_similarity(snippet1.code, snippet2.code)
        scores.append(code_sim * 0.7)

        path_sim = self._path_similarity(snippet1.file_path, snippet2.file_path)
        scores.append(path_sim * 0.2)

        type_sim = 1.0 if snippet1.chunk_type == snippet2.chunk_type else 0.5
        scores.append(type_sim * 0.1)

        return sum(scores)

    def _code_similarity(self, code1: str, code2: str) -> float:
        """Compute code similarity using structure-aware comparison."""
        norm1 = self._normalize_code(code1)
        norm2 = self._normalize_code(code2)

        matcher = SequenceMatcher(None, norm1, norm2)
        return matcher.ratio()

    def _normalize_code(self, code: str) -> str:
        """Normalize code for similarity comparison."""
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        code = re.sub(r'"[^"]*"', '"STR"', code)
        code = re.sub(r"'[^']*'", "'STR'", code)

        code = re.sub(r'\b\d+\b', 'NUM', code)
        code = re.sub(r'\s+', ' ', code)
        code = code.lower().strip()

        return code

    def _path_similarity(self, path1: str, path2: str) -> float:
        """Compute similarity between file paths."""
        parts1 = path1.split('/')
        parts2 = path2.split('/')

        shared = 0
        for p1, p2 in zip(reversed(parts1), reversed(parts2)):
            if p1 == p2:
                shared += 1
            else:
                break

        max_parts = max(len(parts1), len(parts2))
        if max_parts == 0:
            return 0.0

        return shared / max_parts

    def _generate_pattern_summary(self, group: SnippetGroup) -> str:
        """Generate a human-readable summary of the pattern in a group."""
        rep = group.representative
        members = group.members

        all_snippets = [rep] + members
        chunk_types = set(s.chunk_type for s in all_snippets if s.chunk_type)

        file_patterns = self._find_common_path_patterns(
            [s.file_path for s in all_snippets]
        )

        parts = []

        if chunk_types:
            type_str = "/".join(sorted(chunk_types))
            parts.append(f"Pattern type: {type_str}")

        if file_patterns:
            parts.append(f"Found in: {', '.join(file_patterns)}")

        parts.append(f"Similar occurrences: {group.size}")

        if rep.chunk_name:
            parts.append(f"Example: {rep.chunk_name}")

        return " | ".join(parts)

    def _find_common_path_patterns(self, paths: List[str]) -> List[str]:
        """Find common patterns in a list of file paths."""
        if not paths:
            return []

        dirs = defaultdict(int)
        for path in paths:
            parts = path.split('/')
            for i, part in enumerate(parts[:-1]):
                dirs[part] += 1

        threshold = len(paths) * 0.5
        common = [d for d, count in dirs.items() if count >= threshold and d]

        return common[:3]

    def _get_group_label(self, snippets: List[ScoredSnippet]) -> str:
        """Get descriptive label for snippet group."""
        chunk_types = [s.chunk_type for s in snippets if s.chunk_type]
        if chunk_types:
            most_common = max(set(chunk_types), key=chunk_types.count)
            return {
                "function": "functions",
                "class": "classes",
                "method": "methods",
                "imports": "import blocks",
                "constants": "constant blocks",
                "block": "code blocks",
            }.get(most_common, "code snippets")
        return "code snippets"


class SummaryFormatter:
    """Formats summarized snippets for injection."""

    def __init__(self, config: Optional[SummaryConfig] = None):
        self.config = config or SummaryConfig()

    def format_for_injection(
        self,
        individual: List[ScoredSnippet],
        groups: List[SnippetGroup],
        max_tokens: int = 2000
    ) -> str:
        """
        Format summarized snippets as an injection message.

        Uses summary mode for groups with 3+ similar snippets.

        Args:
            individual: Individual snippets to show in full
            groups: Grouped snippets to summarize
            max_tokens: Maximum tokens for output

        Returns:
            Formatted injection string
        """
        parts = []
        token_count = 0
        tokens_per_char = 0.25

        # Add individual snippets
        for snippet in individual:
            if token_count >= max_tokens:
                break

            snippet_text = self._format_snippet(snippet)
            snippet_tokens = int(len(snippet_text) * tokens_per_char)

            if token_count + snippet_tokens <= max_tokens:
                parts.append(snippet_text)
                token_count += snippet_tokens

        # Add group summaries
        for group in groups:
            if token_count >= max_tokens:
                break

            # Use summary mode for large groups
            if group.size >= self.config.summary_mode_threshold:
                group_text = self._format_summary(group)
            else:
                group_text = self._format_group(group)

            group_tokens = int(len(group_text) * tokens_per_char)

            if token_count + group_tokens <= max_tokens:
                parts.append(group_text)
                token_count += group_tokens

        return "\n\n".join(parts)

    def _format_snippet(self, snippet: ScoredSnippet) -> str:
        """Format a single snippet."""
        short_path = "/".join(Path(snippet.file_path).parts[-3:])

        lines = [f"### {short_path}"]

        if snippet.chunk_name and snippet.chunk_type:
            lines.append(f"*{snippet.chunk_type}: {snippet.chunk_name}*")

        lines.append(f"```\n{snippet.code}\n```")
        lines.append(f"*Relevance: {int(snippet.relevance_score * 100)}%*")

        return "\n".join(lines)

    def _format_group(self, group: SnippetGroup) -> str:
        """Format a snippet group (shows all members)."""
        rep = group.representative
        short_path = "/".join(Path(rep.file_path).parts[-3:])

        lines = [f"### Pattern ({group.size} similar {group.label})"]

        if group.pattern_summary:
            lines.append(f"*{group.pattern_summary}*")

        lines.append(f"\n**Representative example** ({short_path}):")
        lines.append(f"```\n{rep.code}\n```")

        if group.members and self.config.show_group_locations:
            other_files = ["/".join(Path(m.file_path).parts[-2:]) for m in group.members[:3]]
            lines.append(f"\n*Also in: {', '.join(other_files)}*")

        return "\n".join(lines)

    def _format_summary(self, group: SnippetGroup) -> str:
        """Format a group as summary with representative sample (summary mode)."""
        rep = group.representative
        short_path = "/".join(Path(rep.file_path).parts[-3:])

        # Get unique file locations
        unique_files = set(
            "/".join(Path(s.file_path).parts[-2:])
            for s in [rep] + group.members[:3]
        )

        lines = [f"**{group.size} similar {group.label} found**"]

        if group.pattern_summary:
            lines.append(f"*{group.pattern_summary}*")

        lines.append(f"\nRepresentative ({int(group.similarity * 100)}% similarity):")
        lines.append(f"```\n{rep.code[:300]}")
        if len(rep.code) > 300:
            lines.append("... (truncated)")
        lines.append("```")

        if self.config.show_group_locations and unique_files:
            lines.append(f"\n[+{group.size - 1} similar in: {', '.join(list(unique_files)[:3])}...]")

        return "\n".join(lines)


def summarize_snippets(
    snippets: List[ScoredSnippet],
    max_output: int = 5,
    similarity_threshold: float = 0.7
) -> Tuple[List[ScoredSnippet], List[SnippetGroup]]:
    """
    Convenience function to summarize snippets.

    Args:
        snippets: Scored snippets to summarize
        max_output: Maximum output items
        similarity_threshold: Threshold for grouping

    Returns:
        Tuple of (individual_snippets, grouped_snippets)
    """
    config = SummaryConfig(similarity_threshold=similarity_threshold)
    summarizer = SnippetSummarizer(config)
    return summarizer.summarize(snippets, max_output)
