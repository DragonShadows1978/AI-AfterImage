"""
Context Injection: Format retrieved code for injection into Claude's context.

Creates the "You have written similar code before..." messages that
help Claude maintain consistency across sessions.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import yaml

from .search import SearchResult


def load_config() -> dict:
    """Load configuration from ~/.afterimage/config.yaml if it exists."""
    config_path = Path.home() / ".afterimage" / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


@dataclass
class InjectionConfig:
    """Configuration for context injection."""
    max_results: int = 5
    max_tokens: int = 2000  # Approximate token limit for injection
    relevance_threshold: float = 0.4
    include_context: bool = True
    include_file_path: bool = True
    include_timestamp: bool = False
    code_fence_language: str = ""  # Auto-detect if empty


class ContextInjector:
    """
    Formats search results for injection into Claude's context.

    Creates structured messages that inform Claude about previously
    written similar code, helping maintain consistency.
    """

    def __init__(self, config: Optional[InjectionConfig] = None):
        """
        Initialize the injector.

        Args:
            config: Injection configuration (uses defaults if None)
        """
        if config is None:
            # Try to load from config file
            file_config = load_config().get("search", {})
            config = InjectionConfig(
                max_results=file_config.get("max_results", 5),
                max_tokens=file_config.get("max_injection_tokens", 2000),
                relevance_threshold=file_config.get("relevance_threshold", 0.4),
            )
        self.config = config

    def format_injection(
        self,
        results: List[SearchResult],
        query_context: Optional[str] = None
    ) -> Optional[str]:
        """
        Format search results as an injection message.

        Args:
            results: Search results to format
            query_context: Optional context about what triggered the search

        Returns:
            Formatted injection string, or None if no relevant results
        """
        # Filter by relevance
        relevant = [
            r for r in results
            if r.relevance_score >= self.config.relevance_threshold
        ]

        if not relevant:
            return None

        # Limit results
        relevant = relevant[:self.config.max_results]

        # Build injection message
        parts = []

        # Header
        if len(relevant) == 1:
            parts.append("You have written similar code before:\n")
        else:
            parts.append(f"You have written similar code before ({len(relevant)} matches):\n")

        # Track approximate token count
        current_tokens = self._estimate_tokens(parts[0])

        # Add each result
        for i, result in enumerate(relevant, 1):
            section = self._format_result(result, i, len(relevant) > 1)

            # Check token budget
            section_tokens = self._estimate_tokens(section)
            if current_tokens + section_tokens > self.config.max_tokens:
                # Add truncation notice and stop
                parts.append(f"\n... ({len(relevant) - i + 1} more matches truncated)")
                break

            parts.append(section)
            current_tokens += section_tokens

        return "\n".join(parts)

    def format_single(self, result: SearchResult) -> str:
        """
        Format a single result for display.

        Args:
            result: Search result to format

        Returns:
            Formatted string
        """
        return self._format_result(result, numbered=False)

    def format_for_hook(
        self,
        results: List[SearchResult],
        file_path: str,
        tool_type: str
    ) -> Optional[str]:
        """
        Format results specifically for Claude Code hook injection.

        Args:
            results: Search results
            file_path: Path of file being written
            tool_type: "Write" or "Edit"

        Returns:
            Hook-formatted injection string
        """
        relevant = [
            r for r in results
            if r.relevance_score >= self.config.relevance_threshold
        ]

        if not relevant:
            return None

        # For hooks, we want a concise format
        parts = []

        # Contextual header
        action = "creating" if tool_type == "Write" else "editing"
        file_name = Path(file_path).name
        parts.append(f"<memory context=\"You are {action} {file_name}\">\n")
        parts.append("Previously written similar code:\n")

        # Add relevant code with minimal decoration
        for result in relevant[:self.config.max_results]:
            parts.append(self._format_compact(result))

        parts.append("</memory>")

        return "\n".join(parts)

    def _format_result(
        self,
        result: SearchResult,
        index: Optional[int] = None,
        numbered: bool = True
    ) -> str:
        """Format a single search result."""
        parts = []

        # Header line
        header_parts = []
        if numbered and index:
            header_parts.append(f"### Match {index}")
        else:
            header_parts.append("### Previous Code")

        if self.config.include_file_path:
            # Show only the relevant part of the path
            path = Path(result.file_path)
            short_path = "/".join(path.parts[-3:]) if len(path.parts) > 3 else str(path)
            header_parts.append(f"({short_path})")

        if self.config.include_timestamp:
            header_parts.append(f"[{result.timestamp[:10]}]")

        parts.append(" ".join(header_parts))

        # Code block
        lang = self._detect_language(result.file_path)
        code = self._truncate_code(result.new_code)
        parts.append(f"\n```{lang}\n{code}\n```")

        # Context if available and configured
        if self.config.include_context and result.context:
            context = self._truncate_context(result.context)
            parts.append(f"\n**Context:** {context}")

        # Relevance indicator
        if result.relevance_score > 0:
            relevance_percent = int(result.relevance_score * 100)
            parts.append(f"\n*Relevance: {relevance_percent}%*")

        return "\n".join(parts)

    def _format_compact(self, result: SearchResult) -> str:
        """Format a result in compact form for hooks."""
        path = Path(result.file_path)
        short_path = path.name

        lang = self._detect_language(result.file_path)
        code = self._truncate_code(result.new_code, max_lines=20)

        parts = [f"--- {short_path} ---"]
        parts.append(f"```{lang}\n{code}\n```")

        if result.context:
            context = result.context[:100]
            if len(result.context) > 100:
                context += "..."
            parts.append(f"Why: {context}")

        parts.append("")
        return "\n".join(parts)

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension for code fencing."""
        if self.config.code_fence_language:
            return self.config.code_fence_language

        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".jsx": "jsx",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".sh": "bash",
            ".sql": "sql",
            ".vue": "vue",
            ".svelte": "svelte",
        }

        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, "")

    def _truncate_code(self, code: str, max_lines: int = 30) -> str:
        """Truncate code to reasonable length."""
        lines = code.split("\n")

        if len(lines) <= max_lines:
            return code

        # Keep first and last parts
        head = lines[:max_lines // 2]
        tail = lines[-(max_lines // 2):]

        return "\n".join(head) + "\n# ... (truncated) ...\n" + "\n".join(tail)

    def _truncate_context(self, context: str, max_chars: int = 200) -> str:
        """Truncate context to reasonable length."""
        if len(context) <= max_chars:
            return context

        return context[:max_chars] + "..."

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)."""
        return len(text) // 4


def create_injection_message(
    results: List[SearchResult],
    config: Optional[InjectionConfig] = None
) -> Optional[str]:
    """
    Convenience function to create an injection message.

    Args:
        results: Search results to format
        config: Optional configuration

    Returns:
        Formatted injection string or None
    """
    injector = ContextInjector(config)
    return injector.format_injection(results)
