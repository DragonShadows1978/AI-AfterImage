"""
Token Budget Manager: Token-aware context injection for Claude.

Manages the token budget for AfterImage context injection, ensuring
injected code doesn't exceed Claude's context window limits while
maximizing the usefulness of the injected context.

Part of AfterImage Semantic Chunking v0.3.0.
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class TokenBudgetTier(Enum):
    """
    Predefined token budget tiers based on Claude model context windows.

    These define how much of the context window should be reserved for
    AfterImage memory injection.
    """
    MINIMAL = 500       # Very conservative, just hints
    COMPACT = 1000      # Brief snippets only
    STANDARD = 2000     # Default balance
    GENEROUS = 4000     # Rich context
    EXTENSIVE = 8000    # Maximum detail


@dataclass
class TokenBudgetConfig:
    """Configuration for token budget management."""
    # Total tokens available for injection
    max_tokens: int = 2000

    # Reserved tokens for overhead (headers, formatting)
    overhead_tokens: int = 100

    # Minimum tokens per code snippet (don't truncate below this)
    min_snippet_tokens: int = 50

    # Maximum tokens per single code snippet
    max_snippet_tokens: int = 500

    # Tokens reserved for summary when condensing
    summary_tokens: int = 200

    # Whether to prefer fewer complete snippets vs more truncated ones
    prefer_complete: bool = True


class TokenEstimator:
    """
    Estimates token counts for text content.

    Uses character-based heuristics calibrated for Claude's tokenizer.
    """

    CODE_RATIO = 4.0
    PROSE_RATIO = 5.0
    MIXED_RATIO = 4.5

    def __init__(self, default_ratio: float = 4.0):
        """
        Initialize the estimator.

        Args:
            default_ratio: Default characters per token ratio
        """
        self.default_ratio = default_ratio
        self._cache: Dict[int, int] = {}

    def estimate(self, text: str, content_type: str = "code") -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate tokens for
            content_type: Type of content ("code", "prose", "mixed")

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        text_hash = hash((text, content_type))
        if text_hash in self._cache:
            return self._cache[text_hash]

        if content_type == "code":
            ratio = self.CODE_RATIO
        elif content_type == "prose":
            ratio = self.PROSE_RATIO
        else:
            ratio = self.MIXED_RATIO

        base_estimate = len(text) / ratio
        estimate = int(base_estimate * 1.1) + 1

        self._cache[text_hash] = estimate
        return estimate

    def estimate_code_block(self, code: str, with_fence: bool = True) -> int:
        """
        Estimate tokens for a code block with optional markdown fencing.

        Args:
            code: Code content
            with_fence: Include markdown code fence tokens

        Returns:
            Estimated token count
        """
        code_tokens = self.estimate(code, "code")
        if with_fence:
            code_tokens += 4
        return code_tokens

    def estimate_injection(
        self,
        header: str,
        snippets: List[str],
        footer: str = ""
    ) -> int:
        """
        Estimate total tokens for a complete injection.

        Args:
            header: Header text
            snippets: List of code snippets
            footer: Optional footer text

        Returns:
            Total estimated tokens
        """
        total = self.estimate(header, "prose")
        total += self.estimate(footer, "prose")

        for snippet in snippets:
            total += self.estimate_code_block(snippet)

        return total


class TokenBudgetManager:
    """
    Manages token budget for context injection.

    Ensures injected context fits within the allocated budget while
    maximizing usefulness.
    """

    def __init__(self, config: Optional[TokenBudgetConfig] = None):
        """
        Initialize the budget manager.

        Args:
            config: Budget configuration (uses defaults if None)
        """
        self.config = config or TokenBudgetConfig()
        self.estimator = TokenEstimator()

    @property
    def available_tokens(self) -> int:
        """Get tokens available for content (after overhead)."""
        return self.config.max_tokens - self.config.overhead_tokens

    def allocate_for_snippets(
        self,
        snippets: List[Tuple[str, float]],
        preserve_order: bool = False
    ) -> List[Tuple[str, int]]:
        """
        Allocate tokens across multiple snippets.

        Args:
            snippets: List of (code, relevance_score) tuples, sorted by relevance
            preserve_order: If True, maintain input order

        Returns:
            List of (code, allocated_tokens) tuples that fit in budget
        """
        if not snippets:
            return []

        available = self.available_tokens
        allocated: List[Tuple[str, int]] = []

        estimates = [
            (code, score, self.estimator.estimate_code_block(code))
            for code, score in snippets
        ]

        if self.config.prefer_complete:
            allocated = self._allocate_complete(estimates, available)
        else:
            allocated = self._allocate_with_truncation(estimates, available)

        return allocated

    def _allocate_complete(
        self,
        estimates: List[Tuple[str, float, int]],
        available: int
    ) -> List[Tuple[str, int]]:
        """Allocate complete snippets only."""
        allocated = []
        remaining = available

        for code, score, tokens in estimates:
            if tokens <= remaining and tokens <= self.config.max_snippet_tokens:
                allocated.append((code, tokens))
                remaining -= tokens
            elif remaining < self.config.min_snippet_tokens:
                break

        return allocated

    def _allocate_with_truncation(
        self,
        estimates: List[Tuple[str, float, int]],
        available: int
    ) -> List[Tuple[str, int]]:
        """Allocate snippets, truncating if needed."""
        allocated = []
        remaining = available

        for code, score, tokens in estimates:
            if remaining < self.config.min_snippet_tokens:
                break

            if tokens <= remaining:
                allocated.append((code, tokens))
                remaining -= tokens
            elif tokens > self.config.max_snippet_tokens:
                truncated = self._truncate_code(
                    code,
                    min(remaining, self.config.max_snippet_tokens)
                )
                if truncated:
                    truncated_tokens = self.estimator.estimate_code_block(truncated)
                    allocated.append((truncated, truncated_tokens))
                    remaining -= truncated_tokens
            else:
                truncated = self._truncate_code(code, remaining)
                if truncated:
                    truncated_tokens = self.estimator.estimate_code_block(truncated)
                    allocated.append((truncated, truncated_tokens))
                    remaining -= truncated_tokens

        return allocated

    def _truncate_code(self, code: str, max_tokens: int) -> Optional[str]:
        """
        Truncate code to fit within token limit.

        Args:
            code: Code to truncate
            max_tokens: Maximum tokens for result

        Returns:
            Truncated code, or None if can't fit min_snippet_tokens
        """
        if max_tokens < self.config.min_snippet_tokens:
            return None

        lines = code.split("\n")

        low, high = 1, len(lines)
        best_code = None

        while low <= high:
            mid = (low + high) // 2
            candidate = "\n".join(lines[:mid])
            tokens = self.estimator.estimate_code_block(candidate)

            if tokens <= max_tokens:
                best_code = candidate
                low = mid + 1
            else:
                high = mid - 1

        if best_code:
            if len(best_code.split("\n")) < len(lines):
                best_code += "\n# ... (truncated)"

        return best_code

    def create_injection(
        self,
        snippets: List[Tuple[str, str, float]],
        query_context: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Create a complete injection message within budget.

        Args:
            snippets: List of (code, file_path, relevance_score) tuples
            query_context: Optional context about the query

        Returns:
            (injection_text, tokens_used) tuple
        """
        if not snippets:
            return "", 0

        if len(snippets) == 1:
            header = "You have written similar code before:\n\n"
        else:
            header = f"You have written similar code before ({len(snippets)} matches):\n\n"

        header_tokens = self.estimator.estimate(header, "prose")
        available = self.available_tokens - header_tokens

        code_scores = [(code, score) for code, _, score in snippets]
        allocated = self.allocate_for_snippets(code_scores)

        if not allocated:
            return "", 0

        parts = [header]
        total_tokens = header_tokens

        for i, (code, tokens) in enumerate(allocated):
            original = next(
                (s for s in snippets if s[0] == code or code.startswith(s[0][:100])),
                (code, "unknown", 0.0)
            )
            _, file_path, score = original

            short_path = "/".join(Path(file_path).parts[-3:]) if file_path != "unknown" else "unknown"

            parts.append(f"### Match {i + 1} ({short_path})")
            parts.append(f"```\n{code}\n```")
            parts.append(f"*Relevance: {int(score * 100)}%*\n")

            total_tokens += tokens + 20

        injection = "\n".join(parts)
        return injection, total_tokens

    def fits_in_budget(self, text: str) -> bool:
        """Check if text fits in the available budget."""
        tokens = self.estimator.estimate(text, "mixed")
        return tokens <= self.available_tokens

    def remaining_budget(self, used_tokens: int) -> int:
        """Get remaining tokens after some have been used."""
        return max(0, self.available_tokens - used_tokens)


def create_token_budget(tier: TokenBudgetTier = TokenBudgetTier.STANDARD) -> TokenBudgetManager:
    """
    Create a token budget manager with a predefined tier.

    Args:
        tier: Budget tier to use

    Returns:
        Configured TokenBudgetManager
    """
    config = TokenBudgetConfig(max_tokens=tier.value)
    return TokenBudgetManager(config)
