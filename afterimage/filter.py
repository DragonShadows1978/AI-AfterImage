"""
Code Filter: Determines if a file is "code" vs artifacts.

Uses extension whitelists/blacklists and path patterns to decide
whether a file should be stored in the knowledge base.
"""

import os
import re
from pathlib import Path
from typing import List, Set, Optional
import yaml


# Default code extensions (whitelist)
DEFAULT_CODE_EXTENSIONS: Set[str] = {
    ".py", ".pyw",          # Python
    ".js", ".mjs", ".cjs",  # JavaScript
    ".ts", ".tsx", ".jsx",  # TypeScript/React
    ".rs",                  # Rust
    ".go",                  # Go
    ".java",                # Java
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx",  # C/C++
    ".cs",                  # C#
    ".rb", ".rake",         # Ruby
    ".php",                 # PHP
    ".swift",               # Swift
    ".kt", ".kts",          # Kotlin
    ".scala",               # Scala
    ".clj", ".cljs",        # Clojure
    ".ex", ".exs",          # Elixir
    ".erl", ".hrl",         # Erlang
    ".hs", ".lhs",          # Haskell
    ".ml", ".mli",          # OCaml
    ".fs", ".fsx", ".fsi",  # F#
    ".pl", ".pm",           # Perl
    ".lua",                 # Lua
    ".r", ".R",             # R
    ".jl",                  # Julia
    ".nim",                 # Nim
    ".zig",                 # Zig
    ".v",                   # V
    ".d",                   # D
    ".dart",                # Dart
    ".vue",                 # Vue
    ".svelte",              # Svelte
    ".elm",                 # Elm
    ".sol",                 # Solidity
    ".sql",                 # SQL
    ".sh", ".bash", ".zsh", # Shell
    ".ps1", ".psm1",        # PowerShell
}

# Default skip extensions (blacklist)
DEFAULT_SKIP_EXTENSIONS: Set[str] = {
    ".md", ".markdown", ".rst", ".txt",  # Documentation
    ".json", ".yaml", ".yml", ".toml",   # Config (often not "code")
    ".xml", ".html", ".htm",             # Markup
    ".css", ".scss", ".sass", ".less",   # Styles
    ".log", ".out",                       # Logs
    ".env", ".env.local", ".env.example", # Environment
    ".lock", ".sum",                      # Lock files
    ".min.js", ".min.css",               # Minified
    ".map",                               # Source maps
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",  # Images
    ".woff", ".woff2", ".ttf", ".eot",   # Fonts
    ".pdf", ".doc", ".docx",             # Documents
    ".csv", ".tsv",                       # Data
}

# Default skip paths (contains any of these)
DEFAULT_SKIP_PATHS: List[str] = [
    "artifacts/",
    "docs/",
    "documentation/",
    "research/",
    "test_data/",
    "__pycache__/",
    ".git/",
    ".venv/",
    "venv/",
    "node_modules/",
    ".mypy_cache/",
    ".pytest_cache/",
    "dist/",
    "build/",
    ".egg-info/",
    "migrations/",  # Database migrations are usually generated
]


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


class CodeFilter:
    """
    Filter to determine if a file path represents code that should
    be stored in the knowledge base.

    Supports:
    - Extension whitelist (code extensions)
    - Extension blacklist (skip extensions)
    - Path pattern blacklist (skip paths)
    - Content heuristics for unknown extensions
    """

    def __init__(
        self,
        code_extensions: Optional[Set[str]] = None,
        skip_extensions: Optional[Set[str]] = None,
        skip_paths: Optional[List[str]] = None,
        load_config_file: bool = True
    ):
        """
        Initialize the code filter.

        Args:
            code_extensions: Set of extensions to consider as code
            skip_extensions: Set of extensions to skip
            skip_paths: List of path patterns to skip
            load_config_file: Whether to load ~/.afterimage/config.yaml
        """
        # Load config file first
        config = {}
        if load_config_file:
            config = load_config()

        filter_config = config.get("filter", {})

        # Code extensions: use provided, then config, then default
        if code_extensions is not None:
            self.code_extensions = code_extensions
        elif "code_extensions" in filter_config:
            self.code_extensions = set(filter_config["code_extensions"])
        else:
            self.code_extensions = DEFAULT_CODE_EXTENSIONS.copy()

        # Skip extensions: use provided, then config, then default
        if skip_extensions is not None:
            self.skip_extensions = skip_extensions
        elif "skip_extensions" in filter_config:
            self.skip_extensions = set(filter_config["skip_extensions"])
        else:
            self.skip_extensions = DEFAULT_SKIP_EXTENSIONS.copy()

        # Skip paths: use provided, then config, then default
        if skip_paths is not None:
            self.skip_paths = skip_paths
        elif "skip_paths" in filter_config:
            self.skip_paths = filter_config["skip_paths"]
        else:
            self.skip_paths = DEFAULT_SKIP_PATHS.copy()

    def is_code(self, file_path: str, content: Optional[str] = None) -> bool:
        """
        Determine if a file path represents code.

        Args:
            file_path: Path to the file
            content: Optional file content for heuristic analysis

        Returns:
            True if the file should be considered code
        """
        path = Path(file_path)
        name = path.name

        # Check skip paths first
        path_str = str(file_path)
        for skip_pattern in self.skip_paths:
            if skip_pattern in path_str:
                return False

        # Check for minified files before other extension checks
        if ".min." in name:
            return False

        # Get extension (handle multiple dots like .test.js)
        ext = self._get_extension(path)

        # Check explicit skip list - but allow content heuristics to override for unknown ext
        if ext in self.skip_extensions:
            # If content is provided and ext is a "soft skip" (like .txt), use heuristics
            if content is not None and ext in {".txt", ".unknown"}:
                return self._content_heuristics(content)
            return False

        # Check explicit code list
        if ext in self.code_extensions:
            return True

        # Unknown extension - use heuristics if content provided
        if content is not None:
            return self._content_heuristics(content)

        # Unknown extension and no content - skip to be safe
        return False

    def _get_extension(self, path: Path) -> str:
        """Get file extension, handling special cases."""
        name = path.name

        # Handle no extension
        if "." not in name:
            return ""

        # Handle dotfiles (like .gitignore)
        if name.startswith(".") and name.count(".") == 1:
            return name  # Return the whole name as "extension"

        # Handle compound extensions like .test.js
        parts = name.split(".")
        if len(parts) >= 2:
            # Check if the last two parts form a known pattern
            compound = f".{parts[-2]}.{parts[-1]}"
            if compound in {".test.js", ".test.ts", ".spec.js", ".spec.ts",
                           ".test.py", ".spec.py", ".stories.js", ".stories.tsx"}:
                return f".{parts[-1]}"  # Return just the last part

        return path.suffix.lower()

    def _content_heuristics(self, content: str) -> bool:
        """
        Use content heuristics to determine if text is code.

        Returns True if the content appears to be code.
        """
        # Empty or very short content - not useful
        if len(content.strip()) < 20:
            return False

        # Count code indicators
        code_indicators = 0

        # Function/method definitions
        if re.search(r'\bdef\s+\w+\s*\(', content):  # Python
            code_indicators += 2
        if re.search(r'\bfunction\s+\w*\s*\(', content):  # JS
            code_indicators += 2
        if re.search(r'\bfn\s+\w+\s*\(', content):  # Rust
            code_indicators += 2
        if re.search(r'\bfunc\s+\w+\s*\(', content):  # Go
            code_indicators += 2

        # Class definitions
        if re.search(r'\bclass\s+\w+', content):
            code_indicators += 2

        # Import statements
        if re.search(r'\b(import|from|require|use|include)\b', content):
            code_indicators += 1

        # Common programming constructs
        if re.search(r'\b(if|else|for|while|return|try|catch)\b', content):
            code_indicators += 1

        # Variable assignments with types or keywords
        if re.search(r'\b(const|let|var|val)\s+\w+\s*=', content):
            code_indicators += 1

        # Brackets and braces (common in code)
        bracket_ratio = (content.count('{') + content.count('}') +
                        content.count('[') + content.count(']')) / max(len(content), 1)
        if bracket_ratio > 0.02:
            code_indicators += 1

        # Semicolons at end of lines (common in many languages)
        if re.search(r';\s*$', content, re.MULTILINE):
            code_indicators += 1

        # Arrow functions or lambdas
        if '=>' in content or 'lambda' in content:
            code_indicators += 1

        return code_indicators >= 2

    def add_code_extension(self, ext: str):
        """Add an extension to the code list."""
        if not ext.startswith("."):
            ext = "." + ext
        self.code_extensions.add(ext.lower())

    def add_skip_extension(self, ext: str):
        """Add an extension to the skip list."""
        if not ext.startswith("."):
            ext = "." + ext
        self.skip_extensions.add(ext.lower())

    def add_skip_path(self, path: str):
        """Add a path pattern to skip."""
        self.skip_paths.append(path)

    def get_config(self) -> dict:
        """Get current configuration as a dictionary."""
        return {
            "code_extensions": sorted(self.code_extensions),
            "skip_extensions": sorted(self.skip_extensions),
            "skip_paths": self.skip_paths,
        }
