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


# Extensionless files that are commonly code (by exact filename)
EXTENSIONLESS_CODE_FILES: Set[str] = {
    # Build files
    "Makefile", "GNUmakefile", "BSDmakefile", "makefile",
    # Container/Infrastructure
    "Dockerfile", "Containerfile",
    # CI/CD
    "Jenkinsfile", "Vagrantfile",
    # Ruby
    "Rakefile", "Gemfile", "Guardfile", "Brewfile", "Berksfile",
    "Fastfile", "Appfile", "Deliverfile", "Matchfile", "Scanfile",
    # JavaScript/Node
    "Gruntfile", "Gulpfile",
    # Other
    "Procfile", "Justfile", "Cakefile", "Snakefile",
    "SConstruct", "SConscript", "Thorfile", "Puppetfile",
    "Capfile", "Buildfile", "Jarfile",
}


# Pre-compiled regex patterns for content heuristics (performance optimization)
_HEURISTIC_PATTERNS = {
    "python_def": re.compile(r'\bdef\s+\w+\s*\('),
    "js_function": re.compile(r'\bfunction\s+\w*\s*\('),
    "rust_fn": re.compile(r'\bfn\s+\w+\s*\('),
    "go_func": re.compile(r'\bfunc\s+\w+\s*\('),
    "perl_sub": re.compile(r'\bsub\s+\w+\s*[\(\{]'),
    "class": re.compile(r'\bclass\s+\w+'),
    "struct": re.compile(r'\bstruct\s+\w+'),
    "trait": re.compile(r'\btrait\s+\w+'),
    "interface": re.compile(r'\binterface\s+\w+'),
    "import": re.compile(r'\b(import|from|require|use|include)\b'),
    "control_flow": re.compile(r'\b(if|else|for|while|return|try|catch)\b'),
    "variable": re.compile(r'\b(const|let|var|val|mut)\s+\w+\s*[=:]'),
    "semicolon_eol": re.compile(r';\s*$', re.MULTILINE),
    "type_annotation": re.compile(r':\s*(str|int|bool|float|string|number|boolean|i32|u32|Vec)'),
    "decorator": re.compile(r'^@\w+', re.MULTILINE),
}

# Pre-compiled shebang patterns
_SHEBANG_PATTERNS = [
    re.compile(r'#!\s*/usr/bin/env\s+\w+'),
    re.compile(r'#!\s*/bin/(ba)?sh'),
    re.compile(r'#!\s*/usr/bin/python'),
    re.compile(r'#!\s*/usr/bin/perl'),
    re.compile(r'#!\s*/usr/bin/ruby'),
    re.compile(r'#!\s*/usr/bin/node'),
]


# Default code extensions (whitelist)
DEFAULT_CODE_EXTENSIONS: Set[str] = {
    # Python
    ".py", ".pyw", ".pyi",
    # JavaScript
    ".js", ".mjs", ".cjs",
    # TypeScript/React
    ".ts", ".tsx", ".jsx",
    # Rust
    ".rs",
    # Go
    ".go",
    # Java
    ".java",
    # C/C++
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".c++", ".h++",
    # C#
    ".cs",
    # Objective-C/C++
    ".m", ".mm",
    # Ruby
    ".rb", ".rake", ".gemspec",
    # PHP
    ".php", ".phtml",
    # Swift
    ".swift",
    # Kotlin
    ".kt", ".kts",
    # Scala
    ".scala", ".sc",
    # Clojure
    ".clj", ".cljs", ".cljc", ".edn",
    # Elixir
    ".ex", ".exs",
    # Erlang
    ".erl", ".hrl",
    # Haskell
    ".hs", ".lhs",
    # OCaml
    ".ml", ".mli",
    # F#
    ".fs", ".fsx", ".fsi",
    # Perl
    ".pl", ".pm", ".pod",
    # Lua
    ".lua",
    # R
    ".r", ".R",
    # Julia
    ".jl",
    # Nim
    ".nim", ".nims",
    # Zig
    ".zig",
    # V
    ".v",
    # D
    ".d",
    # Dart
    ".dart",
    # Vue
    ".vue",
    # Svelte
    ".svelte",
    # Elm
    ".elm",
    # Solidity
    ".sol",
    # SQL
    ".sql", ".pgsql", ".plsql", ".mysql",
    # Shell scripts
    ".sh", ".bash", ".zsh", ".fish", ".csh", ".tcsh", ".ksh",
    # PowerShell
    ".ps1", ".psm1", ".psd1",
    # Windows batch
    ".bat", ".cmd",
    # Assembly
    ".asm", ".s", ".S",
    # Groovy
    ".groovy", ".gradle", ".gvy",
    # Fortran
    ".f", ".f90", ".f95", ".f03", ".f08", ".for", ".ftn",
    # Visual Basic
    ".vb", ".vbs", ".bas",
    # CoffeeScript
    ".coffee", ".litcoffee",
    # Protocol Buffers / gRPC
    ".proto",
    # GraphQL
    ".graphql", ".gql",
    # Terraform / Infrastructure as Code
    ".tf", ".tfvars", ".hcl",
    # Azure Bicep
    ".bicep",
    # Nix
    ".nix",
    # PureScript
    ".purs",
    # Crystal
    ".cr",
    # Haxe
    ".hx",
    # WebAssembly
    ".wasm", ".wat",
    # Lisp family
    ".lisp", ".lsp", ".cl", ".scm", ".ss", ".rkt",
    # Ada
    ".ada", ".adb", ".ads",
    # COBOL
    ".cob", ".cbl",
    # Pascal/Delphi
    ".pas", ".dpr",
    # Tcl
    ".tcl",
    # AWK/sed
    ".awk", ".sed",
    # Hardware description
    ".vhd", ".vhdl", ".sv", ".svh",
    # ActionScript/Flex
    ".as", ".mxml",
    # Dhall
    ".dhall",
    # Makefile extensions (though Makefile has no ext typically)
    ".mk",
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
    - Extensionless code file detection (Makefile, Dockerfile, etc.)
    - Content heuristics for unknown extensions
    """

    def __init__(
        self,
        code_extensions: Optional[Set[str]] = None,
        skip_extensions: Optional[Set[str]] = None,
        skip_paths: Optional[List[str]] = None,
        extensionless_code_files: Optional[Set[str]] = None,
        load_config_file: bool = True
    ):
        """
        Initialize the code filter.

        Args:
            code_extensions: Set of extensions to consider as code
            skip_extensions: Set of extensions to skip
            skip_paths: List of path patterns to skip
            extensionless_code_files: Set of filenames without extensions that are code
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

        # Extensionless code files: use provided, then config, then default
        if extensionless_code_files is not None:
            self.extensionless_code_files = extensionless_code_files
        elif "extensionless_code_files" in filter_config:
            self.extensionless_code_files = set(filter_config["extensionless_code_files"])
        else:
            self.extensionless_code_files = EXTENSIONLESS_CODE_FILES.copy()

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

        # Check for extensionless code files (Makefile, Dockerfile, etc.)
        if name in self.extensionless_code_files:
            return True

        # Get extension (handle multiple dots like .test.js)
        ext = self._get_extension(path)

        # Check explicit skip list - no heuristic override for skip extensions
        # Text files (.txt) are documentation by definition, even if they contain code examples
        if ext in self.skip_extensions:
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
        This is only used for files with unknown extensions (not in whitelist or blacklist).
        Uses pre-compiled regex patterns for performance.
        """
        # Empty or very short content - not useful (early exit)
        if len(content.strip()) < 20:
            return False

        # Count code indicators
        code_indicators = 0

        # Shebang line is a strong indicator of executable script
        if self._has_shebang(content):
            code_indicators += 3
            # Early exit: shebang alone is strong enough
            if code_indicators >= 2:
                return True

        # Function/method definitions (using pre-compiled patterns)
        if _HEURISTIC_PATTERNS["python_def"].search(content):
            code_indicators += 2
        if _HEURISTIC_PATTERNS["js_function"].search(content):
            code_indicators += 2
        if _HEURISTIC_PATTERNS["rust_fn"].search(content):
            code_indicators += 2
        if _HEURISTIC_PATTERNS["go_func"].search(content):
            code_indicators += 2
        if _HEURISTIC_PATTERNS["perl_sub"].search(content):
            code_indicators += 2

        # Early exit if we already have enough indicators
        if code_indicators >= 2:
            return True

        # Class/struct/trait definitions
        if _HEURISTIC_PATTERNS["class"].search(content):
            code_indicators += 2
        if _HEURISTIC_PATTERNS["struct"].search(content):
            code_indicators += 2
        if _HEURISTIC_PATTERNS["trait"].search(content):
            code_indicators += 2
        if _HEURISTIC_PATTERNS["interface"].search(content):
            code_indicators += 2

        # Early exit if we already have enough indicators
        if code_indicators >= 2:
            return True

        # Import statements
        if _HEURISTIC_PATTERNS["import"].search(content):
            code_indicators += 1

        # Common programming constructs
        if _HEURISTIC_PATTERNS["control_flow"].search(content):
            code_indicators += 1

        # Variable assignments with types or keywords
        if _HEURISTIC_PATTERNS["variable"].search(content):
            code_indicators += 1

        # Early exit if we already have enough indicators
        if code_indicators >= 2:
            return True

        # Brackets and braces (common in code)
        bracket_ratio = (content.count('{') + content.count('}') +
                        content.count('[') + content.count(']')) / max(len(content), 1)
        if bracket_ratio > 0.02:
            code_indicators += 1

        # Semicolons at end of lines (common in many languages)
        if _HEURISTIC_PATTERNS["semicolon_eol"].search(content):
            code_indicators += 1

        # Arrow functions or lambdas
        if '=>' in content or 'lambda' in content:
            code_indicators += 1

        # Type annotations (TypeScript, Python type hints, Rust)
        if _HEURISTIC_PATTERNS["type_annotation"].search(content):
            code_indicators += 1

        # Decorators (Python, TypeScript)
        if _HEURISTIC_PATTERNS["decorator"].search(content):
            code_indicators += 1

        return code_indicators >= 2

    def _has_shebang(self, content: str) -> bool:
        """
        Check if content starts with a shebang line.

        Shebangs like #!/usr/bin/env python or #!/bin/bash are strong
        indicators that a file is an executable script.
        Uses pre-compiled patterns for performance.
        """
        first_line = content.lstrip().split('\n')[0] if content else ""
        if not first_line.startswith('#!'):
            return False

        # Use pre-compiled shebang patterns
        for pattern in _SHEBANG_PATTERNS:
            if pattern.match(first_line):
                return True

        return False

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

    def add_extensionless_code_file(self, filename: str):
        """Add a filename (without extension) to the extensionless code files list."""
        self.extensionless_code_files.add(filename)

    def get_config(self) -> dict:
        """Get current configuration as a dictionary."""
        return {
            "code_extensions": sorted(self.code_extensions),
            "skip_extensions": sorted(self.skip_extensions),
            "skip_paths": self.skip_paths,
            "extensionless_code_files": sorted(self.extensionless_code_files),
        }
