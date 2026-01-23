"""
Language Signature Database - Tiered pattern definitions for programming language detection.

This module provides pre-compiled regex patterns organized by confidence tier:
- Tier 1: High-confidence, language-specific patterns (unique to one language)
- Tier 2: Medium-confidence, shared patterns (appear in multiple languages)
- Tier 3: Disambiguating patterns (resolve Tier 2 conflicts)

Each pattern includes weight for confidence scoring.

CONFIDENCE SCORING ALGORITHM:
============================
1. Each language has a set of Tier 1 (high-confidence) patterns
2. When code is analyzed, each matching pattern adds its weight to the language score
3. The language with the highest total score is selected
4. Final confidence = base (0.50) + min(score, 0.45) + match_count_bonus
   - 3+ matches: +0.05 bonus
   - 5+ matches: +0.05 additional bonus
   - Maximum confidence: 0.95

PATTERN WEIGHT GUIDELINES:
=========================
- 0.40-0.50: Definitive patterns (e.g., PHP <?php tag, Rust #[derive])
- 0.30-0.40: Strong indicators (e.g., function declarations, type annotations)
- 0.20-0.30: Moderate indicators (e.g., common idioms, standard library usage)
- 0.10-0.20: Weak indicators (e.g., shared syntax elements)

ADDING A NEW LANGUAGE:
=====================
1. Create a LANG_TIER1 list with 5-10 patterns
2. Focus on patterns unique to that language
3. Add LanguageSignature to LANGUAGE_SIGNATURES dict
4. Test with diverse code samples to verify detection accuracy

SUPPORTED LANGUAGES (20):
========================
python, rust, go, typescript, javascript, java, csharp, cpp, c, ruby,
php, bash, sql, html, css, yaml, json, markdown, kotlin, swift
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Set
from enum import Enum


class ConfidenceTier(Enum):
    """Pattern confidence tiers."""
    HIGH = 1      # Unique to one language
    MEDIUM = 2    # Shared across languages
    LOW = 3       # Generic code indicators


@dataclass
class LanguagePattern:
    """A single pattern with metadata."""
    pattern: Pattern
    weight: float
    tier: ConfidenceTier
    description: str = ""

    @classmethod
    def compile(cls, regex: str, weight: float, tier: ConfidenceTier, desc: str = "", dotall: bool = False) -> 'LanguagePattern':
        """Create a compiled pattern."""
        flags = re.MULTILINE | re.IGNORECASE
        if dotall:
            flags |= re.DOTALL
        return cls(
            pattern=re.compile(regex, flags),
            weight=weight,
            tier=tier,
            description=desc
        )


@dataclass
class LanguageSignature:
    """Complete signature for a programming language."""
    name: str
    display_name: str
    extensions: Set[str]
    shebangs: List[str] = field(default_factory=list)
    modelines: List[str] = field(default_factory=list)
    tier1_patterns: List[LanguagePattern] = field(default_factory=list)
    tier2_patterns: List[LanguagePattern] = field(default_factory=list)
    disambiguators: Dict[str, List[LanguagePattern]] = field(default_factory=dict)


# =============================================================================
# TIER 1: HIGH-CONFIDENCE PATTERNS (Unique to language)
# =============================================================================

PYTHON_TIER1 = [
    LanguagePattern.compile(r'^from\s+[\w.]+\s+import\s+', 0.35, ConfidenceTier.HIGH, "Python from-import"),
    LanguagePattern.compile(r'^def\s+\w+\s*\([^)]*\)\s*->\s*\w+:', 0.35, ConfidenceTier.HIGH, "Python typed function def"),
    LanguagePattern.compile(r'^def\s+\w+\s*\([^)]*\)\s*:', 0.30, ConfidenceTier.HIGH, "Python function def"),
    LanguagePattern.compile(r"if\s+__name__\s*==\s*['\"]__main__['\"]", 0.40, ConfidenceTier.HIGH, "Python main guard"),
    LanguagePattern.compile(r'^\s*@\w+(?:\.\w+)*(?:\([^)]*\))?\s*$', 0.25, ConfidenceTier.HIGH, "Python decorator"),
    LanguagePattern.compile(r'self\.\w+', 0.20, ConfidenceTier.HIGH, "Python self reference"),
    LanguagePattern.compile(r'__init__|__str__|__repr__|__call__', 0.30, ConfidenceTier.HIGH, "Python dunder methods"),
    LanguagePattern.compile(r'^\s*async\s+def\s+\w+', 0.30, ConfidenceTier.HIGH, "Python async def"),
    LanguagePattern.compile(r'print\s*\([^)]*\)', 0.15, ConfidenceTier.HIGH, "Python print function"),
    LanguagePattern.compile(r'raise\s+\w+(?:Error|Exception)', 0.25, ConfidenceTier.HIGH, "Python raise exception"),
]

RUST_TIER1 = [
    LanguagePattern.compile(r'^(?:pub\s+)?fn\s+\w+\s*(?:<[^>]+>)?\s*\([^)]*\)\s*(?:->\s*[^{]+)?\s*\{', 0.40, ConfidenceTier.HIGH, "Rust function def"),
    LanguagePattern.compile(r'^(?:pub\s+)?struct\s+\w+(?:\s*<[^>]+>)?\s*\{', 0.35, ConfidenceTier.HIGH, "Rust struct def"),
    LanguagePattern.compile(r'^impl(?:\s*<[^>]+>)?\s+\w+', 0.40, ConfidenceTier.HIGH, "Rust impl block"),
    LanguagePattern.compile(r'^use\s+[\w:]+(?:::\{[^}]+\})?;', 0.35, ConfidenceTier.HIGH, "Rust use statement"),
    LanguagePattern.compile(r'#\[derive\([^\]]+\)\]', 0.40, ConfidenceTier.HIGH, "Rust derive attribute"),
    LanguagePattern.compile(r'#\[\w+(?:\([^\]]*\))?\]', 0.25, ConfidenceTier.HIGH, "Rust attribute"),
    LanguagePattern.compile(r'&mut\s+\w+|&\w+', 0.20, ConfidenceTier.HIGH, "Rust borrow"),
    LanguagePattern.compile(r'let\s+mut\s+\w+', 0.30, ConfidenceTier.HIGH, "Rust mutable let"),
    LanguagePattern.compile(r'Option<|Result<|Vec<|Box<', 0.35, ConfidenceTier.HIGH, "Rust standard types"),
    LanguagePattern.compile(r'\.unwrap\(\)|\.expect\(|\.ok\(\)|\.err\(\)', 0.30, ConfidenceTier.HIGH, "Rust Result methods"),
    LanguagePattern.compile(r'println!\s*\(|eprintln!\s*\(|format!\s*\(', 0.35, ConfidenceTier.HIGH, "Rust macros"),
]

GO_TIER1 = [
    LanguagePattern.compile(r'^package\s+\w+\s*$', 0.40, ConfidenceTier.HIGH, "Go package declaration"),
    LanguagePattern.compile(r'^func\s+(?:\(\w+\s+\*?\w+\)\s+)?\w+\s*\([^)]*\)(?:\s+(?:\([^)]+\)|\w+))?\s*\{', 0.40, ConfidenceTier.HIGH, "Go function def"),
    LanguagePattern.compile(r'^import\s+\(', 0.35, ConfidenceTier.HIGH, "Go multi-import"),
    LanguagePattern.compile(r'^type\s+\w+\s+struct\s*\{', 0.40, ConfidenceTier.HIGH, "Go struct type"),
    LanguagePattern.compile(r'^type\s+\w+\s+interface\s*\{', 0.40, ConfidenceTier.HIGH, "Go interface type"),
    LanguagePattern.compile(r':=', 0.25, ConfidenceTier.HIGH, "Go short assignment"),
    LanguagePattern.compile(r'go\s+\w+\s*\(', 0.30, ConfidenceTier.HIGH, "Go goroutine"),
    LanguagePattern.compile(r'defer\s+\w+', 0.30, ConfidenceTier.HIGH, "Go defer"),
    LanguagePattern.compile(r'make\s*\(\s*(?:chan|map|\[\])', 0.30, ConfidenceTier.HIGH, "Go make builtin"),
    LanguagePattern.compile(r'fmt\.Print|fmt\.Sprint|fmt\.Errorf', 0.30, ConfidenceTier.HIGH, "Go fmt package"),
]

TYPESCRIPT_TIER1 = [
    LanguagePattern.compile(r'^interface\s+\w+(?:\s+extends\s+\w+)?\s*\{', 0.40, ConfidenceTier.HIGH, "TS interface"),
    LanguagePattern.compile(r'^type\s+\w+(?:\s*<[^>]+>)?\s*=', 0.35, ConfidenceTier.HIGH, "TS type alias"),
    LanguagePattern.compile(r':\s*(?:string|number|boolean|any|void|never|unknown)\s*[;=,)]', 0.30, ConfidenceTier.HIGH, "TS type annotation"),
    LanguagePattern.compile(r'<\w+(?:,\s*\w+)*>\s*\(', 0.25, ConfidenceTier.HIGH, "TS generics"),
    LanguagePattern.compile(r'\w+\s*:\s*\w+\s*\[\]', 0.25, ConfidenceTier.HIGH, "TS array type"),
    LanguagePattern.compile(r'as\s+(?:const|string|number|boolean|\w+)', 0.25, ConfidenceTier.HIGH, "TS type assertion"),
    LanguagePattern.compile(r'\?\.\w+|!\.\w+', 0.20, ConfidenceTier.HIGH, "TS optional chaining"),
    LanguagePattern.compile(r'private\s+\w+:|public\s+\w+:|protected\s+\w+:', 0.30, ConfidenceTier.HIGH, "TS access modifiers"),
]

JAVASCRIPT_TIER1 = [
    LanguagePattern.compile(r'^(?:export\s+)?(?:default\s+)?function\s+\w+\s*\(', 0.30, ConfidenceTier.HIGH, "JS function"),
    LanguagePattern.compile(r'^const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>', 0.35, ConfidenceTier.HIGH, "JS arrow function"),
    LanguagePattern.compile(r'^const\s+\w+\s*=\s*require\s*\(', 0.35, ConfidenceTier.HIGH, "JS require"),
    LanguagePattern.compile(r'module\.exports\s*=', 0.40, ConfidenceTier.HIGH, "JS module.exports"),
    LanguagePattern.compile(r'console\.(?:log|error|warn|info)\s*\(', 0.25, ConfidenceTier.HIGH, "JS console"),
    LanguagePattern.compile(r'document\.(?:getElementById|querySelector|createElement)', 0.35, ConfidenceTier.HIGH, "JS DOM"),
    LanguagePattern.compile(r'window\.\w+|localStorage\.\w+', 0.30, ConfidenceTier.HIGH, "JS browser globals"),
    LanguagePattern.compile(r'async\s+function\s+\w+|await\s+\w+', 0.25, ConfidenceTier.HIGH, "JS async/await"),
]

JAVA_TIER1 = [
    LanguagePattern.compile(r'^public\s+class\s+\w+(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{', 0.45, ConfidenceTier.HIGH, "Java class"),
    LanguagePattern.compile(r'^(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:\w+(?:<[^>]+>)?)\s+\w+\s*\(', 0.40, ConfidenceTier.HIGH, "Java method"),
    LanguagePattern.compile(r'^package\s+[\w.]+;', 0.40, ConfidenceTier.HIGH, "Java package"),
    LanguagePattern.compile(r'^import\s+(?:static\s+)?[\w.]+(?:\.\*)?;', 0.35, ConfidenceTier.HIGH, "Java import"),
    LanguagePattern.compile(r'System\.out\.print(?:ln)?\s*\(', 0.35, ConfidenceTier.HIGH, "Java print"),
    LanguagePattern.compile(r'@Override|@SuppressWarnings|@Deprecated', 0.30, ConfidenceTier.HIGH, "Java annotation"),
    LanguagePattern.compile(r'new\s+\w+(?:<[^>]+>)?\s*\(', 0.25, ConfidenceTier.HIGH, "Java new"),
    LanguagePattern.compile(r'(?:public|private|protected)\s+(?:static\s+)?void\s+\w+\s*\(', 0.35, ConfidenceTier.HIGH, "Java void method"),
    LanguagePattern.compile(r'(?:public|private|protected)\s+String\s+\w+', 0.30, ConfidenceTier.HIGH, "Java String field"),
    LanguagePattern.compile(r'class\s+\w+[^}]*System\.out\.print', 0.35, ConfidenceTier.HIGH, "Java class with System.out", dotall=True),
]

CSHARP_TIER1 = [
    LanguagePattern.compile(r'^(?:public|private|internal)\s+(?:partial\s+)?class\s+\w+', 0.40, ConfidenceTier.HIGH, "C# class"),
    LanguagePattern.compile(r'^namespace\s+[\w.]+', 0.40, ConfidenceTier.HIGH, "C# namespace"),
    LanguagePattern.compile(r'^using\s+[\w.]+;', 0.30, ConfidenceTier.HIGH, "C# using"),
    LanguagePattern.compile(r'Console\.Write(?:Line)?\s*\(', 0.35, ConfidenceTier.HIGH, "C# Console"),
    LanguagePattern.compile(r'\[(?:Serializable|Obsolete|DllImport|Attribute)\]', 0.30, ConfidenceTier.HIGH, "C# attributes"),
    LanguagePattern.compile(r'get\s*\{|set\s*\{|get;|set;', 0.30, ConfidenceTier.HIGH, "C# properties"),
    LanguagePattern.compile(r'(?:async\s+)?Task<\w+>', 0.30, ConfidenceTier.HIGH, "C# Task"),
]

CPP_TIER1 = [
    LanguagePattern.compile(r'^#include\s*<[\w./]+>', 0.35, ConfidenceTier.HIGH, "C++ include"),
    LanguagePattern.compile(r'^#include\s*"[\w./]+"', 0.30, ConfidenceTier.HIGH, "C include local"),
    LanguagePattern.compile(r'^using\s+namespace\s+\w+;', 0.40, ConfidenceTier.HIGH, "C++ using namespace"),
    LanguagePattern.compile(r'std::\w+', 0.30, ConfidenceTier.HIGH, "C++ std namespace"),
    LanguagePattern.compile(r'cout\s*<<|cin\s*>>', 0.35, ConfidenceTier.HIGH, "C++ IO streams"),
    LanguagePattern.compile(r'^template\s*<', 0.40, ConfidenceTier.HIGH, "C++ template"),
    LanguagePattern.compile(r'nullptr|auto\s+\w+\s*=', 0.25, ConfidenceTier.HIGH, "C++ modern keywords"),
    LanguagePattern.compile(r'::\s*\w+\s*\(', 0.20, ConfidenceTier.HIGH, "C++ scope resolution"),
    LanguagePattern.compile(r'unique_ptr<|shared_ptr<|weak_ptr<|make_unique<|make_shared<', 0.40, ConfidenceTier.HIGH, "C++ smart pointers"),
    LanguagePattern.compile(r'^\s*(?:public|private|protected)\s*:', 0.35, ConfidenceTier.HIGH, "C++ access specifiers"),
    LanguagePattern.compile(r'virtual\s+\w+\s+\w+\s*\(|override\s*[;{]', 0.35, ConfidenceTier.HIGH, "C++ virtual/override"),
    LanguagePattern.compile(r'\[[^\]]*\]\s*\([^)]*\)\s*(?:->|\{)', 0.35, ConfidenceTier.HIGH, "C++ lambda"),
    LanguagePattern.compile(r'for\s*\(\s*(?:auto|const\s+auto)[&\s]+\w+\s*:\s*\w+', 0.35, ConfidenceTier.HIGH, "C++ range-for"),
    LanguagePattern.compile(r'std::(?:vector|map|unordered_map|set|list|deque|array)<', 0.35, ConfidenceTier.HIGH, "C++ containers"),
    LanguagePattern.compile(r'^namespace\s+\w+\s*\{', 0.35, ConfidenceTier.HIGH, "C++ namespace def"),
    LanguagePattern.compile(r'^class\s+\w+\s*:\s*(?:public|private|protected)\s+\w+', 0.35, ConfidenceTier.HIGH, "C++ class inheritance"),
]

RUBY_TIER1 = [
    LanguagePattern.compile(r'^require\s+[\'"][\w/]+[\'"]', 0.35, ConfidenceTier.HIGH, "Ruby require"),
    LanguagePattern.compile(r'^require_relative\s+', 0.40, ConfidenceTier.HIGH, "Ruby require_relative"),
    LanguagePattern.compile(r'^class\s+\w+(?:\s*<\s*\w+)?', 0.30, ConfidenceTier.HIGH, "Ruby class"),
    LanguagePattern.compile(r'^module\s+\w+', 0.35, ConfidenceTier.HIGH, "Ruby module"),
    LanguagePattern.compile(r'^\s*def\s+\w+(?:\s*\([^)]*\))?', 0.30, ConfidenceTier.HIGH, "Ruby def"),
    LanguagePattern.compile(r'@\w+\s*=', 0.25, ConfidenceTier.HIGH, "Ruby instance var"),
    LanguagePattern.compile(r'attr_(?:reader|writer|accessor)', 0.40, ConfidenceTier.HIGH, "Ruby attr"),
    LanguagePattern.compile(r'\.each\s*do\s*\|', 0.30, ConfidenceTier.HIGH, "Ruby each block"),
    LanguagePattern.compile(r'puts\s+[\'"]|puts\s+\w+', 0.25, ConfidenceTier.HIGH, "Ruby puts"),
    LanguagePattern.compile(r'end\s*$', 0.15, ConfidenceTier.HIGH, "Ruby end keyword"),
]

PHP_TIER1 = [
    LanguagePattern.compile(r'^<\?php', 0.50, ConfidenceTier.HIGH, "PHP opening tag"),
    LanguagePattern.compile(r'^\$\w+\s*=', 0.25, ConfidenceTier.HIGH, "PHP variable"),
    LanguagePattern.compile(r'^function\s+\w+\s*\([^)]*\)', 0.30, ConfidenceTier.HIGH, "PHP function"),
    LanguagePattern.compile(r'^class\s+\w+(?:\s+extends\s+\w+)?', 0.30, ConfidenceTier.HIGH, "PHP class"),
    LanguagePattern.compile(r'echo\s+[\'"\$]', 0.30, ConfidenceTier.HIGH, "PHP echo"),
    LanguagePattern.compile(r'\$this->\w+', 0.35, ConfidenceTier.HIGH, "PHP this"),
    LanguagePattern.compile(r'->\w+\s*\(', 0.20, ConfidenceTier.HIGH, "PHP method call"),
]

BASH_TIER1 = [
    LanguagePattern.compile(r'^#!/bin/(?:ba)?sh', 0.50, ConfidenceTier.HIGH, "Bash shebang"),
    LanguagePattern.compile(r'^#!/usr/bin/env\s+(?:ba)?sh', 0.50, ConfidenceTier.HIGH, "Env shebang"),
    LanguagePattern.compile(r'^\s*\w+=(?:\$\(|[\'"])', 0.25, ConfidenceTier.HIGH, "Shell assignment"),
    LanguagePattern.compile(r'\$\{\w+\}|\$\w+', 0.20, ConfidenceTier.HIGH, "Shell variable"),
    LanguagePattern.compile(r'\|\s*grep|\|\s*awk|\|\s*sed|\|\s*cut', 0.30, ConfidenceTier.HIGH, "Shell pipe"),
    LanguagePattern.compile(r'^\s*if\s+\[\[?\s+', 0.30, ConfidenceTier.HIGH, "Shell if"),
    LanguagePattern.compile(r'^\s*for\s+\w+\s+in\s+', 0.30, ConfidenceTier.HIGH, "Shell for"),
    LanguagePattern.compile(r'^\s*function\s+\w+\s*\(\s*\)|^\s*\w+\s*\(\s*\)\s*\{', 0.30, ConfidenceTier.HIGH, "Shell function"),
]

SQL_TIER1 = [
    LanguagePattern.compile(r'^SELECT\s+(?:DISTINCT\s+)?\*?\s*(?:\w+|\w+\.\w+)', 0.40, ConfidenceTier.HIGH, "SQL SELECT"),
    LanguagePattern.compile(r'^INSERT\s+INTO\s+\w+', 0.40, ConfidenceTier.HIGH, "SQL INSERT"),
    LanguagePattern.compile(r'^UPDATE\s+\w+\s+SET', 0.40, ConfidenceTier.HIGH, "SQL UPDATE"),
    LanguagePattern.compile(r'^DELETE\s+FROM\s+\w+', 0.40, ConfidenceTier.HIGH, "SQL DELETE"),
    LanguagePattern.compile(r'^CREATE\s+(?:TABLE|INDEX|VIEW|DATABASE)', 0.40, ConfidenceTier.HIGH, "SQL CREATE"),
    LanguagePattern.compile(r'^ALTER\s+TABLE\s+\w+', 0.40, ConfidenceTier.HIGH, "SQL ALTER"),
    LanguagePattern.compile(r'FROM\s+\w+\s+(?:WHERE|JOIN|GROUP BY|ORDER BY)', 0.30, ConfidenceTier.HIGH, "SQL clauses"),
    LanguagePattern.compile(r'INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|OUTER\s+JOIN', 0.35, ConfidenceTier.HIGH, "SQL JOIN"),
]

# =============================================================================
# TIER 2: MEDIUM-CONFIDENCE PATTERNS (Shared across languages)
# =============================================================================

SHARED_PATTERNS = {
    "class_definition": LanguagePattern.compile(r'^\s*class\s+\w+', 0.15, ConfidenceTier.MEDIUM, "Class definition"),
    "import_statement": LanguagePattern.compile(r'^import\s+', 0.10, ConfidenceTier.MEDIUM, "Import statement"),
    "function_keyword": LanguagePattern.compile(r'function\s+\w+', 0.10, ConfidenceTier.MEDIUM, "Function keyword"),
    "arrow_function": LanguagePattern.compile(r'=>', 0.10, ConfidenceTier.MEDIUM, "Arrow syntax"),
    "curly_braces": LanguagePattern.compile(r'\{[^}]+\}', 0.05, ConfidenceTier.MEDIUM, "Curly braces"),
    "semicolons": LanguagePattern.compile(r';\s*$', 0.05, ConfidenceTier.MEDIUM, "Semicolons"),
    "comments_slash": LanguagePattern.compile(r'//.*$|/\*.*?\*/', 0.05, ConfidenceTier.MEDIUM, "C-style comments"),
    "comments_hash": LanguagePattern.compile(r'#.*$', 0.05, ConfidenceTier.MEDIUM, "Hash comments"),
}


# =============================================================================
# POLYGLOT PATTERNS (Embedded languages)
# =============================================================================

POLYGLOT_PATTERNS = {
    "html_with_js": [
        LanguagePattern.compile(r'<script(?:\s+[^>]*)?>.*?</script>', 0.30, ConfidenceTier.HIGH, "Script tag", dotall=True),
    ],
    "html_with_css": [
        LanguagePattern.compile(r'<style(?:\s+[^>]*)?>.*?</style>', 0.30, ConfidenceTier.HIGH, "Style tag", dotall=True),
    ],
    "html_with_php": [
        LanguagePattern.compile(r'<\?(?:php|=)', 0.35, ConfidenceTier.HIGH, "PHP tag"),
    ],
    "template_jinja": [
        LanguagePattern.compile(r'\{\{.*?\}\}|\{%.*?%\}', 0.30, ConfidenceTier.HIGH, "Jinja/Django template", dotall=True),
    ],
    "template_erb": [
        LanguagePattern.compile(r'<%=?.*?%>', 0.30, ConfidenceTier.HIGH, "ERB template", dotall=True),
    ],
    "vue_sfc": [
        LanguagePattern.compile(r'<template>.*?</template>', 0.35, ConfidenceTier.HIGH, "Vue template", dotall=True),
    ],
}


# =============================================================================
# SPECIAL FILE PATTERNS (No extension needed)
# =============================================================================

SPECIAL_FILES = {
    "Makefile": "makefile",
    "makefile": "makefile",
    "GNUmakefile": "makefile",
    "Dockerfile": "dockerfile",
    "dockerfile": "dockerfile",
    "Containerfile": "dockerfile",
    "Vagrantfile": "ruby",
    "Gemfile": "ruby",
    "Rakefile": "ruby",
    "Jenkinsfile": "groovy",
    "BUILD": "starlark",
    "WORKSPACE": "starlark",
    ".gitignore": "gitignore",
    ".dockerignore": "dockerignore",
    ".editorconfig": "editorconfig",
    "CMakeLists.txt": "cmake",
    "package.json": "json",
    "tsconfig.json": "json",
    "requirements.txt": "requirements",
    "Pipfile": "toml",
    "pyproject.toml": "toml",
    "Cargo.toml": "toml",
    "go.mod": "gomod",
    "go.sum": "gosum",
}


# =============================================================================
# SHEBANG PATTERNS
# =============================================================================

SHEBANG_PATTERNS = {
    r'python3?': 'python',
    r'node': 'javascript',
    r'ruby': 'ruby',
    r'perl': 'perl',
    r'php': 'php',
    r'bash': 'bash',
    r'sh': 'bash',
    r'zsh': 'bash',
    r'env\s+python': 'python',
    r'env\s+node': 'javascript',
    r'env\s+ruby': 'ruby',
    r'env\s+perl': 'perl',
    r'env\s+bash': 'bash',
}


# =============================================================================
# VIM/EMACS MODELINES
# =============================================================================

MODELINE_PATTERNS = {
    r'vim:.*(?:ft|filetype)=(\w+)': 'vim',
    r'-\*-.*mode:\s*(\w+).*-\*-': 'emacs',
}


# =============================================================================
# LANGUAGE REGISTRY
# =============================================================================

LANGUAGE_SIGNATURES: Dict[str, LanguageSignature] = {
    "python": LanguageSignature(
        name="python",
        display_name="Python",
        extensions={".py", ".pyw", ".pyi", ".pyx"},
        shebangs=["python", "python3", "python2"],
        tier1_patterns=PYTHON_TIER1,
    ),
    "rust": LanguageSignature(
        name="rust",
        display_name="Rust",
        extensions={".rs"},
        tier1_patterns=RUST_TIER1,
    ),
    "go": LanguageSignature(
        name="go",
        display_name="Go",
        extensions={".go"},
        tier1_patterns=GO_TIER1,
    ),
    "typescript": LanguageSignature(
        name="typescript",
        display_name="TypeScript",
        extensions={".ts", ".tsx", ".mts", ".cts"},
        tier1_patterns=TYPESCRIPT_TIER1,
    ),
    "javascript": LanguageSignature(
        name="javascript",
        display_name="JavaScript",
        extensions={".js", ".jsx", ".mjs", ".cjs"},
        shebangs=["node"],
        tier1_patterns=JAVASCRIPT_TIER1,
    ),
    "java": LanguageSignature(
        name="java",
        display_name="Java",
        extensions={".java"},
        tier1_patterns=JAVA_TIER1,
    ),
    "csharp": LanguageSignature(
        name="csharp",
        display_name="C#",
        extensions={".cs", ".csx"},
        tier1_patterns=CSHARP_TIER1,
    ),
    "cpp": LanguageSignature(
        name="cpp",
        display_name="C++",
        extensions={".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx", ".h++"},
        tier1_patterns=CPP_TIER1,
    ),
    "c": LanguageSignature(
        name="c",
        display_name="C",
        extensions={".c", ".h"},
        tier1_patterns=[
            LanguagePattern.compile(r'^#include\s*<\w+\.h>', 0.35, ConfidenceTier.HIGH, "C include"),
            LanguagePattern.compile(r'^int\s+main\s*\(', 0.35, ConfidenceTier.HIGH, "C main"),
            LanguagePattern.compile(r'printf\s*\(|scanf\s*\(', 0.30, ConfidenceTier.HIGH, "C stdio"),
            LanguagePattern.compile(r'malloc\s*\(|free\s*\(|sizeof\s*\(', 0.30, ConfidenceTier.HIGH, "C memory"),
        ],
    ),
    "ruby": LanguageSignature(
        name="ruby",
        display_name="Ruby",
        extensions={".rb", ".rake", ".gemspec"},
        shebangs=["ruby"],
        tier1_patterns=RUBY_TIER1,
    ),
    "php": LanguageSignature(
        name="php",
        display_name="PHP",
        extensions={".php", ".phtml", ".php3", ".php4", ".php5", ".php7"},
        shebangs=["php"],
        tier1_patterns=PHP_TIER1,
    ),
    "bash": LanguageSignature(
        name="bash",
        display_name="Bash/Shell",
        extensions={".sh", ".bash", ".zsh"},
        shebangs=["bash", "sh", "zsh"],
        tier1_patterns=BASH_TIER1,
    ),
    "sql": LanguageSignature(
        name="sql",
        display_name="SQL",
        extensions={".sql"},
        tier1_patterns=SQL_TIER1,
    ),
    "html": LanguageSignature(
        name="html",
        display_name="HTML",
        extensions={".html", ".htm", ".xhtml"},
        tier1_patterns=[
            LanguagePattern.compile(r'^<!DOCTYPE\s+html', 0.50, ConfidenceTier.HIGH, "HTML doctype"),
            LanguagePattern.compile(r'<html(?:\s+[^>]*)?>|</html>', 0.40, ConfidenceTier.HIGH, "HTML tag"),
            LanguagePattern.compile(r'<head>|<body>|<div\s|<span\s|<p\s|<a\s', 0.25, ConfidenceTier.HIGH, "HTML elements"),
        ],
    ),
    "css": LanguageSignature(
        name="css",
        display_name="CSS",
        extensions={".css"},
        tier1_patterns=[
            LanguagePattern.compile(r'^\s*\.\w+\s*\{', 0.30, ConfidenceTier.HIGH, "CSS class selector"),
            LanguagePattern.compile(r'^\s*#\w+\s*\{', 0.30, ConfidenceTier.HIGH, "CSS ID selector"),
            LanguagePattern.compile(r'@media\s*\(|@keyframes\s+\w+|@import\s+', 0.35, ConfidenceTier.HIGH, "CSS at-rules"),
            LanguagePattern.compile(r'(?:color|background|margin|padding|font-size)\s*:', 0.25, ConfidenceTier.HIGH, "CSS properties"),
        ],
    ),
    "yaml": LanguageSignature(
        name="yaml",
        display_name="YAML",
        extensions={".yaml", ".yml"},
        tier1_patterns=[
            LanguagePattern.compile(r'^---\s*$', 0.30, ConfidenceTier.HIGH, "YAML document start"),
            LanguagePattern.compile(r'^\w+:\s*$', 0.20, ConfidenceTier.HIGH, "YAML key"),
            LanguagePattern.compile(r'^\s+-\s+\w+', 0.20, ConfidenceTier.HIGH, "YAML list item"),
        ],
    ),
    "json": LanguageSignature(
        name="json",
        display_name="JSON",
        extensions={".json", ".jsonl"},
        tier1_patterns=[
            LanguagePattern.compile(r'^\s*\{', 0.20, ConfidenceTier.HIGH, "JSON object start"),
            LanguagePattern.compile(r'^\s*\[', 0.20, ConfidenceTier.HIGH, "JSON array start"),
            LanguagePattern.compile(r'"[\w-]+"\s*:', 0.25, ConfidenceTier.HIGH, "JSON key"),
        ],
    ),
    "markdown": LanguageSignature(
        name="markdown",
        display_name="Markdown",
        extensions={".md", ".markdown", ".mdown"},
        tier1_patterns=[
            LanguagePattern.compile(r'^#{1,6}\s+.+$', 0.25, ConfidenceTier.HIGH, "MD heading"),
            LanguagePattern.compile(r'^\s*[-*+]\s+\w+', 0.15, ConfidenceTier.HIGH, "MD list"),
            LanguagePattern.compile(r'\[.+\]\(.+\)', 0.20, ConfidenceTier.HIGH, "MD link"),
            LanguagePattern.compile(r'```\w*\n', 0.25, ConfidenceTier.HIGH, "MD code fence"),
        ],
    ),
    "kotlin": LanguageSignature(
        name="kotlin",
        display_name="Kotlin",
        extensions={".kt", ".kts"},
        tier1_patterns=[
            LanguagePattern.compile(r'^fun\s+\w+\s*\(', 0.35, ConfidenceTier.HIGH, "Kotlin fun"),
            LanguagePattern.compile(r'^data\s+class\s+\w+', 0.40, ConfidenceTier.HIGH, "Kotlin data class"),
            LanguagePattern.compile(r'^(?:open\s+|sealed\s+)?class\s+\w+\s*\([^)]+\)', 0.35, ConfidenceTier.HIGH, "Kotlin class with constructor"),
            LanguagePattern.compile(r'val\s+\w+\s*:', 0.25, ConfidenceTier.HIGH, "Kotlin val"),
            LanguagePattern.compile(r'var\s+\w+\s*:', 0.25, ConfidenceTier.HIGH, "Kotlin var"),
            LanguagePattern.compile(r'println\s*\(|print\s*\(', 0.20, ConfidenceTier.HIGH, "Kotlin print"),
            LanguagePattern.compile(r'suspend\s+fun|runBlocking|launch\s*\{', 0.35, ConfidenceTier.HIGH, "Kotlin coroutines"),
            LanguagePattern.compile(r'object\s+\w+\s*:', 0.30, ConfidenceTier.HIGH, "Kotlin object"),
        ],
    ),
    "swift": LanguageSignature(
        name="swift",
        display_name="Swift",
        extensions={".swift"},
        tier1_patterns=[
            LanguagePattern.compile(r'^(?:public\s+|private\s+|internal\s+)?func\s+\w+\s*\([^)]*\)\s*->\s*\w+', 0.40, ConfidenceTier.HIGH, "Swift func with return"),
            LanguagePattern.compile(r'^(?:public\s+|private\s+|internal\s+)?func\s+\w+\s*\([^)]*\)\s*\{', 0.30, ConfidenceTier.HIGH, "Swift func"),
            LanguagePattern.compile(r'^(?:public\s+|private\s+)?(?:class|struct)\s+\w+\s*:\s*\w+(?:\s*,\s*\w+)*\s*\{', 0.35, ConfidenceTier.HIGH, "Swift type with protocol"),
            LanguagePattern.compile(r'^\s*let\s+\w+\s*:\s*\w+\s*=', 0.30, ConfidenceTier.HIGH, "Swift let with type"),
            LanguagePattern.compile(r'^\s*var\s+\w+\s*:\s*\w+\s*=', 0.30, ConfidenceTier.HIGH, "Swift var with type"),
            LanguagePattern.compile(r'guard\s+let\s+\w+\s*=', 0.35, ConfidenceTier.HIGH, "Swift guard let"),
            LanguagePattern.compile(r'if\s+let\s+\w+\s*=', 0.35, ConfidenceTier.HIGH, "Swift if let"),
            LanguagePattern.compile(r'\?\?|\?\.\w+', 0.25, ConfidenceTier.HIGH, "Swift optionals"),
            LanguagePattern.compile(r'print\s*\("[^"]*\\?\([^)]*\)[^"]*"\)', 0.30, ConfidenceTier.HIGH, "Swift string interpolation print"),
            LanguagePattern.compile(r'@(?:IBOutlet|IBAction|objc|available)', 0.40, ConfidenceTier.HIGH, "Swift attributes"),
        ],
    ),
}


def get_signature(language: str) -> LanguageSignature | None:
    """Get the signature for a language by name."""
    return LANGUAGE_SIGNATURES.get(language.lower())


def get_all_extensions() -> Dict[str, str]:
    """Get mapping of all extensions to their languages."""
    ext_map = {}
    for name, sig in LANGUAGE_SIGNATURES.items():
        for ext in sig.extensions:
            ext_map[ext] = name
    return ext_map


def get_language_names() -> List[str]:
    """Get list of all supported language names."""
    return list(LANGUAGE_SIGNATURES.keys())
