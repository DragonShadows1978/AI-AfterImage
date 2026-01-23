"""
AST Parser - Language-specific Abstract Syntax Tree generation using tree-sitter.

This package provides unified AST parsing for multiple programming languages,
building on the LanguageResult from the language detection system.

Supported Languages:
- Python (.py, .pyw, .pyi, .pyx)
- JavaScript (.js, .jsx, .mjs, .cjs)
- TypeScript (.ts, .tsx, .mts, .cts)
- Rust (.rs)

Usage:
    from ast_parser import parse, ASTResult
    from language_detector import detect_language

    # Detect language first
    lang_result = detect_language(code)

    # Parse to AST
    ast_result = parse(code, lang_result)

    # Access semantic information
    for func in ast_result.functions:
        print(f"{func.name}: {func.signature}")

    for cls in ast_result.classes:
        print(f"class {cls.name}")

Incremental Parsing:
    # For files that change frequently, use incremental parsing
    parser = get_parser(language="python")
    result1 = parser.parse(code_v1, file_path="example.py")

    # After edit, incremental parse is more efficient
    result2 = parser.parse(code_v2, file_path="example.py")
"""

# Models are always available (no tree-sitter dependency)
from .models import (
    ASTResult,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    ErrorInfo,
    ParameterInfo,
    DocumentationInfo,
    SemanticInfo,
)

__all__ = [
    # Result types
    "ASTResult",
    "FunctionInfo",
    "ClassInfo",
    "ImportInfo",
    "ErrorInfo",
    "ParameterInfo",
    "DocumentationInfo",
    "SemanticInfo",
    # Functions (lazy loaded)
    "parse",
    "parse_file",
    "get_parser",
    "supports_language",
    "get_supported_languages",
    "ASTParserFactory",
]

__version__ = "1.0.0"


def __getattr__(name):
    """Lazy loading for tree-sitter dependent functions."""
    if name in ("parse", "parse_file", "get_parser", "supports_language",
                "get_supported_languages", "ASTParserFactory"):
        from . import parser
        return getattr(parser, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
