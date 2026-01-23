"""
Parser Factory - Main entry point for AST parsing.

This module provides:
- ASTParserFactory: Creates appropriate parser based on language
- Convenience functions for common operations
- Parser registry for language support lookup
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import ASTResult, SemanticInfo
from .base_parser import BaseParser
from .python_parser import PythonParser
from .javascript_parser import JavaScriptParser, TypeScriptParser
from .rust_parser import RustParser
from .go_parser import GoParser
from .c_parser import CParser, CppParser


# Language name aliases
LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "rs": "rust",
    "golang": "go",
    "c++": "cpp",
    "cxx": "cpp",
    "cc": "cpp",
    "h": "c",
    "hpp": "cpp",
}


class ASTParserFactory:
    """
    Factory for creating language-specific AST parsers.

    Manages parser instances and provides language lookup.
    """

    # Singleton parser instances (lazily created)
    _parsers: Dict[str, BaseParser] = {}

    # Supported language -> parser class mapping
    _parser_classes: Dict[str, type] = {
        "python": PythonParser,
        "javascript": JavaScriptParser,
        "typescript": TypeScriptParser,
        "rust": RustParser,
        "go": GoParser,
        "c": CParser,
        "cpp": CppParser,
    }

    @classmethod
    def get_parser(cls, language: str) -> Optional[BaseParser]:
        """
        Get or create a parser for the given language.

        Args:
            language: Language name (e.g., "python", "javascript")

        Returns:
            Parser instance or None if language not supported
        """
        # Normalize language name
        language = cls._normalize_language(language)

        if language not in cls._parser_classes:
            return None

        # Lazy instantiation
        if language not in cls._parsers:
            cls._parsers[language] = cls._parser_classes[language]()

        return cls._parsers[language]

    @classmethod
    def supports_language(cls, language: str) -> bool:
        """Check if a language is supported."""
        language = cls._normalize_language(language)
        return language in cls._parser_classes

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported language names."""
        return list(cls._parser_classes.keys())

    @classmethod
    def _normalize_language(cls, language: str) -> str:
        """Normalize language name using aliases."""
        language = language.lower().strip()
        return LANGUAGE_ALIASES.get(language, language)

    @classmethod
    def parse(
        cls,
        source: str,
        language: str,
        file_path: Optional[str] = None,
        language_result: Optional[Any] = None,
    ) -> ASTResult:
        """
        Parse source code with the appropriate parser.

        Args:
            source: Source code string
            language: Language name
            file_path: Optional file path for incremental parsing
            language_result: Optional LanguageResult from detection

        Returns:
            ASTResult with parsed AST

        Raises:
            ValueError: If language is not supported
        """
        parser = cls.get_parser(language)
        if parser is None:
            raise ValueError(f"Unsupported language: {language}")

        return parser.parse(source, file_path=file_path, language_result=language_result)

    @classmethod
    def parse_with_detection(
        cls,
        source: str,
        file_path: Optional[str] = None,
    ) -> ASTResult:
        """
        Parse source code, auto-detecting the language.

        Args:
            source: Source code string
            file_path: Optional file path for detection hints

        Returns:
            ASTResult with parsed AST

        Raises:
            ValueError: If language cannot be detected or is not supported
        """
        # Try to import language detector
        try:
            from ..language_detection import detect_language
            lang_result = detect_language(source, file_path=file_path)

            if not lang_result.language:
                raise ValueError("Could not detect language from source")

            if not cls.supports_language(lang_result.language):
                raise ValueError(f"Detected language '{lang_result.language}' is not supported for AST parsing")

            return cls.parse(
                source,
                lang_result.language,
                file_path=file_path,
                language_result=lang_result,
            )

        except ImportError:
            raise ValueError("Language detection requires language_detector module")

    @classmethod
    def clear_cache(cls, language: Optional[str] = None, file_path: Optional[str] = None):
        """
        Clear parser caches.

        Args:
            language: Clear cache for specific language (or all if None)
            file_path: Clear cache for specific file (or all if None)
        """
        if language:
            language = cls._normalize_language(language)
            if language in cls._parsers:
                cls._parsers[language].clear_cache(file_path)
        else:
            for parser in cls._parsers.values():
                parser.clear_cache(file_path)


# Convenience functions

def parse(
    source: str,
    language_or_result: Union[str, Any],
    file_path: Optional[str] = None,
) -> ASTResult:
    """
    Parse source code to AST.

    Args:
        source: Source code string
        language_or_result: Language name string or LanguageResult object
        file_path: Optional file path for incremental parsing

    Returns:
        ASTResult with parsed AST
    """
    # Check if it's a LanguageResult object
    if hasattr(language_or_result, 'language'):
        language = language_or_result.language
        language_result = language_or_result
    else:
        language = language_or_result
        language_result = None

    return ASTParserFactory.parse(
        source,
        language,
        file_path=file_path,
        language_result=language_result,
    )


def parse_file(file_path: str, language: Optional[str] = None) -> ASTResult:
    """
    Parse a file to AST.

    Args:
        file_path: Path to the file
        language: Optional language override (auto-detected if not provided)

    Returns:
        ASTResult with parsed AST
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    source = path.read_text(encoding='utf-8')

    if language:
        return ASTParserFactory.parse(source, language, file_path=file_path)
    else:
        return ASTParserFactory.parse_with_detection(source, file_path=file_path)


def get_parser(language: str) -> Optional[BaseParser]:
    """Get a parser instance for direct use."""
    return ASTParserFactory.get_parser(language)


def supports_language(language: str) -> bool:
    """Check if a language is supported for AST parsing."""
    return ASTParserFactory.supports_language(language)


def get_supported_languages() -> List[str]:
    """Get list of supported languages."""
    return ASTParserFactory.get_supported_languages()


# Self-test when run directly
if __name__ == "__main__":
    print("AST Parser - Self Test")
    print("=" * 60)

    # Test Python
    python_code = '''
"""Module docstring."""

import os
from typing import List, Optional

class Calculator:
    """A simple calculator class."""

    def __init__(self, initial: int = 0):
        self.value = initial

    def add(self, x: int) -> int:
        """Add x to the value."""
        self.value += x
        return self.value

    @staticmethod
    def multiply(a: int, b: int) -> int:
        return a * b

async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    pass

def generator_example():
    yield 1
    yield 2
'''

    print("\nPython parsing:")
    result = parse(python_code, "python")
    print(f"  Parse confidence: {result.parse_confidence:.2f}")
    print(f"  Functions: {[f.name for f in result.functions]}")
    print(f"  Classes: {[c.name for c in result.classes]}")
    print(f"  Imports: {[i.module for i in result.imports]}")
    if result.semantic.module_doc:
        print(f"  Module doc: {result.semantic.module_doc.content[:50]}...")

    # Test JavaScript
    js_code = '''
import { useState } from 'react';

/**
 * A greeting component
 */
class Greeter {
    constructor(name) {
        this.name = name;
    }

    async greet() {
        return `Hello, ${this.name}!`;
    }

    static create(name) {
        return new Greeter(name);
    }
}

const multiply = (a, b) => a * b;

export default Greeter;
'''

    print("\nJavaScript parsing:")
    result = parse(js_code, "javascript")
    print(f"  Parse confidence: {result.parse_confidence:.2f}")
    print(f"  Functions: {[f.name for f in result.functions]}")
    print(f"  Classes: {[c.name for c in result.classes]}")
    print(f"  Imports: {[i.module for i in result.imports]}")

    # Test TypeScript
    ts_code = '''
interface User {
    name: string;
    age: number;
}

type Status = 'active' | 'inactive';

class UserService {
    private users: User[] = [];

    async getUser(id: number): Promise<User | null> {
        return this.users.find(u => u.name === id.toString()) || null;
    }
}

export { User, UserService };
'''

    print("\nTypeScript parsing:")
    result = parse(ts_code, "typescript")
    print(f"  Parse confidence: {result.parse_confidence:.2f}")
    print(f"  Functions: {[f.name for f in result.functions]}")
    print(f"  Classes: {[c.name for c in result.classes]}")

    # Test Rust
    rust_code = '''
//! Module documentation

use std::collections::HashMap;

/// A point in 2D space
#[derive(Debug, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    /// Create a new point
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    pub fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

pub enum Color {
    Red,
    Green,
    Blue,
}

pub trait Drawable {
    fn draw(&self);
}
'''

    print("\nRust parsing:")
    result = parse(rust_code, "rust")
    print(f"  Parse confidence: {result.parse_confidence:.2f}")
    print(f"  Functions: {[f.name for f in result.functions]}")
    print(f"  Classes: {[c.name for c in result.classes]}")
    print(f"  Imports: {[i.module for i in result.imports]}")
    if result.semantic.module_doc:
        print(f"  Module doc: {result.semantic.module_doc.content}")

    # Test malformed code
    malformed_code = '''
def broken_function(
    print("missing close paren"

class IncompleteClass:
    def method(self:
        pass
'''

    print("\nMalformed Python parsing:")
    result = parse(malformed_code, "python")
    print(f"  Parse confidence: {result.parse_confidence:.2f}")
    print(f"  Error count: {result.error_count}")
    print(f"  Has errors: {result.has_errors()}")
    if result.errors:
        print(f"  First error: {result.errors[0].message}")

    print("\n" + "=" * 60)
    print(f"Supported languages: {get_supported_languages()}")
    print("Self-test complete!")
