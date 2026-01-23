"""
AST Parser Models - Data structures for AST results and semantic information.

These models provide a unified interface for AST data regardless of source language.
All semantic information (functions, classes, imports) uses consistent structures.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class Visibility(Enum):
    """Visibility/access modifier for declarations."""
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


@dataclass
class ParameterInfo:
    """Information about a function/method parameter."""
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None
    is_variadic: bool = False  # *args, ...rest
    is_keyword_variadic: bool = False  # **kwargs

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type_annotation": self.type_annotation,
            "default_value": self.default_value,
            "is_variadic": self.is_variadic,
            "is_keyword_variadic": self.is_keyword_variadic,
        }


@dataclass
class DocumentationInfo:
    """Documentation string or comment information."""
    content: str
    format: str = "plain"  # "plain", "docstring", "jsdoc", "rustdoc"
    start_line: int = 0
    end_line: int = 0

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "format": self.format,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


@dataclass
class FunctionInfo:
    """Information about a function or method."""
    name: str
    parameters: List[ParameterInfo] = field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    is_generator: bool = False
    is_method: bool = False
    is_static: bool = False
    is_class_method: bool = False  # Python @classmethod
    visibility: Visibility = Visibility.UNKNOWN
    decorators: List[str] = field(default_factory=list)
    documentation: Optional[DocumentationInfo] = None

    # Location in source
    start_line: int = 0
    end_line: int = 0
    start_column: int = 0
    end_column: int = 0

    # Parent class name if this is a method
    parent_class: Optional[str] = None

    @property
    def signature(self) -> str:
        """Generate a human-readable signature."""
        params = ", ".join(
            f"{p.name}: {p.type_annotation}" if p.type_annotation else p.name
            for p in self.parameters
        )
        sig = f"{self.name}({params})"
        if self.return_type:
            sig += f" -> {self.return_type}"
        if self.is_async:
            sig = f"async {sig}"
        return sig

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "signature": self.signature,
            "parameters": [p.to_dict() for p in self.parameters],
            "return_type": self.return_type,
            "is_async": self.is_async,
            "is_generator": self.is_generator,
            "is_method": self.is_method,
            "is_static": self.is_static,
            "is_class_method": self.is_class_method,
            "visibility": self.visibility.value,
            "decorators": self.decorators,
            "documentation": self.documentation.to_dict() if self.documentation else None,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parent_class": self.parent_class,
        }


@dataclass
class ClassInfo:
    """Information about a class, struct, or interface."""
    name: str
    kind: str = "class"  # "class", "struct", "interface", "trait", "enum"
    bases: List[str] = field(default_factory=list)  # Parent classes/interfaces
    methods: List[FunctionInfo] = field(default_factory=list)
    fields: List[Dict[str, Any]] = field(default_factory=list)  # Class/instance variables
    visibility: Visibility = Visibility.UNKNOWN
    decorators: List[str] = field(default_factory=list)
    documentation: Optional[DocumentationInfo] = None
    is_abstract: bool = False

    # Generic type parameters
    type_parameters: List[str] = field(default_factory=list)

    # Location
    start_line: int = 0
    end_line: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "bases": self.bases,
            "methods": [m.to_dict() for m in self.methods],
            "fields": self.fields,
            "visibility": self.visibility.value,
            "decorators": self.decorators,
            "documentation": self.documentation.to_dict() if self.documentation else None,
            "is_abstract": self.is_abstract,
            "type_parameters": self.type_parameters,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str  # The module being imported
    names: List[str] = field(default_factory=list)  # Specific names imported (from x import a, b)
    alias: Optional[str] = None  # import x as alias
    is_wildcard: bool = False  # from x import *
    is_type_only: bool = False  # TypeScript: import type { X }

    # Location
    start_line: int = 0
    end_line: int = 0

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "names": self.names,
            "alias": self.alias,
            "is_wildcard": self.is_wildcard,
            "is_type_only": self.is_type_only,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


@dataclass
class ErrorInfo:
    """Information about a parse error in the AST."""
    message: str
    error_type: str = "syntax"  # "syntax", "missing", "unexpected"
    start_line: int = 0
    end_line: int = 0
    start_column: int = 0
    end_column: int = 0
    context: str = ""  # Surrounding code context

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "error_type": self.error_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_column": self.start_column,
            "end_column": self.end_column,
            "context": self.context,
        }


@dataclass
class SemanticInfo:
    """Aggregated semantic information extracted from AST."""
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)

    # Top-level variables/constants
    variables: List[Dict[str, Any]] = field(default_factory=list)

    # Module-level documentation
    module_doc: Optional[DocumentationInfo] = None

    def to_dict(self) -> dict:
        return {
            "functions": [f.to_dict() for f in self.functions],
            "classes": [c.to_dict() for c in self.classes],
            "imports": [i.to_dict() for i in self.imports],
            "variables": self.variables,
            "module_doc": self.module_doc.to_dict() if self.module_doc else None,
        }


@dataclass
class ASTResult:
    """
    Result of AST parsing with semantic information.

    This is the unified interface for AST data regardless of source language.
    It wraps the tree-sitter parse tree and provides extracted semantic information.
    """
    # Source language info
    language: str
    source_language_result: Optional[Any] = None  # LanguageResult from detector

    # Raw tree-sitter data (for advanced use)
    parse_tree: Optional[Any] = None  # tree_sitter.Tree
    root_node: Optional[Any] = None  # tree_sitter.Node

    # Extracted semantic information
    semantic: SemanticInfo = field(default_factory=SemanticInfo)

    # Convenience accessors
    @property
    def functions(self) -> List[FunctionInfo]:
        return self.semantic.functions

    @property
    def classes(self) -> List[ClassInfo]:
        return self.semantic.classes

    @property
    def imports(self) -> List[ImportInfo]:
        return self.semantic.imports

    # Parse quality metrics
    parse_confidence: float = 1.0  # 0.0-1.0 based on error ratio
    total_nodes: int = 0
    error_count: int = 0
    errors: List[ErrorInfo] = field(default_factory=list)

    # Incremental parsing info
    is_incremental: bool = False
    previous_tree_id: Optional[str] = None

    # Source info
    source_bytes: Optional[bytes] = None
    source_hash: Optional[str] = None
    file_path: Optional[str] = None

    def has_errors(self) -> bool:
        """Check if the parse had any errors."""
        return self.error_count > 0

    def is_complete(self) -> bool:
        """Check if the parse completed without significant errors."""
        return self.parse_confidence >= 0.9

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "language": self.language,
            "semantic": self.semantic.to_dict(),
            "parse_confidence": self.parse_confidence,
            "total_nodes": self.total_nodes,
            "error_count": self.error_count,
            "errors": [e.to_dict() for e in self.errors],
            "is_incremental": self.is_incremental,
            "file_path": self.file_path,
        }

    def get_function(self, name: str) -> Optional[FunctionInfo]:
        """Find a function by name."""
        for func in self.functions:
            if func.name == name:
                return func
        return None

    def get_class(self, name: str) -> Optional[ClassInfo]:
        """Find a class by name."""
        for cls in self.classes:
            if cls.name == name:
                return cls
        return None

    def get_import_graph(self) -> Dict[str, List[str]]:
        """Get dependency graph from imports."""
        graph = {}
        for imp in self.imports:
            if imp.module not in graph:
                graph[imp.module] = []
            graph[imp.module].extend(imp.names)
        return graph


# Type alias for language result from detector
# This avoids circular import with language_detector
LanguageResultType = Any
