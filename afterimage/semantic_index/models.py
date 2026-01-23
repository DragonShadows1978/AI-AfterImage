"""
Core data models for semantic intelligence.

Provides unified data structures for:
- Source locations and ranges
- Symbol definitions and references
- Scopes and scope chains
- Type information
- Call sites and hover information
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


class SymbolKind(Enum):
    """Classification of symbol types."""
    VARIABLE = auto()
    FUNCTION = auto()
    CLASS = auto()
    METHOD = auto()
    PARAMETER = auto()
    IMPORT = auto()
    MODULE = auto()
    ATTRIBUTE = auto()
    PROPERTY = auto()
    CONSTANT = auto()
    TYPE_ALIAS = auto()
    ENUM = auto()
    ENUM_MEMBER = auto()
    INTERFACE = auto()
    NAMESPACE = auto()
    UNKNOWN = auto()


class ScopeKind(Enum):
    """Classification of scope types."""
    GLOBAL = auto()
    MODULE = auto()
    CLASS = auto()
    FUNCTION = auto()
    METHOD = auto()
    BLOCK = auto()  # if/for/with/try blocks
    COMPREHENSION = auto()  # list/dict/set comprehensions
    LAMBDA = auto()


@dataclass
class Location:
    """A specific position in source code."""
    file_path: str
    line: int  # 1-indexed
    column: int  # 0-indexed

    def __hash__(self) -> int:
        return hash((self.file_path, self.line, self.column))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Location):
            return False
        return (
            self.file_path == other.file_path
            and self.line == other.line
            and self.column == other.column
        )

    def __lt__(self, other: 'Location') -> bool:
        if self.file_path != other.file_path:
            return self.file_path < other.file_path
        if self.line != other.line:
            return self.line < other.line
        return self.column < other.column

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
        }

    def __repr__(self) -> str:
        return f"{self.file_path}:{self.line}:{self.column}"


@dataclass
class LocationRange:
    """A range of positions in source code."""
    start: Location
    end: Location

    @property
    def file_path(self) -> str:
        return self.start.file_path

    def contains(self, loc: Location) -> bool:
        """Check if a location is within this range."""
        if loc.file_path != self.file_path:
            return False
        if loc.line < self.start.line or loc.line > self.end.line:
            return False
        if loc.line == self.start.line and loc.column < self.start.column:
            return False
        if loc.line == self.end.line and loc.column > self.end.column:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
        }

    def __repr__(self) -> str:
        return f"{self.start} - {self.end}"


@dataclass
class TypeInfo:
    """Type information for a symbol."""
    type_string: str  # The type as a string (e.g., "int", "List[str]")
    is_inferred: bool = False  # Whether the type was inferred vs explicit
    is_generic: bool = False
    type_parameters: List[str] = field(default_factory=list)

    # For callable types
    is_callable: bool = False
    parameter_types: List['TypeInfo'] = field(default_factory=list)
    return_type: Optional['TypeInfo'] = None

    def to_dict(self) -> dict:
        return {
            "type_string": self.type_string,
            "is_inferred": self.is_inferred,
            "is_generic": self.is_generic,
            "type_parameters": self.type_parameters,
            "is_callable": self.is_callable,
        }

    def __repr__(self) -> str:
        return self.type_string


@dataclass
class Symbol:
    """
    A declared symbol (variable, function, class, etc.).

    Symbols are the core unit of semantic analysis - they represent
    named entities that can be defined and referenced.
    """
    name: str
    kind: SymbolKind
    location: LocationRange

    # Fully qualified name (e.g., "mymodule.MyClass.my_method")
    qualified_name: str = ""

    # Type information
    type_info: Optional[TypeInfo] = None

    # Documentation
    docstring: Optional[str] = None

    # Scope information
    scope_id: str = ""  # ID of the scope this symbol is defined in

    # For methods/nested classes - the containing symbol
    parent_symbol: Optional[str] = None  # qualified_name of parent

    # For imports - the source module
    imported_from: Optional[str] = None
    import_alias: Optional[str] = None

    # Visibility
    is_public: bool = True
    is_exported: bool = True

    # For functions/methods
    parameters: List['Symbol'] = field(default_factory=list)
    return_type: Optional[TypeInfo] = None
    is_async: bool = False
    is_generator: bool = False
    decorators: List[str] = field(default_factory=list)

    # For classes
    base_classes: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.qualified_name or (self.name, self.location.start.file_path))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Symbol):
            return False
        if self.qualified_name and other.qualified_name:
            return self.qualified_name == other.qualified_name
        return self.name == other.name and self.location == other.location

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind.name,
            "location": self.location.to_dict(),
            "qualified_name": self.qualified_name,
            "type_info": self.type_info.to_dict() if self.type_info else None,
            "docstring": self.docstring,
            "scope_id": self.scope_id,
            "parent_symbol": self.parent_symbol,
            "is_public": self.is_public,
            "is_async": self.is_async,
            "decorators": self.decorators,
            "base_classes": self.base_classes,
        }

    @property
    def signature(self) -> str:
        """Generate a human-readable signature."""
        if self.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            params = ", ".join(
                f"{p.name}: {p.type_info}" if p.type_info else p.name
                for p in self.parameters
            )
            sig = f"{self.name}({params})"
            if self.return_type:
                sig += f" -> {self.return_type}"
            if self.is_async:
                sig = f"async {sig}"
            return sig
        elif self.kind == SymbolKind.CLASS:
            bases = f"({', '.join(self.base_classes)})" if self.base_classes else ""
            return f"class {self.name}{bases}"
        elif self.type_info:
            return f"{self.name}: {self.type_info}"
        return self.name


@dataclass
class Reference:
    """
    A reference to a symbol.

    References are usages of symbols - places where a symbol is
    accessed, called, or otherwise used.
    """
    name: str  # The referenced name as it appears in code
    location: LocationRange

    # The symbol being referenced (resolved during analysis)
    resolved_symbol: Optional[Symbol] = None
    resolved_qualified_name: Optional[str] = None

    # Context about how the reference is used
    is_read: bool = True
    is_write: bool = False
    is_call: bool = False
    is_import: bool = False
    is_type_annotation: bool = False

    # For attribute access (e.g., obj.attr)
    receiver_name: Optional[str] = None  # "obj" in obj.attr

    # Scope where the reference occurs
    scope_id: str = ""

    def __hash__(self) -> int:
        return hash((self.name, self.location.start.file_path,
                     self.location.start.line, self.location.start.column))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "location": self.location.to_dict(),
            "resolved_qualified_name": self.resolved_qualified_name,
            "is_read": self.is_read,
            "is_write": self.is_write,
            "is_call": self.is_call,
            "scope_id": self.scope_id,
        }


@dataclass
class Scope:
    """
    A lexical scope in the source code.

    Scopes form a tree structure representing the nesting of
    namespaces, classes, functions, and blocks.
    """
    id: str  # Unique identifier for this scope
    kind: ScopeKind
    name: str  # Name of the scope (function name, class name, etc.)
    range: LocationRange

    # Parent scope (None for global scope)
    parent_id: Optional[str] = None

    # Symbols defined directly in this scope
    symbols: Dict[str, Symbol] = field(default_factory=dict)

    # Child scopes
    children: List[str] = field(default_factory=list)  # child scope IDs

    # For class scopes - the class symbol
    class_symbol: Optional[str] = None  # qualified_name

    # For function scopes - the function symbol
    function_symbol: Optional[str] = None  # qualified_name

    def add_symbol(self, symbol: Symbol) -> None:
        """Add a symbol to this scope."""
        self.symbols[symbol.name] = symbol
        symbol.scope_id = self.id

    def get_symbol(self, name: str) -> Optional[Symbol]:
        """Get a symbol by name from this scope only."""
        return self.symbols.get(name)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind.name,
            "name": self.name,
            "range": self.range.to_dict(),
            "parent_id": self.parent_id,
            "symbols": {k: v.to_dict() for k, v in self.symbols.items()},
            "children": self.children,
        }


@dataclass
class CallSite:
    """
    A function/method call site.

    Used to build the call graph showing which functions
    call which other functions.
    """
    location: LocationRange

    # The caller (function containing this call)
    caller_qualified_name: str

    # The callee (function being called)
    callee_name: str  # As it appears in code
    callee_qualified_name: Optional[str] = None  # Resolved

    # Arguments
    argument_count: int = 0
    has_keyword_args: bool = False
    has_star_args: bool = False

    # For method calls
    receiver_type: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "location": self.location.to_dict(),
            "caller": self.caller_qualified_name,
            "callee": self.callee_name,
            "callee_resolved": self.callee_qualified_name,
            "argument_count": self.argument_count,
        }


@dataclass
class HoverInfo:
    """
    Information to display when hovering over a symbol.

    Provides a rich preview of what a symbol is and how it's defined.
    """
    symbol: Symbol

    # Formatted display
    title: str  # e.g., "function greet(name: str) -> str"
    description: str  # Docstring or generated description

    # Definition location
    definition_location: LocationRange

    # Additional context
    module_name: Optional[str] = None
    containing_class: Optional[str] = None

    # Type information
    inferred_type: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "definition_location": self.definition_location.to_dict(),
            "module_name": self.module_name,
            "containing_class": self.containing_class,
            "inferred_type": self.inferred_type,
        }

    def format_markdown(self) -> str:
        """Format hover info as markdown."""
        lines = [f"```\n{self.title}\n```"]

        if self.module_name:
            lines.append(f"*Module: {self.module_name}*")

        if self.containing_class:
            lines.append(f"*Class: {self.containing_class}*")

        if self.description:
            lines.append("")
            lines.append(self.description)

        return "\n".join(lines)


@dataclass
class DefinitionResult:
    """Result of a go-to-definition query."""
    success: bool
    symbol: Optional[Symbol] = None
    location: Optional[LocationRange] = None
    error: Optional[str] = None

    # For symbols that have multiple definitions (e.g., overloads)
    additional_locations: List[LocationRange] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "symbol": self.symbol.to_dict() if self.symbol else None,
            "location": self.location.to_dict() if self.location else None,
            "error": self.error,
            "additional_locations": [loc.to_dict() for loc in self.additional_locations],
        }


@dataclass
class ReferenceResult:
    """Result of a find-all-references query."""
    success: bool
    symbol: Optional[Symbol] = None
    references: List[Reference] = field(default_factory=list)
    error: Optional[str] = None

    # Separate lists for different reference types
    read_references: List[Reference] = field(default_factory=list)
    write_references: List[Reference] = field(default_factory=list)
    call_references: List[Reference] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return len(self.references)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "symbol": self.symbol.to_dict() if self.symbol else None,
            "total_references": self.total_count,
            "references": [r.to_dict() for r in self.references],
            "error": self.error,
        }
