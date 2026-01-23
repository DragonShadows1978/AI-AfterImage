"""
SymbolTable - Per-file symbol storage with scope chain awareness.

Provides:
- Hierarchical scope management
- Symbol lookup with scope chain traversal
- Name resolution following Python's LEGB rule
- Support for incremental updates
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set, Tuple
import uuid

from .models import (
    Location,
    LocationRange,
    Symbol,
    SymbolKind,
    Reference,
    Scope,
    ScopeKind,
)


@dataclass
class SymbolTable:
    """
    Per-file symbol table with scope chain tracking.

    The symbol table maintains:
    - All symbols declared in a file
    - All references to symbols in a file
    - Scope hierarchy for name resolution
    - Mapping from locations to symbols/references
    """
    file_path: str

    # All symbols indexed by qualified name
    symbols: Dict[str, Symbol] = field(default_factory=dict)

    # Symbols indexed by simple name (for lookup)
    symbols_by_name: Dict[str, List[Symbol]] = field(default_factory=lambda: defaultdict(list))

    # All references in this file
    references: List[Reference] = field(default_factory=list)

    # References indexed by the name they reference
    references_by_name: Dict[str, List[Reference]] = field(default_factory=lambda: defaultdict(list))

    # Scope hierarchy
    scopes: Dict[str, Scope] = field(default_factory=dict)
    root_scope_id: Optional[str] = None

    # Location-based indices for quick lookup
    _symbol_at_location: Dict[Tuple[int, int], Symbol] = field(default_factory=dict)
    _reference_at_location: Dict[Tuple[int, int], Reference] = field(default_factory=dict)

    # Module-level information
    module_name: str = ""
    imports: List[Symbol] = field(default_factory=list)

    # Source hash for change detection
    source_hash: str = ""

    def __post_init__(self):
        """Initialize the root scope."""
        if self.root_scope_id is None:
            self._create_root_scope()

    def _create_root_scope(self) -> Scope:
        """Create the module-level root scope."""
        scope_id = f"scope_{uuid.uuid4().hex[:8]}"
        root_scope = Scope(
            id=scope_id,
            kind=ScopeKind.MODULE,
            name=self.module_name or self.file_path,
            range=LocationRange(
                start=Location(self.file_path, 1, 0),
                end=Location(self.file_path, 999999, 0),
            ),
            parent_id=None,
        )
        self.scopes[scope_id] = root_scope
        self.root_scope_id = scope_id
        return root_scope

    def create_scope(
        self,
        kind: ScopeKind,
        name: str,
        range: LocationRange,
        parent_id: Optional[str] = None,
    ) -> Scope:
        """Create a new scope within the symbol table."""
        scope_id = f"scope_{uuid.uuid4().hex[:8]}"
        scope = Scope(
            id=scope_id,
            kind=kind,
            name=name,
            range=range,
            parent_id=parent_id or self.root_scope_id,
        )
        self.scopes[scope_id] = scope

        # Add to parent's children
        if scope.parent_id and scope.parent_id in self.scopes:
            self.scopes[scope.parent_id].children.append(scope_id)

        return scope

    def add_symbol(self, symbol: Symbol, scope_id: Optional[str] = None) -> None:
        """Add a symbol to the table."""
        # Determine the scope to add to
        target_scope_id = scope_id or symbol.scope_id or self.root_scope_id

        # Update symbol's scope_id
        symbol.scope_id = target_scope_id

        # Add to scope's symbol dictionary
        if target_scope_id and target_scope_id in self.scopes:
            self.scopes[target_scope_id].add_symbol(symbol)

        # Generate qualified name if not set
        if not symbol.qualified_name:
            symbol.qualified_name = self._generate_qualified_name(symbol)

        # Add to indices
        self.symbols[symbol.qualified_name] = symbol
        self.symbols_by_name[symbol.name].append(symbol)

        # Index by location for hover/definition lookup
        loc_key = (symbol.location.start.line, symbol.location.start.column)
        self._symbol_at_location[loc_key] = symbol

        # Track imports separately
        if symbol.kind == SymbolKind.IMPORT:
            self.imports.append(symbol)

    def add_reference(self, reference: Reference) -> None:
        """Add a reference to the table."""
        self.references.append(reference)
        self.references_by_name[reference.name].append(reference)

        # Index by location
        loc_key = (reference.location.start.line, reference.location.start.column)
        self._reference_at_location[loc_key] = reference

    def _generate_qualified_name(self, symbol: Symbol) -> str:
        """Generate a fully qualified name for a symbol."""
        parts = []

        # Add module prefix
        if self.module_name:
            parts.append(self.module_name)

        # Traverse scope chain to build qualified name
        if symbol.scope_id and symbol.scope_id in self.scopes:
            scope = self.scopes[symbol.scope_id]
            scope_parts = []

            while scope and scope.kind not in (ScopeKind.MODULE, ScopeKind.GLOBAL):
                if scope.name:
                    scope_parts.append(scope.name)
                if scope.parent_id:
                    scope = self.scopes.get(scope.parent_id)
                else:
                    break

            parts.extend(reversed(scope_parts))

        parts.append(symbol.name)
        return ".".join(parts)

    def get_symbol(self, qualified_name: str) -> Optional[Symbol]:
        """Get a symbol by its fully qualified name."""
        return self.symbols.get(qualified_name)

    def get_symbols_by_name(self, name: str) -> List[Symbol]:
        """Get all symbols with a given simple name."""
        return self.symbols_by_name.get(name, [])

    def get_symbol_at(self, line: int, column: int) -> Optional[Symbol]:
        """Get the symbol defined at a specific location."""
        return self._symbol_at_location.get((line, column))

    def get_reference_at(self, line: int, column: int) -> Optional[Reference]:
        """Get the reference at a specific location."""
        return self._reference_at_location.get((line, column))

    def get_entity_at(self, line: int, column: int) -> Optional[Tuple[str, object]]:
        """Get either a symbol or reference at a location."""
        sym = self.get_symbol_at(line, column)
        if sym:
            return ("symbol", sym)

        ref = self.get_reference_at(line, column)
        if ref:
            return ("reference", ref)

        # Search for containing ranges
        for ref in self.references:
            if ref.location.contains(Location(self.file_path, line, column)):
                return ("reference", ref)

        for sym in self.symbols.values():
            if sym.location.contains(Location(self.file_path, line, column)):
                return ("symbol", sym)

        return None

    def resolve_name(
        self,
        name: str,
        from_scope_id: Optional[str] = None,
        follow_imports: bool = True
    ) -> Optional[Symbol]:
        """
        Resolve a name to a symbol following Python's LEGB rule.

        LEGB: Local -> Enclosing -> Global -> Built-in

        Args:
            name: The name to resolve
            from_scope_id: The scope to start resolution from
            follow_imports: Whether to resolve imported names

        Returns:
            The resolved symbol, or None if not found
        """
        # Start from provided scope or root
        scope_id = from_scope_id or self.root_scope_id
        if not scope_id:
            return None

        # Walk up the scope chain
        while scope_id:
            scope = self.scopes.get(scope_id)
            if not scope:
                break

            # Check this scope
            symbol = scope.get_symbol(name)
            if symbol:
                return symbol

            # Move to parent scope
            scope_id = scope.parent_id

        # Check imports if not found locally
        if follow_imports:
            for imp in self.imports:
                if imp.name == name or imp.import_alias == name:
                    return imp
                # For "from x import y" style imports
                if name in (imp.metadata.get("imported_names") or []):
                    return imp

        return None

    def get_scope_at(self, line: int, column: int) -> Optional[Scope]:
        """Find the innermost scope containing a location."""
        loc = Location(self.file_path, line, column)

        # Find all containing scopes
        containing = []
        for scope in self.scopes.values():
            if scope.range.contains(loc):
                containing.append(scope)

        if not containing:
            return self.scopes.get(self.root_scope_id)

        # Return the innermost (smallest range)
        return min(
            containing,
            key=lambda s: (
                s.range.end.line - s.range.start.line,
                s.range.end.column - s.range.start.column
            )
        )

    def get_references_to(self, symbol: Symbol) -> List[Reference]:
        """Get all references to a symbol."""
        refs = []
        for ref in self.references:
            if ref.resolved_qualified_name == symbol.qualified_name:
                refs.append(ref)
            elif ref.resolved_symbol == symbol:
                refs.append(ref)
        return refs

    def get_all_symbols(self, kind: Optional[SymbolKind] = None) -> Iterator[Symbol]:
        """Iterate over all symbols, optionally filtered by kind."""
        for symbol in self.symbols.values():
            if kind is None or symbol.kind == kind:
                yield symbol

    def get_exported_symbols(self) -> List[Symbol]:
        """Get symbols that are exported (public API)."""
        return [s for s in self.symbols.values() if s.is_exported and s.is_public]

    def clear(self) -> None:
        """Clear all data for incremental update."""
        self.symbols.clear()
        self.symbols_by_name.clear()
        self.references.clear()
        self.references_by_name.clear()
        self._symbol_at_location.clear()
        self._reference_at_location.clear()
        self.imports.clear()

        # Keep root scope but clear its symbols
        if self.root_scope_id and self.root_scope_id in self.scopes:
            root = self.scopes[self.root_scope_id]
            root.symbols.clear()
            root.children.clear()

        # Remove all non-root scopes
        self.scopes = {k: v for k, v in self.scopes.items() if k == self.root_scope_id}

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "file_path": self.file_path,
            "module_name": self.module_name,
            "symbols": {k: v.to_dict() for k, v in self.symbols.items()},
            "references": [r.to_dict() for r in self.references],
            "scopes": {k: v.to_dict() for k, v in self.scopes.items()},
            "source_hash": self.source_hash,
            "import_count": len(self.imports),
            "symbol_count": len(self.symbols),
            "reference_count": len(self.references),
        }

    def __repr__(self) -> str:
        return (
            f"SymbolTable({self.file_path}, "
            f"symbols={len(self.symbols)}, "
            f"references={len(self.references)}, "
            f"scopes={len(self.scopes)})"
        )
