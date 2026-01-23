"""
Definition Resolver - Go to definition implementation.

Resolves symbol references to their declaration locations,
handling imports and cross-file module boundaries.

Enhanced features:
- Star import expansion
- Complex relative import resolution
- Re-export tracking
- External package marking
"""

from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

from .models import (
    Location,
    Symbol,
    SymbolKind,
    DefinitionResult,
)
from .symbol_table import SymbolTable
from .import_graph import ImportGraph

if TYPE_CHECKING:
    from .performance import DefinitionCache


class DefinitionResolver:
    """
    Resolves symbol references to their definitions.

    Supports:
    - Local symbol resolution within a file
    - Cross-file resolution via imports
    - Module-level symbol lookup
    - Star import expansion
    - Re-export resolution
    - External package handling
    """

    def __init__(
        self,
        import_graph: Optional[ImportGraph] = None,
        definition_cache: Optional['DefinitionCache'] = None,
    ):
        # Map of file_path -> SymbolTable
        self.symbol_tables: Dict[str, SymbolTable] = {}

        # Map of module_name -> file_path
        self.module_paths: Dict[str, str] = {}

        # Map of module_name -> exported symbols
        self.exports: Dict[str, Dict[str, Symbol]] = {}

        # Import graph for advanced resolution
        self.import_graph = import_graph or ImportGraph()

        # Cache for resolved imports (key: (from_module, name))
        self._resolution_cache: Dict[Tuple[str, str], Optional[Symbol]] = {}

        # External package markers
        self._external_packages: Set[str] = set()

        # Optional external DefinitionCache for import resolution
        self._definition_cache = definition_cache

    def register_file(
        self,
        file_path: str,
        symbol_table: SymbolTable,
        module_name: Optional[str] = None,
    ) -> None:
        """Register a file's symbol table for resolution."""
        self.symbol_tables[file_path] = symbol_table

        if module_name:
            self.module_paths[module_name] = file_path
            # Build exports map
            self.exports[module_name] = {
                sym.name: sym
                for sym in symbol_table.get_exported_symbols()
            }

    def unregister_file(self, file_path: str) -> None:
        """Remove a file from the resolver."""
        if file_path in self.symbol_tables:
            table = self.symbol_tables[file_path]
            # Remove from module paths
            if table.module_name and table.module_name in self.module_paths:
                del self.module_paths[table.module_name]
                if table.module_name in self.exports:
                    del self.exports[table.module_name]
            del self.symbol_tables[file_path]

    def go_to_definition(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> DefinitionResult:
        """
        Find the definition of the symbol at the given location.

        Args:
            file_path: Path to the source file
            line: Line number (1-indexed)
            column: Column number (0-indexed)

        Returns:
            DefinitionResult with the definition location
        """
        # Get the symbol table for this file
        symbol_table = self.symbol_tables.get(file_path)
        if not symbol_table:
            return DefinitionResult(
                success=False,
                error=f"File not indexed: {file_path}"
            )

        # Find what's at this location
        entity = symbol_table.get_entity_at(line, column)
        if not entity:
            return DefinitionResult(
                success=False,
                error=f"No symbol found at {file_path}:{line}:{column}"
            )

        entity_type, entity_obj = entity

        # If it's already a definition, return it
        if entity_type == "symbol":
            symbol = entity_obj
            return DefinitionResult(
                success=True,
                symbol=symbol,
                location=symbol.location,
            )

        # It's a reference - resolve it
        reference = entity_obj

        # First check if reference is already resolved
        if reference.resolved_symbol:
            return DefinitionResult(
                success=True,
                symbol=reference.resolved_symbol,
                location=reference.resolved_symbol.location,
            )

        # Try to resolve locally first
        scope = symbol_table.get_scope_at(line, column)
        resolved = symbol_table.resolve_name(
            reference.name,
            from_scope_id=scope.id if scope else None,
        )

        if resolved:
            # Check if it's an import that needs cross-file resolution
            if resolved.kind == SymbolKind.IMPORT:
                cross_file_result = self._resolve_import(resolved)
                if cross_file_result.success:
                    return cross_file_result
                # Fall back to the import symbol itself
                return DefinitionResult(
                    success=True,
                    symbol=resolved,
                    location=resolved.location,
                )

            return DefinitionResult(
                success=True,
                symbol=resolved,
                location=resolved.location,
            )

        # Try cross-file resolution for attribute access
        if reference.receiver_name:
            cross_result = self._resolve_attribute(
                reference.receiver_name,
                reference.name,
                symbol_table,
                scope.id if scope else None,
            )
            if cross_result.success:
                return cross_result

        return DefinitionResult(
            success=False,
            error=f"Cannot resolve '{reference.name}'"
        )

    def _resolve_import(self, import_symbol: Symbol) -> DefinitionResult:
        """Resolve an import to its source definition."""
        imported_from = import_symbol.imported_from
        if not imported_from:
            return DefinitionResult(
                success=False,
                error="Import has no source module"
            )

        # Check external cache first
        if self._definition_cache:
            cached = self._definition_cache.get_import(imported_from, import_symbol.name)
            if cached is not None:
                return cached

        # Parse the import path
        parts = imported_from.split(".")

        # Try to find the module
        module_name = parts[0]
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in self.module_paths:
                module_name = candidate
                remaining_parts = parts[i:]
                break
        else:
            remaining_parts = parts[1:] if len(parts) > 1 else []

        # Get the module's symbol table
        module_path = self.module_paths.get(module_name)
        if not module_path:
            return DefinitionResult(
                success=False,
                error=f"Module not found: {module_name}"
            )

        module_table = self.symbol_tables.get(module_path)
        if not module_table:
            return DefinitionResult(
                success=False,
                error=f"Module not indexed: {module_name}"
            )

        # If we're importing the module itself
        if not remaining_parts:
            # Return the first symbol or module-level info
            for sym in module_table.get_all_symbols():
                if sym.kind not in (SymbolKind.IMPORT, SymbolKind.PARAMETER):
                    return DefinitionResult(
                        success=True,
                        symbol=sym,
                        location=sym.location,
                    )
            return DefinitionResult(
                success=False,
                error=f"No definition found in {module_name}"
            )

        # Look up the specific name in the module
        target_name = remaining_parts[-1]
        exports = self.exports.get(module_name, {})

        if target_name in exports:
            symbol = exports[target_name]
            return DefinitionResult(
                success=True,
                symbol=symbol,
                location=symbol.location,
            )

        # Try direct lookup
        symbols = module_table.get_symbols_by_name(target_name)
        if symbols:
            symbol = symbols[0]
            return DefinitionResult(
                success=True,
                symbol=symbol,
                location=symbol.location,
            )

        return DefinitionResult(
            success=False,
            error=f"'{target_name}' not found in {module_name}"
        )

    def _resolve_attribute(
        self,
        receiver_name: str,
        attr_name: str,
        symbol_table: SymbolTable,
        scope_id: Optional[str],
    ) -> DefinitionResult:
        """Resolve an attribute access (receiver.attr)."""
        # First resolve the receiver
        receiver = symbol_table.resolve_name(receiver_name, scope_id)
        if not receiver:
            return DefinitionResult(
                success=False,
                error=f"Cannot resolve receiver '{receiver_name}'"
            )

        # If receiver is an import, look in the imported module
        if receiver.kind == SymbolKind.IMPORT:
            module_name = receiver.imported_from
            if module_name in self.module_paths:
                module_path = self.module_paths[module_name]
                module_table = self.symbol_tables.get(module_path)
                if module_table:
                    symbols = module_table.get_symbols_by_name(attr_name)
                    if symbols:
                        return DefinitionResult(
                            success=True,
                            symbol=symbols[0],
                            location=symbols[0].location,
                        )

        # If receiver is a class, look for methods/attributes
        if receiver.kind == SymbolKind.CLASS:
            # Look in class scope for the attribute
            for scope in symbol_table.scopes.values():
                if scope.class_symbol == receiver.qualified_name:
                    symbol = scope.get_symbol(attr_name)
                    if symbol:
                        return DefinitionResult(
                            success=True,
                            symbol=symbol,
                            location=symbol.location,
                        )

        return DefinitionResult(
            success=False,
            error=f"Cannot resolve '{receiver_name}.{attr_name}'"
        )

    def resolve_qualified_name(self, qualified_name: str) -> DefinitionResult:
        """Resolve a fully qualified name to its definition."""
        # Try direct lookup in all tables
        for table in self.symbol_tables.values():
            symbol = table.get_symbol(qualified_name)
            if symbol:
                return DefinitionResult(
                    success=True,
                    symbol=symbol,
                    location=symbol.location,
                )

        # Try parsing as module.name
        parts = qualified_name.rsplit(".", 1)
        if len(parts) == 2:
            module_name, symbol_name = parts
            if module_name in self.exports:
                if symbol_name in self.exports[module_name]:
                    symbol = self.exports[module_name][symbol_name]
                    return DefinitionResult(
                        success=True,
                        symbol=symbol,
                        location=symbol.location,
                    )

        return DefinitionResult(
            success=False,
            error=f"Cannot resolve '{qualified_name}'"
        )

    def resolve_star_import(
        self,
        name: str,
        from_module: str,
    ) -> DefinitionResult:
        """
        Resolve a name that might come from a star import.

        Args:
            name: The name to resolve
            from_module: The module that has the star import

        Returns:
            DefinitionResult with the resolved symbol
        """
        # Check cache first
        cache_key = (from_module, name)
        if cache_key in self._resolution_cache:
            cached = self._resolution_cache[cache_key]
            if cached:
                return DefinitionResult(success=True, symbol=cached, location=cached.location)
            return DefinitionResult(success=False, error=f"'{name}' not found via star import")

        # Get the symbol table for this module
        if from_module not in self.module_paths:
            return DefinitionResult(success=False, error=f"Module not found: {from_module}")

        file_path = self.module_paths[from_module]
        symbol_table = self.symbol_tables.get(file_path)
        if not symbol_table:
            return DefinitionResult(success=False, error=f"Symbol table not found: {from_module}")

        # Look for star imports in this module
        for imp_symbol in symbol_table.imports:
            if imp_symbol.metadata.get("star_import"):
                source_module = imp_symbol.imported_from
                if source_module:
                    # Expand the star import and look for the name
                    expanded = self.import_graph.expand_star_import(source_module, from_module)
                    if name in expanded:
                        symbol = expanded[name]
                        self._resolution_cache[cache_key] = symbol
                        return DefinitionResult(success=True, symbol=symbol, location=symbol.location)

        self._resolution_cache[cache_key] = None
        return DefinitionResult(success=False, error=f"'{name}' not found via star import")

    def resolve_with_re_exports(
        self,
        name: str,
        module_name: str,
    ) -> DefinitionResult:
        """
        Resolve a name considering re-exports.

        Args:
            name: The name to resolve
            module_name: The module to search in

        Returns:
            DefinitionResult with the resolved symbol
        """
        # Try direct resolution first
        if module_name in self.exports and name in self.exports[module_name]:
            symbol = self.exports[module_name][name]
            return DefinitionResult(success=True, symbol=symbol, location=symbol.location)

        # Try re-export resolution via import graph
        result = self.import_graph.resolve_re_export(module_name, name)
        if result:
            source_module, symbol = result
            return DefinitionResult(success=True, symbol=symbol, location=symbol.location)

        return DefinitionResult(success=False, error=f"'{name}' not found in {module_name}")

    def resolve_external_import(
        self,
        import_symbol: Symbol,
    ) -> DefinitionResult:
        """
        Handle resolution for external package imports.

        Args:
            import_symbol: The import symbol to resolve

        Returns:
            DefinitionResult indicating external package status
        """
        imported_from = import_symbol.imported_from
        if not imported_from:
            return DefinitionResult(success=False, error="Import has no source module")

        # Check if it's an external package
        root_module = imported_from.split('.')[0]
        if self.import_graph.is_external_package(root_module):
            self._external_packages.add(root_module)
            self.import_graph.mark_external_package(root_module)
            # Return the import symbol itself with a note that it's external
            return DefinitionResult(
                success=True,
                symbol=import_symbol,
                location=import_symbol.location,
                error="external_package",  # Use error field as status indicator
            )

        return DefinitionResult(success=False, error=f"Module not found: {imported_from}")

    def clear_cache(self) -> None:
        """Clear the resolution cache."""
        self._resolution_cache.clear()

    def is_external_package(self, module_name: str) -> bool:
        """Check if a module is an external package."""
        return module_name in self._external_packages or self.import_graph.is_external_package(module_name)
