"""
References Finder - Find all references to a symbol.

Locates every usage of a symbol across the codebase.
"""

from typing import Dict, List, Optional

from .models import (
    Symbol,
    Reference,
    ReferenceResult,
)
from .symbol_table import SymbolTable


class ReferencesFinder:
    """
    Finds all references to a symbol across the codebase.

    Supports:
    - Local references within a file
    - Cross-file references via exports
    - Categorization of read/write/call references
    """

    def __init__(self):
        # Map of file_path -> SymbolTable
        self.symbol_tables: Dict[str, SymbolTable] = {}

        # Map of qualified_name -> list of references
        self._reference_cache: Dict[str, List[Reference]] = {}

    def register_file(
        self,
        file_path: str,
        symbol_table: SymbolTable,
    ) -> None:
        """Register a file's symbol table for reference search."""
        self.symbol_tables[file_path] = symbol_table
        # Invalidate cache when new file is registered
        self._reference_cache.clear()

    def unregister_file(self, file_path: str) -> None:
        """Remove a file from the finder."""
        if file_path in self.symbol_tables:
            del self.symbol_tables[file_path]
            self._reference_cache.clear()

    def find_references(
        self,
        file_path: str,
        line: int,
        column: int,
        include_declaration: bool = True,
    ) -> ReferenceResult:
        """
        Find all references to the symbol at the given location.

        Args:
            file_path: Path to the source file
            line: Line number (1-indexed)
            column: Column number (0-indexed)
            include_declaration: Whether to include the declaration itself

        Returns:
            ReferenceResult with all found references
        """
        # Get the symbol table for this file
        symbol_table = self.symbol_tables.get(file_path)
        if not symbol_table:
            return ReferenceResult(
                success=False,
                error=f"File not indexed: {file_path}"
            )

        # Find what's at this location
        entity = symbol_table.get_entity_at(line, column)
        if not entity:
            return ReferenceResult(
                success=False,
                error=f"No symbol found at {file_path}:{line}:{column}"
            )

        entity_type, entity_obj = entity

        # Get the symbol we're finding references for
        if entity_type == "symbol":
            symbol = entity_obj
        else:
            # It's a reference - get its resolved symbol
            reference = entity_obj
            if reference.resolved_symbol:
                symbol = reference.resolved_symbol
            else:
                # Try to find symbol by name
                symbols = symbol_table.get_symbols_by_name(reference.name)
                if symbols:
                    symbol = symbols[0]
                else:
                    return ReferenceResult(
                        success=False,
                        error=f"Cannot resolve '{reference.name}'"
                    )

        # Find all references to this symbol
        return self.find_references_to_symbol(symbol, include_declaration)

    def find_references_to_symbol(
        self,
        symbol: Symbol,
        include_declaration: bool = True,
    ) -> ReferenceResult:
        """
        Find all references to a specific symbol.

        Args:
            symbol: The symbol to find references for
            include_declaration: Whether to include the declaration itself

        Returns:
            ReferenceResult with all found references
        """
        qualified_name = symbol.qualified_name
        all_refs: List[Reference] = []
        read_refs: List[Reference] = []
        write_refs: List[Reference] = []
        call_refs: List[Reference] = []

        # Search all registered files
        for file_path, table in self.symbol_tables.items():
            # Get references that match this symbol
            refs = self._find_refs_in_table(table, symbol)
            for ref in refs:
                all_refs.append(ref)
                if ref.is_call:
                    call_refs.append(ref)
                elif ref.is_write:
                    write_refs.append(ref)
                else:
                    read_refs.append(ref)

        # Include the declaration itself if requested
        if include_declaration:
            # Create a "reference" for the declaration
            decl_ref = Reference(
                name=symbol.name,
                location=symbol.location,
                resolved_symbol=symbol,
                resolved_qualified_name=qualified_name,
                is_read=False,
                is_write=True,  # Declaration is a "write"
                scope_id=symbol.scope_id,
            )
            all_refs.insert(0, decl_ref)
            write_refs.insert(0, decl_ref)

        # Sort references by location
        all_refs.sort(key=lambda r: (r.location.file_path, r.location.start.line))

        return ReferenceResult(
            success=True,
            symbol=symbol,
            references=all_refs,
            read_references=read_refs,
            write_references=write_refs,
            call_references=call_refs,
        )

    def _find_refs_in_table(
        self,
        table: SymbolTable,
        symbol: Symbol,
    ) -> List[Reference]:
        """Find references to a symbol in a specific symbol table."""
        refs = []

        # Check by qualified name
        for ref in table.references:
            if ref.resolved_qualified_name == symbol.qualified_name:
                refs.append(ref)
                continue

            # Check by simple name match (for unresolved references)
            if ref.name == symbol.name and not ref.resolved_qualified_name:
                # Might be a match - could do additional checking here
                refs.append(ref)

        return refs

    def find_all_references_by_name(
        self,
        name: str,
        file_path: Optional[str] = None,
    ) -> List[Reference]:
        """
        Find all references to symbols with a given name.

        Args:
            name: The simple name to search for
            file_path: Optionally limit search to specific file

        Returns:
            List of all matching references
        """
        refs = []

        tables = (
            {file_path: self.symbol_tables[file_path]}
            if file_path and file_path in self.symbol_tables
            else self.symbol_tables
        )

        for table in tables.values():
            refs.extend(table.references_by_name.get(name, []))

        return refs

    def get_reference_count(self, symbol: Symbol) -> int:
        """Get the number of references to a symbol."""
        count = 0
        for table in self.symbol_tables.values():
            for ref in table.references:
                if ref.resolved_qualified_name == symbol.qualified_name:
                    count += 1
        return count

    def find_unused_symbols(self, file_path: str) -> List[Symbol]:
        """
        Find symbols in a file that have no references.

        Useful for finding dead code.
        """
        table = self.symbol_tables.get(file_path)
        if not table:
            return []

        unused = []
        for symbol in table.symbols.values():
            # Skip imports and parameters
            if symbol.kind.name in ("IMPORT", "PARAMETER"):
                continue

            # Check if there are any references
            ref_count = self.get_reference_count(symbol)
            if ref_count == 0:
                unused.append(symbol)

        return unused
