"""
Hover Provider - Provides hover information for symbols.

Shows type signatures, documentation, and declaration context.
Enhanced with type inference cache integration.
"""

from typing import Dict, Optional, TYPE_CHECKING

from .models import (
    Symbol,
    SymbolKind,
    HoverInfo,
    TypeInfo,
)
from .symbol_table import SymbolTable

if TYPE_CHECKING:
    from .type_inference import TypeInferencer


class HoverProvider:
    """
    Provides hover information for symbols.

    Shows:
    - Type signatures
    - Documentation strings
    - Declaration context (module, class)
    - Inferred types from type inference cache
    """

    def __init__(self, type_inferencer: Optional['TypeInferencer'] = None):
        # Map of file_path -> SymbolTable
        self.symbol_tables: Dict[str, SymbolTable] = {}
        # Optional type inferencer for cache lookups
        self._type_inferencer = type_inferencer

    def register_file(
        self,
        file_path: str,
        symbol_table: SymbolTable,
    ) -> None:
        """Register a file's symbol table for hover info."""
        self.symbol_tables[file_path] = symbol_table

    def unregister_file(self, file_path: str) -> None:
        """Remove a file from the provider."""
        if file_path in self.symbol_tables:
            del self.symbol_tables[file_path]

    def get_hover(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> Optional[HoverInfo]:
        """
        Get hover information for the symbol at the given location.

        Args:
            file_path: Path to the source file
            line: Line number (1-indexed)
            column: Column number (0-indexed)

        Returns:
            HoverInfo if a symbol is found, None otherwise
        """
        symbol_table = self.symbol_tables.get(file_path)
        if not symbol_table:
            return None

        # Find what's at this location
        entity = symbol_table.get_entity_at(line, column)
        if not entity:
            return None

        entity_type, entity_obj = entity

        # Get the symbol
        if entity_type == "symbol":
            symbol = entity_obj
        else:
            # It's a reference - get its resolved symbol
            reference = entity_obj
            if reference.resolved_symbol:
                symbol = reference.resolved_symbol
            else:
                # Try to find by name
                symbols = symbol_table.get_symbols_by_name(reference.name)
                if symbols:
                    symbol = symbols[0]
                else:
                    return None

        return self._build_hover_info(symbol, symbol_table)

    def _build_hover_info(
        self,
        symbol: Symbol,
        symbol_table: SymbolTable,
    ) -> HoverInfo:
        """Build hover information for a symbol."""
        # Generate title based on symbol kind
        title = self._generate_title(symbol)

        # Get description (docstring or generated)
        description = symbol.docstring or self._generate_description(symbol)

        # Get context
        module_name = symbol_table.module_name or None
        containing_class = None
        if symbol.parent_symbol:
            # Extract class name from qualified name
            parts = symbol.parent_symbol.split(".")
            if parts:
                containing_class = parts[-1]

        # Get inferred type if available
        inferred_type = None
        if symbol.type_info:
            inferred_type = symbol.type_info.type_string
        elif self._type_inferencer and symbol.qualified_name:
            # Fall back to type inference cache
            cached_type = self._type_inferencer.get_cached_type(symbol.qualified_name)
            if cached_type:
                inferred_type = cached_type.type_string

        return HoverInfo(
            symbol=symbol,
            title=title,
            description=description,
            definition_location=symbol.location,
            module_name=module_name,
            containing_class=containing_class,
            inferred_type=inferred_type,
        )

    def _generate_title(self, symbol: Symbol) -> str:
        """Generate a title/signature for the symbol."""
        kind = symbol.kind

        if kind == SymbolKind.FUNCTION:
            return f"def {symbol.signature}"

        elif kind == SymbolKind.METHOD:
            prefix = "async def " if symbol.is_async else "def "
            return f"{prefix}{symbol.signature}"

        elif kind == SymbolKind.CLASS:
            bases = f"({', '.join(symbol.base_classes)})" if symbol.base_classes else ""
            return f"class {symbol.name}{bases}"

        elif kind == SymbolKind.VARIABLE:
            if symbol.type_info:
                return f"{symbol.name}: {symbol.type_info.type_string}"
            return f"{symbol.name}"

        elif kind == SymbolKind.PARAMETER:
            if symbol.type_info:
                return f"(parameter) {symbol.name}: {symbol.type_info.type_string}"
            return f"(parameter) {symbol.name}"

        elif kind == SymbolKind.IMPORT:
            if symbol.import_alias:
                return f"import {symbol.imported_from} as {symbol.import_alias}"
            elif symbol.imported_from:
                return f"from {symbol.imported_from.rsplit('.', 1)[0]} import {symbol.name}"
            return f"import {symbol.name}"

        elif kind == SymbolKind.CONSTANT:
            if symbol.type_info:
                return f"(constant) {symbol.name}: {symbol.type_info.type_string}"
            return f"(constant) {symbol.name}"

        elif kind == SymbolKind.PROPERTY:
            return f"(property) {symbol.signature}"

        else:
            return symbol.name

    def _generate_description(self, symbol: Symbol) -> str:
        """Generate a description when docstring is not available."""
        kind = symbol.kind
        descriptions = []

        # Add kind description
        kind_map = {
            SymbolKind.FUNCTION: "Function",
            SymbolKind.METHOD: "Method",
            SymbolKind.CLASS: "Class",
            SymbolKind.VARIABLE: "Variable",
            SymbolKind.PARAMETER: "Parameter",
            SymbolKind.IMPORT: "Imported module/name",
            SymbolKind.CONSTANT: "Constant",
            SymbolKind.PROPERTY: "Property",
            SymbolKind.ATTRIBUTE: "Attribute",
        }
        kind_desc = kind_map.get(kind, "Symbol")
        descriptions.append(f"*{kind_desc}*")

        # Add visibility info
        if not symbol.is_public:
            descriptions.append("(private)")

        # Add async info
        if symbol.is_async:
            descriptions.append("async")

        # Add decorator info
        if symbol.decorators:
            deco_str = ", ".join(f"@{d}" for d in symbol.decorators[:3])
            if len(symbol.decorators) > 3:
                deco_str += f" (+{len(symbol.decorators) - 3} more)"
            descriptions.append(f"Decorators: {deco_str}")

        # Add base class info
        if symbol.base_classes:
            bases_str = ", ".join(symbol.base_classes[:3])
            if len(symbol.base_classes) > 3:
                bases_str += f" (+{len(symbol.base_classes) - 3} more)"
            descriptions.append(f"Inherits from: {bases_str}")

        # Add import source
        if symbol.imported_from:
            descriptions.append(f"Imported from: {symbol.imported_from}")

        return "\n".join(descriptions) if descriptions else ""

    def get_signature_help(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> Optional[str]:
        """
        Get signature help for a function call.

        This is typically used when the cursor is inside function arguments.
        """
        # For now, just return the hover info title
        hover = self.get_hover(file_path, line, column)
        if hover and hover.symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            return hover.title
        return None
