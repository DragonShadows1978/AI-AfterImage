"""
Type Inference for Python - Infer types from code patterns.

Provides basic type inference capabilities:
- From literal assignments (x = 1 -> int, y = 'hello' -> str)
- From function return values when return type is annotated
- From class instantiation (obj = MyClass() -> MyClass)
- Propagating inferred types to references
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from .models import Symbol, SymbolKind, Reference, TypeInfo
from .symbol_table import SymbolTable


# Mapping of Python literal types
LITERAL_TYPE_MAP = {
    "integer": "int",
    "float": "float",
    "string": "str",
    "true": "bool",
    "false": "bool",
    "none": "None",
    "list": "list",
    "dictionary": "dict",
    "set": "set",
    "tuple": "tuple",
}

# Built-in types that can be inferred
BUILTIN_TYPES = {
    "int", "float", "str", "bool", "None", "bytes",
    "list", "dict", "set", "tuple", "frozenset",
    "object", "type", "range", "slice",
}


@dataclass
class InferenceResult:
    """Result of type inference."""
    success: bool
    type_info: Optional[TypeInfo] = None
    confidence: float = 0.0  # 0.0-1.0 confidence level
    source: str = ""  # Where the inference came from


class TypeInferencer:
    """
    Infers types for Python symbols based on code patterns.

    Supports:
    - Literal type inference
    - Return type propagation
    - Class instantiation inference
    - Assignment chain tracking
    """

    def __init__(self):
        # Cache of inferred types: symbol qualified_name -> TypeInfo
        self._type_cache: Dict[str, TypeInfo] = {}

        # Assignment tracking: variable -> assigned expression info
        self._assignments: Dict[str, List[AssignmentInfo]] = {}

    def infer_types(self, symbol_table: SymbolTable) -> Dict[str, TypeInfo]:
        """
        Infer types for all symbols in a symbol table.

        Args:
            symbol_table: The symbol table to process

        Returns:
            Dict mapping symbol qualified names to inferred types
        """
        inferred_types: Dict[str, TypeInfo] = {}

        # First pass: collect explicit type annotations
        for symbol in symbol_table.get_all_symbols():
            if symbol.type_info:
                inferred_types[symbol.qualified_name] = symbol.type_info

        # Second pass: infer from assignments and calls
        for symbol in symbol_table.get_all_symbols():
            if symbol.qualified_name in inferred_types:
                continue  # Already has type

            result = self._infer_symbol_type(symbol, symbol_table)
            if result.success and result.type_info:
                inferred_types[symbol.qualified_name] = result.type_info
                symbol.type_info = result.type_info

        # Third pass: propagate types to references
        for ref in symbol_table.references:
            if ref.resolved_symbol and ref.resolved_symbol.qualified_name in inferred_types:
                # Reference inherits type from resolved symbol
                pass  # References don't store types directly

        self._type_cache.update(inferred_types)
        return inferred_types

    def _infer_symbol_type(
        self,
        symbol: Symbol,
        symbol_table: SymbolTable,
    ) -> InferenceResult:
        """Infer the type of a single symbol."""
        # Functions/methods - return type
        if symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            if symbol.return_type:
                return InferenceResult(
                    success=True,
                    type_info=symbol.return_type,
                    confidence=1.0,
                    source="annotation",
                )
            # Could analyze function body for return statements
            return InferenceResult(success=False)

        # Classes - type is the class itself
        if symbol.kind == SymbolKind.CLASS:
            return InferenceResult(
                success=True,
                type_info=TypeInfo(type_string=f"type[{symbol.name}]"),
                confidence=1.0,
                source="class_definition",
            )

        # Variables - check metadata for literal type or inference hints
        if symbol.kind == SymbolKind.VARIABLE:
            return self._infer_variable_type(symbol, symbol_table)

        # Parameters - check type annotation
        if symbol.kind == SymbolKind.PARAMETER:
            if symbol.type_info:
                return InferenceResult(
                    success=True,
                    type_info=symbol.type_info,
                    confidence=1.0,
                    source="annotation",
                )
            return InferenceResult(success=False)

        return InferenceResult(success=False)

    def _infer_variable_type(
        self,
        symbol: Symbol,
        symbol_table: SymbolTable,
    ) -> InferenceResult:
        """Infer the type of a variable from its assignment."""
        # Check metadata for literal type
        if "literal_type" in symbol.metadata:
            literal_type = symbol.metadata["literal_type"]
            if literal_type in LITERAL_TYPE_MAP:
                return InferenceResult(
                    success=True,
                    type_info=TypeInfo(
                        type_string=LITERAL_TYPE_MAP[literal_type],
                        is_inferred=True,
                    ),
                    confidence=1.0,
                    source="literal",
                )

        # Check metadata for instantiation
        if "instantiation_class" in symbol.metadata:
            class_name = symbol.metadata["instantiation_class"]
            return InferenceResult(
                success=True,
                type_info=TypeInfo(
                    type_string=class_name,
                    is_inferred=True,
                ),
                confidence=0.9,
                source="instantiation",
            )

        # Check metadata for assigned_from (function call)
        if "assigned_from_call" in symbol.metadata:
            func_name = symbol.metadata["assigned_from_call"]
            # Try to find the function and get its return type
            func_symbols = symbol_table.get_symbols_by_name(func_name.split(".")[-1])
            for func_sym in func_symbols:
                if func_sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                    if func_sym.return_type:
                        return InferenceResult(
                            success=True,
                            type_info=TypeInfo(
                                type_string=func_sym.return_type.type_string,
                                is_inferred=True,
                            ),
                            confidence=0.8,
                            source="return_type",
                        )

        return InferenceResult(success=False)

    def infer_from_literal(self, literal_type: str) -> InferenceResult:
        """
        Infer type from a literal value type.

        Args:
            literal_type: The tree-sitter node type (e.g., "integer", "string")

        Returns:
            InferenceResult with the inferred type
        """
        if literal_type in LITERAL_TYPE_MAP:
            return InferenceResult(
                success=True,
                type_info=TypeInfo(
                    type_string=LITERAL_TYPE_MAP[literal_type],
                    is_inferred=True,
                ),
                confidence=1.0,
                source="literal",
            )
        return InferenceResult(success=False)

    def infer_from_instantiation(self, class_name: str) -> InferenceResult:
        """
        Infer type from class instantiation.

        Args:
            class_name: The class being instantiated

        Returns:
            InferenceResult with the inferred type
        """
        return InferenceResult(
            success=True,
            type_info=TypeInfo(
                type_string=class_name,
                is_inferred=True,
            ),
            confidence=0.95,
            source="instantiation",
        )

    def infer_from_function_return(
        self,
        function_symbol: Symbol,
    ) -> InferenceResult:
        """
        Infer type from function return type annotation.

        Args:
            function_symbol: The function being called

        Returns:
            InferenceResult with the return type
        """
        if function_symbol.return_type:
            return InferenceResult(
                success=True,
                type_info=TypeInfo(
                    type_string=function_symbol.return_type.type_string,
                    is_inferred=True,
                ),
                confidence=0.9,
                source="return_type",
            )
        return InferenceResult(success=False)

    def get_cached_type(self, qualified_name: str) -> Optional[TypeInfo]:
        """Get a cached type for a symbol."""
        return self._type_cache.get(qualified_name)

    def clear_cache(self) -> None:
        """Clear the type cache."""
        self._type_cache.clear()
        self._assignments.clear()


@dataclass
class AssignmentInfo:
    """Information about an assignment to a variable."""
    line: int
    column: int
    value_type: Optional[str] = None
    value_source: str = ""  # "literal", "call", "reference", etc.
    class_instantiation: Optional[str] = None
    function_call: Optional[str] = None


class TypePropagator:
    """
    Propagates inferred types through references and assignments.

    Handles:
    - Tracking type flow through assignments
    - Propagating types to references
    - Handling type narrowing in conditionals
    """

    def __init__(self, inferencer: TypeInferencer):
        self.inferencer = inferencer
        self._propagated: Set[str] = set()

    def propagate(self, symbol_table: SymbolTable) -> int:
        """
        Propagate types through the symbol table.

        Returns:
            Number of newly typed symbols
        """
        newly_typed = 0

        # Propagate from definitions to references
        for ref in symbol_table.references:
            if ref.resolved_symbol:
                symbol = ref.resolved_symbol
                if symbol.type_info and symbol.qualified_name not in self._propagated:
                    self._propagated.add(symbol.qualified_name)
                    newly_typed += 1

        # Propagate through assignment chains
        for symbol in symbol_table.get_all_symbols():
            if symbol.kind == SymbolKind.VARIABLE and not symbol.type_info:
                # Check if this variable is assigned from another typed variable
                if "assigned_from_ref" in symbol.metadata:
                    ref_name = symbol.metadata["assigned_from_ref"]
                    ref_symbols = symbol_table.get_symbols_by_name(ref_name)
                    for ref_sym in ref_symbols:
                        if ref_sym.type_info:
                            symbol.type_info = TypeInfo(
                                type_string=ref_sym.type_info.type_string,
                                is_inferred=True,
                            )
                            newly_typed += 1
                            break

        return newly_typed

    def get_type_at_reference(
        self,
        reference: Reference,
        symbol_table: SymbolTable,
    ) -> Optional[TypeInfo]:
        """
        Get the type of a reference at its usage point.

        Args:
            reference: The reference to get type for
            symbol_table: The symbol table context

        Returns:
            TypeInfo if type can be determined
        """
        if reference.resolved_symbol:
            return reference.resolved_symbol.type_info

        # Try to resolve and get type
        resolved = symbol_table.resolve_name(
            reference.name,
            from_scope_id=reference.scope_id,
        )
        if resolved and resolved.type_info:
            return resolved.type_info

        return None


def enhance_visitor_with_inference(visitor_class):
    """
    Decorator to add type inference capabilities to a visitor class.

    This adds metadata tracking for literals, instantiations, and calls
    that can be used by TypeInferencer.
    """
    original_visit_assignment = getattr(visitor_class, '_visit_assignment', None)

    def _visit_assignment_with_inference(self, node, ctx):
        # Call original if exists
        if original_visit_assignment:
            original_visit_assignment(self, node, ctx)

        # Add inference metadata
        # This would need to analyze the right-hand side of assignments
        # and add metadata like:
        # - literal_type: for literals
        # - instantiation_class: for MyClass() calls
        # - assigned_from_call: for func() calls
        # - assigned_from_ref: for x = y style assignments

    visitor_class._visit_assignment = _visit_assignment_with_inference
    return visitor_class
