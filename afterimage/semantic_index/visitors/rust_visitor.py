"""
Rust Symbol Visitor - Extracts symbols and references from Rust source code.

Handles:
- fn declarations (functions and methods)
- impl blocks
- struct/enum definitions
- trait definitions
- use statements
- mod declarations
- Rust-specific patterns: lifetime annotations, generic parameters, pattern matching
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

try:
    import tree_sitter_rust as tsrust
    from tree_sitter import Language, Parser, Node, Tree
except ImportError:
    tsrust = None
    Language = None
    Parser = None
    Node = None
    Tree = None

from ..models import (
    Location,
    LocationRange,
    Symbol,
    SymbolKind,
    Reference,
    Scope,
    ScopeKind,
    TypeInfo,
    CallSite,
)
from ..symbol_table import SymbolTable
from ..call_graph import CallGraph


class RustVisitorContext:
    """Context maintained during Rust AST traversal."""

    def __init__(
        self,
        file_path: str,
        module_name: str,
        source_bytes: bytes,
        symbol_table: SymbolTable,
        call_graph: CallGraph,
    ):
        self.file_path = file_path
        self.module_name = module_name
        self.source_bytes = source_bytes
        self.symbol_table = symbol_table
        self.call_graph = call_graph

        # Current scope tracking
        self.current_scope_id: str = symbol_table.root_scope_id

        # Current impl block for method resolution
        self.current_impl_type: Optional[str] = None
        self.current_impl_trait: Optional[str] = None

        # Current function for call graph
        self.current_function: Optional[str] = None

        # Pending attributes (like #[derive(...)])
        self.pending_attributes: List[str] = []

        # Use imports tracking for resolution
        self.use_imports: Dict[str, str] = {}  # local_name -> full_path


class RustSymbolVisitor:
    """
    Visitor that extracts symbols and references from Rust AST.

    Uses tree-sitter-rust for parsing and builds:
    - SymbolTable with all declarations and references
    - CallGraph with function call relationships
    """

    def __init__(self):
        self._parser: Optional[Parser] = None
        self._language: Optional[Language] = None

    @property
    def parser(self) -> Parser:
        """Lazy initialization of tree-sitter parser."""
        if self._parser is None:
            if tsrust is None:
                raise ImportError("tree-sitter-rust not installed. Install with: pip install tree-sitter-rust")
            self._language = Language(tsrust.language())
            self._parser = Parser(self._language)
        return self._parser

    def visit(
        self,
        source: str,
        file_path: str,
        module_name: str = "",
    ) -> Tuple[SymbolTable, CallGraph]:
        """
        Visit Rust source code and extract symbols and references.

        Args:
            source: Rust source code
            file_path: Path to the source file
            module_name: Module name for qualified name generation

        Returns:
            Tuple of (SymbolTable, CallGraph)
        """
        source_bytes = source.encode('utf-8')
        tree = self.parser.parse(source_bytes)

        # Initialize outputs
        symbol_table = SymbolTable(file_path=file_path, module_name=module_name)
        call_graph = CallGraph()

        # Context for tracking current scope during traversal
        context = RustVisitorContext(
            file_path=file_path,
            module_name=module_name,
            source_bytes=source_bytes,
            symbol_table=symbol_table,
            call_graph=call_graph,
        )

        # Visit the tree
        self._visit_node(tree.root_node, context)

        return symbol_table, call_graph

    def _visit_node(self, node: Node, ctx: RustVisitorContext) -> None:
        """Recursively visit AST nodes."""
        handler = getattr(self, f'_visit_{node.type}', None)
        if handler:
            handler(node, ctx)
        else:
            # Default: visit children
            for child in node.children:
                self._visit_node(child, ctx)

    def _get_node_text(self, node: Node, source_bytes: bytes) -> str:
        """Get the text content of a node."""
        return source_bytes[node.start_byte:node.end_byte].decode('utf-8')

    def _get_location(self, node: Node, file_path: str) -> Location:
        """Get location from node position."""
        return Location(
            file_path=file_path,
            line=node.start_point[0] + 1,  # 1-indexed
            column=node.start_point[1],
        )

    def _get_location_range(self, node: Node, file_path: str) -> LocationRange:
        """Get location range from node."""
        return LocationRange(
            start=Location(
                file_path=file_path,
                line=node.start_point[0] + 1,
                column=node.start_point[1],
            ),
            end=Location(
                file_path=file_path,
                line=node.end_point[0] + 1,
                column=node.end_point[1],
            ),
        )

    def _find_child_by_type(self, node: Node, type_name: str) -> Optional[Node]:
        """Find first child of given type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    def _find_children_by_type(self, node: Node, type_name: str) -> List[Node]:
        """Find all children of given type."""
        return [child for child in node.children if child.type == type_name]

    # ==================== Module and Use Statements ====================

    def _visit_source_file(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit source file (root) node."""
        for child in node.children:
            self._visit_node(child, ctx)

    def _visit_mod_item(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit mod declaration."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        symbol = Symbol(
            name=name,
            kind=SymbolKind.MODULE,
            location=location,
            is_public=self._is_public(node),
        )
        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

        # If mod has a body (inline module), visit it
        body = self._find_child_by_type(node, "declaration_list")
        if body:
            mod_scope = ctx.symbol_table.create_scope(
                kind=ScopeKind.MODULE,
                name=name,
                range=self._get_location_range(body, ctx.file_path),
                parent_id=ctx.current_scope_id,
            )
            old_scope = ctx.current_scope_id
            ctx.current_scope_id = mod_scope.id
            for child in body.children:
                self._visit_node(child, ctx)
            ctx.current_scope_id = old_scope

    def _visit_use_declaration(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit use statement."""
        # Extract the use tree
        use_tree = self._find_child_by_type(node, "use_tree")
        if use_tree:
            self._process_use_tree(use_tree, ctx, prefix="")

    def _process_use_tree(self, node: Node, ctx: RustVisitorContext, prefix: str) -> None:
        """Process a use tree recursively."""
        # Handle scoped identifier (e.g., std::collections::HashMap)
        scoped = self._find_child_by_type(node, "scoped_identifier")
        if scoped:
            full_path = self._get_node_text(scoped, ctx.source_bytes)
            parts = full_path.split("::")
            local_name = parts[-1]
            self._add_import(ctx, local_name, full_path, node)
            return

        # Handle identifier (simple use)
        ident = self._find_child_by_type(node, "identifier")
        if ident:
            name = self._get_node_text(ident, ctx.source_bytes)
            full_path = f"{prefix}::{name}" if prefix else name
            self._add_import(ctx, name, full_path, node)
            return

        # Handle use_list (e.g., use std::{io, fs})
        use_list = self._find_child_by_type(node, "use_list")
        if use_list:
            # Get the path prefix
            path = self._find_child_by_type(node, "scoped_identifier")
            if path:
                prefix = self._get_node_text(path, ctx.source_bytes)
            for child in use_list.children:
                if child.type == "use_tree":
                    self._process_use_tree(child, ctx, prefix)
                elif child.type == "identifier":
                    name = self._get_node_text(child, ctx.source_bytes)
                    full_path = f"{prefix}::{name}" if prefix else name
                    self._add_import(ctx, name, full_path, node)

        # Handle use_as_clause (e.g., use std::io as stdio)
        use_as = self._find_child_by_type(node, "use_as_clause")
        if use_as:
            path_node = use_as.children[0] if use_as.children else None
            alias_node = self._find_child_by_type(use_as, "identifier")
            if path_node and alias_node:
                full_path = self._get_node_text(path_node, ctx.source_bytes)
                alias = self._get_node_text(alias_node, ctx.source_bytes)
                self._add_import(ctx, alias, full_path, node, alias=alias)

    def _add_import(
        self,
        ctx: RustVisitorContext,
        local_name: str,
        full_path: str,
        node: Node,
        alias: Optional[str] = None,
    ) -> None:
        """Add an import symbol."""
        location = self._get_location_range(node, ctx.file_path)
        symbol = Symbol(
            name=local_name,
            kind=SymbolKind.IMPORT,
            location=location,
            imported_from=full_path,
            import_alias=alias,
            is_public=self._is_public(node),
        )
        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)
        ctx.use_imports[local_name] = full_path

    # ==================== Function Definitions ====================

    def _visit_function_item(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit function definition."""
        self._extract_function(node, ctx, is_method=False)

    def _extract_function(
        self,
        node: Node,
        ctx: RustVisitorContext,
        is_method: bool = False,
    ) -> None:
        """Extract function/method definition."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Extract parameters
        params = []
        params_node = self._find_child_by_type(node, "parameters")
        if params_node:
            params = self._extract_parameters(params_node, ctx)

        # Extract return type
        return_type = None
        for child in node.children:
            if child.type == "->":
                idx = node.children.index(child)
                if idx + 1 < len(node.children):
                    type_node = node.children[idx + 1]
                    type_str = self._get_node_text(type_node, ctx.source_bytes)
                    return_type = TypeInfo(type_string=type_str)
                break

        # Extract generic parameters
        generics = self._extract_generics(node, ctx)

        # Check for async
        is_async = any(child.type == "async" for child in node.children)

        # Get visibility
        is_public = self._is_public(node)

        # Create symbol
        kind = SymbolKind.METHOD if is_method else SymbolKind.FUNCTION
        symbol = Symbol(
            name=name,
            kind=kind,
            location=location,
            parameters=params,
            return_type=return_type,
            is_async=is_async,
            is_public=is_public,
            parent_symbol=ctx.current_impl_type,
            metadata={"generics": generics} if generics else {},
        )
        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)
        qualified_name = symbol.qualified_name

        # Register in call graph
        ctx.call_graph.add_function(
            qualified_name=qualified_name,
            display_name=name,
            file_path=ctx.file_path,
            location=location,
            is_method=is_method,
            is_async=is_async,
        )

        # Visit function body
        body = self._find_child_by_type(node, "block")
        if body:
            func_scope = ctx.symbol_table.create_scope(
                kind=ScopeKind.FUNCTION,
                name=name,
                range=self._get_location_range(body, ctx.file_path),
                parent_id=ctx.current_scope_id,
            )
            func_scope.function_symbol = qualified_name

            # Add parameters to scope
            for param in params:
                ctx.symbol_table.add_symbol(param, func_scope.id)

            old_scope = ctx.current_scope_id
            old_func = ctx.current_function
            ctx.current_scope_id = func_scope.id
            ctx.current_function = qualified_name

            for child in body.children:
                self._visit_node(child, ctx)

            ctx.current_scope_id = old_scope
            ctx.current_function = old_func

    def _extract_parameters(self, params_node: Node, ctx: RustVisitorContext) -> List[Symbol]:
        """Extract function parameters."""
        params = []
        for child in params_node.children:
            if child.type == "parameter":
                # pattern: type
                pattern = self._find_child_by_type(child, "identifier")
                type_node = self._find_child_by_type(child, "type_identifier")
                if not type_node:
                    # Try other type nodes
                    for c in child.children:
                        if "type" in c.type or c.type == "reference_type":
                            type_node = c
                            break

                if pattern:
                    name = self._get_node_text(pattern, ctx.source_bytes)
                    type_info = None
                    if type_node:
                        type_str = self._get_node_text(type_node, ctx.source_bytes)
                        type_info = TypeInfo(type_string=type_str)

                    param = Symbol(
                        name=name,
                        kind=SymbolKind.PARAMETER,
                        location=self._get_location_range(child, ctx.file_path),
                        type_info=type_info,
                    )
                    params.append(param)

            elif child.type == "self_parameter":
                # self, &self, &mut self
                self_text = self._get_node_text(child, ctx.source_bytes)
                param = Symbol(
                    name="self",
                    kind=SymbolKind.PARAMETER,
                    location=self._get_location_range(child, ctx.file_path),
                    metadata={"self_type": self_text},
                )
                params.append(param)

        return params

    def _extract_generics(self, node: Node, ctx: RustVisitorContext) -> List[str]:
        """Extract generic type parameters."""
        generics = []
        type_params = self._find_child_by_type(node, "type_parameters")
        if type_params:
            for child in type_params.children:
                if child.type == "type_identifier" or child.type == "lifetime":
                    generics.append(self._get_node_text(child, ctx.source_bytes))
                elif child.type == "constrained_type_parameter":
                    ident = self._find_child_by_type(child, "type_identifier")
                    if ident:
                        generics.append(self._get_node_text(ident, ctx.source_bytes))
        return generics

    # ==================== Struct/Enum/Trait Definitions ====================

    def _visit_struct_item(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit struct definition."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Extract generic parameters
        generics = self._extract_generics(node, ctx)

        # Get attributes
        attrs = ctx.pending_attributes.copy()
        ctx.pending_attributes.clear()

        symbol = Symbol(
            name=name,
            kind=SymbolKind.CLASS,  # Struct maps to CLASS
            location=location,
            is_public=self._is_public(node),
            decorators=attrs,  # Attributes as decorators
            metadata={"generics": generics, "rust_kind": "struct"},
        )
        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

        # Visit struct fields
        field_list = self._find_child_by_type(node, "field_declaration_list")
        if field_list:
            struct_scope = ctx.symbol_table.create_scope(
                kind=ScopeKind.CLASS,
                name=name,
                range=self._get_location_range(field_list, ctx.file_path),
                parent_id=ctx.current_scope_id,
            )
            struct_scope.class_symbol = symbol.qualified_name

            for child in field_list.children:
                if child.type == "field_declaration":
                    self._extract_field(child, ctx, struct_scope.id)

    def _visit_enum_item(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit enum definition."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        generics = self._extract_generics(node, ctx)
        attrs = ctx.pending_attributes.copy()
        ctx.pending_attributes.clear()

        symbol = Symbol(
            name=name,
            kind=SymbolKind.ENUM,
            location=location,
            is_public=self._is_public(node),
            decorators=attrs,
            metadata={"generics": generics},
        )
        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

        # Visit enum variants
        variant_list = self._find_child_by_type(node, "enum_variant_list")
        if variant_list:
            enum_scope = ctx.symbol_table.create_scope(
                kind=ScopeKind.CLASS,
                name=name,
                range=self._get_location_range(variant_list, ctx.file_path),
                parent_id=ctx.current_scope_id,
            )

            for child in variant_list.children:
                if child.type == "enum_variant":
                    self._extract_enum_variant(child, ctx, enum_scope.id, symbol.qualified_name)

    def _extract_enum_variant(
        self,
        node: Node,
        ctx: RustVisitorContext,
        scope_id: str,
        parent_enum: str,
    ) -> None:
        """Extract enum variant."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        symbol = Symbol(
            name=name,
            kind=SymbolKind.ENUM_MEMBER,
            location=location,
            parent_symbol=parent_enum,
        )
        ctx.symbol_table.add_symbol(symbol, scope_id)

    def _visit_trait_item(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit trait definition."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        generics = self._extract_generics(node, ctx)

        symbol = Symbol(
            name=name,
            kind=SymbolKind.INTERFACE,  # Trait maps to INTERFACE
            location=location,
            is_public=self._is_public(node),
            metadata={"generics": generics, "rust_kind": "trait"},
        )
        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

        # Visit trait body
        body = self._find_child_by_type(node, "declaration_list")
        if body:
            trait_scope = ctx.symbol_table.create_scope(
                kind=ScopeKind.CLASS,
                name=name,
                range=self._get_location_range(body, ctx.file_path),
                parent_id=ctx.current_scope_id,
            )

            old_scope = ctx.current_scope_id
            ctx.current_scope_id = trait_scope.id

            for child in body.children:
                self._visit_node(child, ctx)

            ctx.current_scope_id = old_scope

    def _extract_field(self, node: Node, ctx: RustVisitorContext, scope_id: str) -> None:
        """Extract struct field."""
        name_node = self._find_child_by_type(node, "field_identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Get field type
        type_info = None
        for child in node.children:
            if "type" in child.type:
                type_str = self._get_node_text(child, ctx.source_bytes)
                type_info = TypeInfo(type_string=type_str)
                break

        symbol = Symbol(
            name=name,
            kind=SymbolKind.ATTRIBUTE,
            location=location,
            type_info=type_info,
            is_public=self._is_public(node),
        )
        ctx.symbol_table.add_symbol(symbol, scope_id)

    # ==================== Impl Blocks ====================

    def _visit_impl_item(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit impl block."""
        # Extract the type being implemented
        type_node = None
        trait_node = None

        for i, child in enumerate(node.children):
            if child.type == "type_identifier" or child.type == "generic_type":
                if trait_node is None:
                    # First type could be trait or target type
                    # Check for "for" keyword after
                    has_for = any(c.type == "for" for c in node.children[i:])
                    if has_for:
                        trait_node = child
                    else:
                        type_node = child
                else:
                    type_node = child
            elif child.type == "for":
                continue

        if type_node:
            ctx.current_impl_type = self._get_node_text(type_node, ctx.source_bytes)
        if trait_node:
            ctx.current_impl_trait = self._get_node_text(trait_node, ctx.source_bytes)

        # Visit impl body
        body = self._find_child_by_type(node, "declaration_list")
        if body:
            for child in body.children:
                if child.type == "function_item":
                    self._extract_function(child, ctx, is_method=True)
                elif child.type == "associated_type":
                    self._visit_associated_type(child, ctx)
                else:
                    self._visit_node(child, ctx)

        ctx.current_impl_type = None
        ctx.current_impl_trait = None

    def _visit_associated_type(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit associated type in impl/trait."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        symbol = Symbol(
            name=name,
            kind=SymbolKind.TYPE_ALIAS,
            location=location,
            parent_symbol=ctx.current_impl_type,
        )
        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

    # ==================== Variables and References ====================

    def _visit_let_declaration(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit let binding."""
        # Find the pattern (variable name)
        pattern = self._find_child_by_type(node, "identifier")
        if not pattern:
            # Try tuple pattern or other patterns
            pattern = self._find_child_by_type(node, "tuple_pattern")

        if pattern and pattern.type == "identifier":
            name = self._get_node_text(pattern, ctx.source_bytes)
            location = self._get_location_range(pattern, ctx.file_path)

            # Extract type annotation
            type_info = None
            for child in node.children:
                if child.type == "type_annotation":
                    type_node = child.children[-1] if child.children else None
                    if type_node:
                        type_str = self._get_node_text(type_node, ctx.source_bytes)
                        type_info = TypeInfo(type_string=type_str)
                    break

            # Check for mutability
            is_mutable = any(child.type == "mutable_specifier" for child in node.children)

            symbol = Symbol(
                name=name,
                kind=SymbolKind.VARIABLE,
                location=location,
                type_info=type_info,
                metadata={"mutable": is_mutable},
            )
            ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

        # Visit the value expression
        for child in node.children:
            if child.type not in ("let", "identifier", "type_annotation", "mutable_specifier", "="):
                self._visit_node(child, ctx)

    def _visit_identifier(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit identifier reference."""
        name = self._get_node_text(node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Skip if parent indicates definition context
        parent = node.parent
        if parent and parent.type in (
            "let_declaration", "parameter", "field_declaration",
            "function_item", "struct_item", "enum_item", "trait_item",
            "mod_item", "use_declaration",
        ):
            return

        # Try to resolve
        resolved = ctx.symbol_table.resolve_name(name, ctx.current_scope_id)

        ref = Reference(
            name=name,
            location=location,
            resolved_symbol=resolved,
            resolved_qualified_name=resolved.qualified_name if resolved else None,
            is_read=True,
            scope_id=ctx.current_scope_id,
        )
        ctx.symbol_table.add_reference(ref)

    def _visit_call_expression(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit function call."""
        func_node = node.children[0] if node.children else None
        if not func_node:
            return

        func_name = self._get_node_text(func_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Count arguments
        args_node = self._find_child_by_type(node, "arguments")
        arg_count = 0
        if args_node:
            arg_count = sum(1 for c in args_node.children if c.type not in ("(", ")", ","))

        # Create call site
        if ctx.current_function:
            call_site = CallSite(
                location=location,
                caller_qualified_name=ctx.current_function,
                callee_name=func_name,
                argument_count=arg_count,
            )
            ctx.call_graph.add_call(
                ctx.current_function,
                func_name,
                call_site,
            )

        # Add reference
        ref = Reference(
            name=func_name,
            location=location,
            is_call=True,
            scope_id=ctx.current_scope_id,
        )
        ctx.symbol_table.add_reference(ref)

        # Visit arguments
        if args_node:
            for child in args_node.children:
                self._visit_node(child, ctx)

    # ==================== Attributes ====================

    def _visit_attribute_item(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit attribute (e.g., #[derive(Debug)])."""
        attr_text = self._get_node_text(node, ctx.source_bytes)
        ctx.pending_attributes.append(attr_text)

    # ==================== Pattern Matching ====================

    def _visit_match_expression(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit match expression."""
        # Visit the scrutinee
        for child in node.children:
            if child.type not in ("match", "{", "}", "match_arm"):
                self._visit_node(child, ctx)

        # Visit match arms
        for child in node.children:
            if child.type == "match_arm":
                self._visit_match_arm(child, ctx)

    def _visit_match_arm(self, node: Node, ctx: RustVisitorContext) -> None:
        """Visit match arm."""
        # Pattern introduces bindings
        pattern = self._find_child_by_type(node, "match_pattern")
        if pattern:
            self._extract_pattern_bindings(pattern, ctx)

        # Visit the arm body
        for child in node.children:
            if child.type not in ("match_pattern", "=>"):
                self._visit_node(child, ctx)

    def _extract_pattern_bindings(self, node: Node, ctx: RustVisitorContext) -> None:
        """Extract variable bindings from pattern."""
        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, ctx.source_bytes)
                # Skip if it looks like a variant name (starts with uppercase)
                if name and not name[0].isupper() and name != "_":
                    symbol = Symbol(
                        name=name,
                        kind=SymbolKind.VARIABLE,
                        location=self._get_location_range(child, ctx.file_path),
                        metadata={"from_pattern": True},
                    )
                    ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)
            else:
                self._extract_pattern_bindings(child, ctx)

    # ==================== Helper Methods ====================

    def _is_public(self, node: Node) -> bool:
        """Check if a node has pub visibility."""
        for child in node.children:
            if child.type == "visibility_modifier":
                return True
        return False


# Convenience function for quick parsing
def parse_rust_file(file_path: str) -> Tuple[SymbolTable, CallGraph]:
    """Parse a Rust file and return symbol table and call graph."""
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    module_name = Path(file_path).stem
    visitor = RustSymbolVisitor()
    return visitor.visit(source, file_path, module_name)
