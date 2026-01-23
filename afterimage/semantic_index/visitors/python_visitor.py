"""
Python Symbol Visitor - Extract symbols and references from Python AST.

Uses tree-sitter to parse Python code and extract:
- Symbol definitions (functions, classes, variables, parameters)
- References to symbols
- Call sites for call graph building
- Scope information
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser, Node, Tree
except ImportError:
    tspython = None
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


class PythonSymbolVisitor:
    """
    Visitor that extracts symbols and references from Python AST.

    Uses tree-sitter for parsing and builds:
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
            if tspython is None:
                raise ImportError("tree-sitter-python not installed")
            self._language = Language(tspython.language())
            self._parser = Parser(self._language)
        return self._parser

    def visit(
        self,
        source: str,
        file_path: str,
        module_name: str = "",
    ) -> Tuple[SymbolTable, CallGraph]:
        """
        Visit Python source code and extract symbols and references.

        Args:
            source: Python source code
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
        context = VisitorContext(
            file_path=file_path,
            module_name=module_name,
            source_bytes=source_bytes,
            symbol_table=symbol_table,
            call_graph=call_graph,
        )

        # Visit the tree
        self._visit_node(tree.root_node, context)

        return symbol_table, call_graph

    def _visit_node(self, node: Node, ctx: 'VisitorContext') -> None:
        """Recursively visit AST nodes."""
        handler = getattr(self, f'_visit_{node.type}', None)
        if handler:
            handler(node, ctx)
        else:
            # Default: visit children
            for child in node.children:
                self._visit_node(child, ctx)

    def _visit_module(self, node: Node, ctx: 'VisitorContext') -> None:
        """Visit module (root) node."""
        for child in node.children:
            self._visit_node(child, ctx)

    def _visit_function_definition(self, node: Node, ctx: 'VisitorContext') -> None:
        """Visit function definition."""
        self._extract_function(node, ctx, is_async=False)

    def _visit_async_function_definition(self, node: Node, ctx: 'VisitorContext') -> None:
        """Visit async function definition."""
        # Find the actual function_definition child
        for child in node.children:
            if child.type == "function_definition":
                self._extract_function(child, ctx, is_async=True)
                return
        # Fallback if structure is different
        self._extract_function(node, ctx, is_async=True)

    def _extract_function(self, node: Node, ctx: 'VisitorContext', is_async: bool = False) -> None:
        """Extract function/method definition."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Determine if it's a method
        is_method = ctx.current_class is not None
        kind = SymbolKind.METHOD if is_method else SymbolKind.FUNCTION

        # Get parameters
        params_node = self._find_child_by_type(node, "parameters")
        parameters = self._extract_parameters(params_node, ctx) if params_node else []

        # Get return type
        return_type = None
        for child in node.children:
            if child.type == "type":
                return_type = TypeInfo(
                    type_string=self._get_node_text(child, ctx.source_bytes)
                )
                break

        # Get docstring
        docstring = self._extract_docstring(node, ctx)

        # Determine visibility
        is_public = not name.startswith("_")

        # Get decorators
        decorators = ctx.pending_decorators.copy()
        ctx.pending_decorators.clear()

        # Create symbol
        symbol = Symbol(
            name=name,
            kind=kind,
            location=location,
            type_info=return_type,
            docstring=docstring,
            parent_symbol=ctx.current_class,
            is_public=is_public,
            parameters=parameters,
            return_type=return_type,
            is_async=is_async,
            decorators=decorators,
        )

        # Add to symbol table
        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

        # Add to call graph
        qualified_name = symbol.qualified_name or f"{ctx.module_name}.{name}"
        ctx.call_graph.add_function(
            qualified_name=qualified_name,
            display_name=name,
            file_path=ctx.file_path,
            location=location,
            is_method=is_method,
            is_async=is_async,
            class_name=ctx.current_class,
        )

        # Create function scope and visit body
        body_node = self._find_child_by_type(node, "block")
        if body_node:
            func_scope = ctx.symbol_table.create_scope(
                kind=ScopeKind.METHOD if is_method else ScopeKind.FUNCTION,
                name=name,
                range=self._get_location_range(body_node, ctx.file_path),
                parent_id=ctx.current_scope_id,
            )
            func_scope.function_symbol = qualified_name

            # Add parameters to function scope
            for param in parameters:
                ctx.symbol_table.add_symbol(param, func_scope.id)

            # Visit body with new context
            old_scope = ctx.current_scope_id
            old_function = ctx.current_function
            ctx.current_scope_id = func_scope.id
            ctx.current_function = qualified_name

            for child in body_node.children:
                self._visit_node(child, ctx)

            ctx.current_scope_id = old_scope
            ctx.current_function = old_function

    def _visit_class_definition(self, node: Node, ctx: 'VisitorContext') -> None:
        """Visit class definition."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Get base classes
        base_classes = []
        arg_list = self._find_child_by_type(node, "argument_list")
        if arg_list:
            for child in arg_list.children:
                if child.type in ("identifier", "attribute"):
                    base_classes.append(self._get_node_text(child, ctx.source_bytes))

        # Get docstring
        docstring = self._extract_docstring(node, ctx)

        # Get decorators
        decorators = ctx.pending_decorators.copy()
        ctx.pending_decorators.clear()

        # Create symbol
        symbol = Symbol(
            name=name,
            kind=SymbolKind.CLASS,
            location=location,
            docstring=docstring,
            is_public=not name.startswith("_"),
            decorators=decorators,
            base_classes=base_classes,
        )

        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)
        qualified_name = symbol.qualified_name

        # Create class scope and visit body
        body_node = self._find_child_by_type(node, "block")
        if body_node:
            class_scope = ctx.symbol_table.create_scope(
                kind=ScopeKind.CLASS,
                name=name,
                range=self._get_location_range(body_node, ctx.file_path),
                parent_id=ctx.current_scope_id,
            )
            class_scope.class_symbol = qualified_name

            # Visit body with class context
            old_scope = ctx.current_scope_id
            old_class = ctx.current_class
            ctx.current_scope_id = class_scope.id
            ctx.current_class = qualified_name

            for child in body_node.children:
                self._visit_node(child, ctx)

            ctx.current_scope_id = old_scope
            ctx.current_class = old_class

    def _visit_decorated_definition(self, node: Node, ctx: 'VisitorContext') -> None:
        """Visit decorated function or class."""
        # Extract decorators
        for child in node.children:
            if child.type == "decorator":
                deco_text = self._get_node_text(child, ctx.source_bytes)
                ctx.pending_decorators.append(deco_text.lstrip("@").strip())

        # Visit the actual definition
        for child in node.children:
            if child.type in ("function_definition", "class_definition", "async_function_definition"):
                self._visit_node(child, ctx)
                break

    def _visit_assignment(self, node: Node, ctx: 'VisitorContext') -> None:
        """Visit assignment statement."""
        # First, analyze the right-hand side for type inference metadata
        rhs_metadata = self._extract_rhs_metadata(node, ctx)

        # Extract left-hand side targets
        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, ctx.source_bytes)
                location = self._get_location_range(child, ctx.file_path)

                # Check if this is a new variable or reassignment
                existing = ctx.symbol_table.resolve_name(name, ctx.current_scope_id)
                if not existing:
                    # New variable with type inference metadata
                    symbol = Symbol(
                        name=name,
                        kind=SymbolKind.VARIABLE,
                        location=location,
                        is_public=not name.startswith("_"),
                        metadata=rhs_metadata.copy(),
                    )
                    ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)
                else:
                    # Reference (write)
                    ref = Reference(
                        name=name,
                        location=location,
                        resolved_symbol=existing,
                        resolved_qualified_name=existing.qualified_name,
                        is_read=False,
                        is_write=True,
                        scope_id=ctx.current_scope_id,
                    )
                    ctx.symbol_table.add_reference(ref)

            elif child.type == "pattern_list":
                # Multiple assignment (a, b = ...)
                for target in child.children:
                    if target.type == "identifier":
                        name = self._get_node_text(target, ctx.source_bytes)
                        location = self._get_location_range(target, ctx.file_path)
                        existing = ctx.symbol_table.resolve_name(name, ctx.current_scope_id)
                        if not existing:
                            symbol = Symbol(
                                name=name,
                                kind=SymbolKind.VARIABLE,
                                location=location,
                                metadata=rhs_metadata.copy(),
                            )
                            ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

        # Visit right-hand side for references
        for i, child in enumerate(node.children):
            if child.type == "=" and i + 1 < len(node.children):
                self._visit_node(node.children[i + 1], ctx)
                break

    def _extract_rhs_metadata(self, node: Node, ctx: 'VisitorContext') -> dict:
        """Extract type inference metadata from assignment right-hand side."""
        metadata = {}

        # Find the right-hand side
        rhs = None
        for i, child in enumerate(node.children):
            if child.type == "=" and i + 1 < len(node.children):
                rhs = node.children[i + 1]
                break

        if not rhs:
            return metadata

        # Check for literal types
        literal_types = {
            "integer": "integer",
            "float": "float",
            "string": "string",
            "true": "true",
            "false": "false",
            "none": "none",
            "list": "list",
            "dictionary": "dictionary",
            "set": "set",
            "tuple": "tuple",
        }
        if rhs.type in literal_types:
            metadata["literal_type"] = literal_types[rhs.type]

        # Check for class instantiation (call with identifier that looks like a class)
        if rhs.type == "call":
            func_node = rhs.children[0] if rhs.children else None
            if func_node:
                func_name = self._get_node_text(func_node, ctx.source_bytes)
                # Convention: class names start with uppercase
                if func_name and func_name[0].isupper():
                    metadata["instantiation_class"] = func_name
                else:
                    metadata["assigned_from_call"] = func_name

        # Check for simple reference assignment (x = y)
        if rhs.type == "identifier":
            ref_name = self._get_node_text(rhs, ctx.source_bytes)
            metadata["assigned_from_ref"] = ref_name

        return metadata

    def _visit_identifier(self, node: Node, ctx: 'VisitorContext') -> None:
        """Visit identifier reference."""
        name = self._get_node_text(node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Skip if this looks like a definition (parent is assignment target)
        parent = node.parent
        if parent and parent.type == "assignment" and node == parent.children[0]:
            return

        # Try to resolve the name
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

    def _visit_call(self, node: Node, ctx: 'VisitorContext') -> None:
        """Visit function call."""
        # Get the function being called
        func_node = node.children[0] if node.children else None
        if not func_node:
            return

        callee_name = self._get_node_text(func_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Count arguments
        args_node = self._find_child_by_type(node, "argument_list")
        arg_count = 0
        has_kwargs = False
        has_star_args = False

        if args_node:
            for child in args_node.children:
                if child.type in ("keyword_argument",):
                    has_kwargs = True
                    arg_count += 1
                elif child.type == "dictionary_splat":
                    has_kwargs = True
                elif child.type == "list_splat":
                    has_star_args = True
                elif child.type not in ("(", ")", ","):
                    arg_count += 1

        # Try to resolve callee
        resolved = ctx.symbol_table.resolve_name(callee_name.split(".")[0], ctx.current_scope_id)

        # Create reference
        ref = Reference(
            name=callee_name,
            location=location,
            resolved_symbol=resolved,
            resolved_qualified_name=resolved.qualified_name if resolved else None,
            is_read=True,
            is_call=True,
            scope_id=ctx.current_scope_id,
        )
        ctx.symbol_table.add_reference(ref)

        # Add to call graph if we're inside a function
        if ctx.current_function:
            call_site = CallSite(
                location=location,
                caller_qualified_name=ctx.current_function,
                callee_name=callee_name,
                callee_qualified_name=resolved.qualified_name if resolved else None,
                argument_count=arg_count,
                has_keyword_args=has_kwargs,
                has_star_args=has_star_args,
            )
            ctx.call_graph.add_call(
                caller=ctx.current_function,
                callee=resolved.qualified_name if resolved else callee_name,
                call_site=call_site,
            )

        # Visit arguments for nested references
        if args_node:
            for child in args_node.children:
                self._visit_node(child, ctx)

    def _visit_import_statement(self, node: Node, ctx: 'VisitorContext') -> None:
        """Visit import statement."""
        for child in node.children:
            if child.type == "dotted_name":
                module = self._get_node_text(child, ctx.source_bytes)
                location = self._get_location_range(node, ctx.file_path)
                symbol = Symbol(
                    name=module.split(".")[-1],
                    kind=SymbolKind.IMPORT,
                    location=location,
                    imported_from=module,
                )
                ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

            elif child.type == "aliased_import":
                name_node = self._find_child_by_type(child, "dotted_name")
                alias_node = self._find_child_by_type(child, "identifier")
                if name_node:
                    module = self._get_node_text(name_node, ctx.source_bytes)
                    alias = self._get_node_text(alias_node, ctx.source_bytes) if alias_node else None
                    location = self._get_location_range(node, ctx.file_path)
                    symbol = Symbol(
                        name=alias or module.split(".")[-1],
                        kind=SymbolKind.IMPORT,
                        location=location,
                        imported_from=module,
                        import_alias=alias,
                    )
                    ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

    def _visit_import_from_statement(self, node: Node, ctx: 'VisitorContext') -> None:
        """Visit from...import statement."""
        module = ""
        location = self._get_location_range(node, ctx.file_path)

        for child in node.children:
            if child.type == "dotted_name":
                module = self._get_node_text(child, ctx.source_bytes)
            elif child.type == "relative_import":
                module = self._get_node_text(child, ctx.source_bytes)
            elif child.type == "import_list":
                for import_child in child.children:
                    if import_child.type == "dotted_name":
                        name = self._get_node_text(import_child, ctx.source_bytes)
                        symbol = Symbol(
                            name=name,
                            kind=SymbolKind.IMPORT,
                            location=location,
                            imported_from=f"{module}.{name}" if module else name,
                        )
                        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)
                    elif import_child.type == "aliased_import":
                        name_node = self._find_child_by_type(import_child, "dotted_name")
                        alias_node = self._find_child_by_type(import_child, "identifier")
                        if name_node:
                            name = self._get_node_text(name_node, ctx.source_bytes)
                            alias = self._get_node_text(alias_node, ctx.source_bytes) if alias_node else None
                            symbol = Symbol(
                                name=alias or name,
                                kind=SymbolKind.IMPORT,
                                location=location,
                                imported_from=f"{module}.{name}" if module else name,
                                import_alias=alias,
                            )
                            ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

    def _extract_parameters(self, params_node: Node, ctx: 'VisitorContext') -> List[Symbol]:
        """Extract function parameters as symbols."""
        params = []

        for child in params_node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, ctx.source_bytes)
                location = self._get_location_range(child, ctx.file_path)
                params.append(Symbol(
                    name=name,
                    kind=SymbolKind.PARAMETER,
                    location=location,
                ))

            elif child.type in ("typed_parameter", "typed_default_parameter"):
                name_node = self._find_child_by_type(child, "identifier")
                type_node = self._find_child_by_type(child, "type")
                if name_node:
                    name = self._get_node_text(name_node, ctx.source_bytes)
                    location = self._get_location_range(child, ctx.file_path)
                    type_info = None
                    if type_node:
                        type_info = TypeInfo(
                            type_string=self._get_node_text(type_node, ctx.source_bytes)
                        )
                    params.append(Symbol(
                        name=name,
                        kind=SymbolKind.PARAMETER,
                        location=location,
                        type_info=type_info,
                    ))

            elif child.type == "default_parameter":
                name_node = self._find_child_by_type(child, "identifier")
                if name_node:
                    name = self._get_node_text(name_node, ctx.source_bytes)
                    location = self._get_location_range(child, ctx.file_path)
                    params.append(Symbol(
                        name=name,
                        kind=SymbolKind.PARAMETER,
                        location=location,
                    ))

            elif child.type == "list_splat_pattern":
                name_node = self._find_child_by_type(child, "identifier")
                if name_node:
                    name = self._get_node_text(name_node, ctx.source_bytes)
                    location = self._get_location_range(child, ctx.file_path)
                    params.append(Symbol(
                        name=name,
                        kind=SymbolKind.PARAMETER,
                        location=location,
                        metadata={"variadic": True},
                    ))

            elif child.type == "dictionary_splat_pattern":
                name_node = self._find_child_by_type(child, "identifier")
                if name_node:
                    name = self._get_node_text(name_node, ctx.source_bytes)
                    location = self._get_location_range(child, ctx.file_path)
                    params.append(Symbol(
                        name=name,
                        kind=SymbolKind.PARAMETER,
                        location=location,
                        metadata={"keyword_variadic": True},
                    ))

        return params

    def _extract_docstring(self, node: Node, ctx: 'VisitorContext') -> Optional[str]:
        """Extract docstring from function/class body."""
        body_node = self._find_child_by_type(node, "block")
        if not body_node or not body_node.children:
            return None

        # First statement should be a string expression
        first_stmt = body_node.children[0]
        if first_stmt.type == "expression_statement":
            string_node = self._find_child_by_type(first_stmt, "string")
            if string_node:
                text = self._get_node_text(string_node, ctx.source_bytes)
                # Remove quotes
                if text.startswith('"""') or text.startswith("'''"):
                    return text[3:-3].strip()
                elif text.startswith('"') or text.startswith("'"):
                    return text[1:-1].strip()
        return None

    def _find_child_by_type(self, node: Node, type_name: str) -> Optional[Node]:
        """Find the first child node of a given type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    def _get_node_text(self, node: Node, source_bytes: bytes) -> str:
        """Get text content of a node."""
        return source_bytes[node.start_byte:node.end_byte].decode('utf-8')

    def _get_location_range(self, node: Node, file_path: str) -> LocationRange:
        """Get LocationRange for a node."""
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


class VisitorContext:
    """Context maintained during AST traversal."""

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
        self.current_scope_id: Optional[str] = symbol_table.root_scope_id
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None

        # Pending decorators (collected before function/class)
        self.pending_decorators: List[str] = []
