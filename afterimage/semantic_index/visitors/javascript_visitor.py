"""
JavaScript/TypeScript Symbol Visitor - Extract symbols and references from JS/TS code.

Uses tree-sitter to parse JavaScript/TypeScript code and extract:
- Symbol definitions (functions, classes, variables, arrow functions)
- References to symbols
- Call sites for call graph building
- Scope information
- Import/Export statements
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

try:
    import tree_sitter_javascript as tsjs
    import tree_sitter_typescript as tsts
    from tree_sitter import Language, Parser, Node
except ImportError:
    tsjs = None
    tsts = None
    Language = None
    Parser = None
    Node = None

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


class JavaScriptSymbolVisitor:
    """
    Visitor that extracts symbols and references from JavaScript/TypeScript AST.

    Uses tree-sitter for parsing and builds:
    - SymbolTable with all declarations and references
    - CallGraph with function call relationships

    Supports:
    - Function declarations and expressions
    - Arrow functions
    - Classes with methods
    - const/let/var variable declarations
    - Import/export statements
    - ES6+ features (destructuring, spread, etc.)
    """

    def __init__(self, language: str = "javascript"):
        """
        Initialize the visitor.

        Args:
            language: "javascript" or "typescript"
        """
        self.language_name = language
        self._parser: Optional[Parser] = None
        self._language: Optional[Language] = None

    @property
    def parser(self) -> Parser:
        """Lazy initialization of tree-sitter parser."""
        if self._parser is None:
            if self.language_name == "typescript":
                if tsts is None:
                    raise ImportError("tree-sitter-typescript not installed")
                self._language = Language(tsts.language_typescript())
            else:
                if tsjs is None:
                    raise ImportError("tree-sitter-javascript not installed")
                self._language = Language(tsjs.language())
            self._parser = Parser(self._language)
        return self._parser

    def visit(
        self,
        source: str,
        file_path: str,
        module_name: str = "",
    ) -> Tuple[SymbolTable, CallGraph]:
        """
        Visit JavaScript/TypeScript source code and extract symbols and references.

        Args:
            source: Source code
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
        context = JSVisitorContext(
            file_path=file_path,
            module_name=module_name,
            source_bytes=source_bytes,
            symbol_table=symbol_table,
            call_graph=call_graph,
        )

        # Visit the tree
        self._visit_node(tree.root_node, context)

        return symbol_table, call_graph

    def _visit_node(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Recursively visit AST nodes."""
        handler = getattr(self, f'_visit_{node.type}', None)
        if handler:
            handler(node, ctx)
        else:
            # Default: visit children
            for child in node.children:
                self._visit_node(child, ctx)

    def _visit_program(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit program (root) node."""
        for child in node.children:
            self._visit_node(child, ctx)

    # ======================
    # Function Declarations
    # ======================

    def _visit_function_declaration(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit function declaration."""
        self._extract_function(node, ctx, is_expression=False)

    def _visit_function(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit function expression."""
        self._extract_function(node, ctx, is_expression=True)

    def _visit_function_expression(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit function expression (aliased)."""
        self._extract_function(node, ctx, is_expression=True)

    def _visit_generator_function_declaration(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit generator function declaration."""
        self._extract_function(node, ctx, is_generator=True)

    def _visit_generator_function(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit generator function expression."""
        self._extract_function(node, ctx, is_expression=True, is_generator=True)

    def _visit_arrow_function(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit arrow function."""
        self._extract_arrow_function(node, ctx)

    def _extract_function(
        self,
        node: Node,
        ctx: 'JSVisitorContext',
        is_expression: bool = False,
        is_async: bool = False,
        is_generator: bool = False,
    ) -> Optional[Symbol]:
        """Extract function/method definition."""
        # Check for async keyword
        for child in node.children:
            if child.type == "async":
                is_async = True
                break

        name_node = self._find_child_by_type(node, "identifier")
        name = self._get_node_text(name_node, ctx.source_bytes) if name_node else None

        # Anonymous function in expression context
        if not name and is_expression:
            name = ctx.get_anon_name("function")

        if not name:
            return None

        location = self._get_location_range(node, ctx.file_path)

        # Determine if it's a method
        is_method = ctx.current_class is not None
        kind = SymbolKind.METHOD if is_method else SymbolKind.FUNCTION

        # Get parameters
        params_node = self._find_child_by_type(node, "formal_parameters")
        parameters = self._extract_parameters(params_node, ctx) if params_node else []

        # Get return type (TypeScript)
        return_type = self._extract_return_type(node, ctx)

        # Determine visibility
        is_public = not name.startswith("_")

        # Create symbol
        symbol = Symbol(
            name=name,
            kind=kind,
            location=location,
            type_info=return_type,
            parent_symbol=ctx.current_class,
            is_public=is_public,
            parameters=parameters,
            return_type=return_type,
            is_async=is_async,
            is_generator=is_generator,
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
        body_node = self._find_child_by_type(node, "statement_block")
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

        return symbol

    def _extract_arrow_function(
        self,
        node: Node,
        ctx: 'JSVisitorContext',
    ) -> Optional[Symbol]:
        """Extract arrow function."""
        # Arrow functions are typically assigned to variables
        # Check if we're in a variable declarator context
        name = ctx.pending_var_name or ctx.get_anon_name("arrow")
        ctx.pending_var_name = None

        location = self._get_location_range(node, ctx.file_path)

        # Check for async
        is_async = False
        parent = node.parent
        if parent and parent.type == "await_expression":
            is_async = True
        for child in node.children:
            if child.type == "async":
                is_async = True
                break

        # Get parameters
        params_node = self._find_child_by_type(node, "formal_parameters")
        if not params_node:
            # Single parameter without parentheses
            params_node = self._find_child_by_type(node, "identifier")
            if params_node:
                parameters = [Symbol(
                    name=self._get_node_text(params_node, ctx.source_bytes),
                    kind=SymbolKind.PARAMETER,
                    location=self._get_location_range(params_node, ctx.file_path),
                )]
            else:
                parameters = []
        else:
            parameters = self._extract_parameters(params_node, ctx)

        # Get return type (TypeScript)
        return_type = self._extract_return_type(node, ctx)

        # Create symbol
        symbol = Symbol(
            name=name,
            kind=SymbolKind.FUNCTION,
            location=location,
            type_info=return_type,
            is_public=not name.startswith("_"),
            parameters=parameters,
            return_type=return_type,
            is_async=is_async,
        )

        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

        # Add to call graph
        qualified_name = symbol.qualified_name or f"{ctx.module_name}.{name}"
        ctx.call_graph.add_function(
            qualified_name=qualified_name,
            display_name=name,
            file_path=ctx.file_path,
            location=location,
            is_async=is_async,
        )

        # Visit body
        body_node = self._find_child_by_type(node, "statement_block")
        if body_node:
            func_scope = ctx.symbol_table.create_scope(
                kind=ScopeKind.FUNCTION,
                name=name,
                range=self._get_location_range(body_node, ctx.file_path),
                parent_id=ctx.current_scope_id,
            )
            func_scope.function_symbol = qualified_name

            for param in parameters:
                ctx.symbol_table.add_symbol(param, func_scope.id)

            old_scope = ctx.current_scope_id
            old_function = ctx.current_function
            ctx.current_scope_id = func_scope.id
            ctx.current_function = qualified_name

            for child in body_node.children:
                self._visit_node(child, ctx)

            ctx.current_scope_id = old_scope
            ctx.current_function = old_function
        else:
            # Expression body (implicit return)
            for child in node.children:
                if child.type not in ("=>", "formal_parameters", "identifier", "async"):
                    # This is the body expression
                    old_function = ctx.current_function
                    ctx.current_function = qualified_name
                    self._visit_node(child, ctx)
                    ctx.current_function = old_function

        return symbol

    # ======================
    # Class Declarations
    # ======================

    def _visit_class_declaration(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit class declaration."""
        self._extract_class(node, ctx)

    def _visit_class(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit class expression."""
        self._extract_class(node, ctx, is_expression=True)

    def _extract_class(
        self,
        node: Node,
        ctx: 'JSVisitorContext',
        is_expression: bool = False,
    ) -> Optional[Symbol]:
        """Extract class definition."""
        # In JavaScript, class name is "identifier"; in TypeScript, it's "type_identifier"
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            name_node = self._find_child_by_type(node, "type_identifier")
        name = self._get_node_text(name_node, ctx.source_bytes) if name_node else None

        if not name and is_expression:
            name = ctx.get_anon_name("class")

        if not name:
            return None

        location = self._get_location_range(node, ctx.file_path)

        # Get base class (extends clause)
        base_classes = []
        extends_clause = self._find_child_by_type(node, "class_heritage")
        if extends_clause:
            for child in extends_clause.children:
                if child.type == "identifier":
                    base_classes.append(self._get_node_text(child, ctx.source_bytes))
                elif child.type == "member_expression":
                    base_classes.append(self._get_node_text(child, ctx.source_bytes))

        # Create symbol
        symbol = Symbol(
            name=name,
            kind=SymbolKind.CLASS,
            location=location,
            is_public=not name.startswith("_"),
            base_classes=base_classes,
        )

        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)
        qualified_name = symbol.qualified_name

        # Create class scope and visit body
        body_node = self._find_child_by_type(node, "class_body")
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

        return symbol

    def _visit_method_definition(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit class method definition."""
        name_node = self._find_child_by_type(node, "property_identifier")
        if not name_node:
            name_node = self._find_child_by_type(node, "computed_property_name")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)

        # Handle computed property names [expr]
        if name.startswith("[") and name.endswith("]"):
            name = name[1:-1]

        location = self._get_location_range(node, ctx.file_path)

        # Check for static, async, getter/setter
        is_static = False
        is_async = False
        is_getter = False
        is_setter = False

        for child in node.children:
            text = self._get_node_text(child, ctx.source_bytes)
            if text == "static":
                is_static = True
            elif text == "async":
                is_async = True
            elif text == "get":
                is_getter = True
            elif text == "set":
                is_setter = True

        # Get parameters
        params_node = self._find_child_by_type(node, "formal_parameters")
        parameters = self._extract_parameters(params_node, ctx) if params_node else []

        # Determine kind
        if is_getter or is_setter:
            kind = SymbolKind.PROPERTY
        else:
            kind = SymbolKind.METHOD

        # Create symbol
        symbol = Symbol(
            name=name,
            kind=kind,
            location=location,
            parent_symbol=ctx.current_class,
            is_public=not name.startswith("_") and not name.startswith("#"),
            parameters=parameters,
            is_async=is_async,
            metadata={"static": is_static, "getter": is_getter, "setter": is_setter},
        )

        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

        # Add to call graph
        qualified_name = symbol.qualified_name or f"{ctx.current_class}.{name}"
        ctx.call_graph.add_function(
            qualified_name=qualified_name,
            display_name=name,
            file_path=ctx.file_path,
            location=location,
            is_method=True,
            is_async=is_async,
            class_name=ctx.current_class,
        )

        # Visit body
        body_node = self._find_child_by_type(node, "statement_block")
        if body_node:
            method_scope = ctx.symbol_table.create_scope(
                kind=ScopeKind.METHOD,
                name=name,
                range=self._get_location_range(body_node, ctx.file_path),
                parent_id=ctx.current_scope_id,
            )
            method_scope.function_symbol = qualified_name

            for param in parameters:
                ctx.symbol_table.add_symbol(param, method_scope.id)

            old_scope = ctx.current_scope_id
            old_function = ctx.current_function
            ctx.current_scope_id = method_scope.id
            ctx.current_function = qualified_name

            for child in body_node.children:
                self._visit_node(child, ctx)

            ctx.current_scope_id = old_scope
            ctx.current_function = old_function

    def _visit_public_field_definition(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit class field definition."""
        self._visit_field_definition(node, ctx)

    def _visit_field_definition(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit class field definition (ES2022+)."""
        name_node = self._find_child_by_type(node, "property_identifier")
        if not name_node:
            name_node = self._find_child_by_type(node, "private_property_identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Check for static
        is_static = any(
            self._get_node_text(child, ctx.source_bytes) == "static"
            for child in node.children
        )

        # Create symbol
        symbol = Symbol(
            name=name,
            kind=SymbolKind.ATTRIBUTE,
            location=location,
            parent_symbol=ctx.current_class,
            is_public=not name.startswith("_") and not name.startswith("#"),
            metadata={"static": is_static},
        )

        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

        # Visit initializer
        for child in node.children:
            if child.type not in ("property_identifier", "private_property_identifier",
                                  "static", ";", "="):
                self._visit_node(child, ctx)

    # ======================
    # Variable Declarations
    # ======================

    def _visit_lexical_declaration(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit let/const declaration."""
        self._extract_variable_declaration(node, ctx)

    def _visit_variable_declaration(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit var declaration."""
        self._extract_variable_declaration(node, ctx)

    def _extract_variable_declaration(
        self,
        node: Node,
        ctx: 'JSVisitorContext',
    ) -> None:
        """Extract variable declaration(s)."""
        # Determine declaration kind (var, let, const)
        decl_kind = None
        for child in node.children:
            if child.type in ("var", "let", "const"):
                decl_kind = self._get_node_text(child, ctx.source_bytes)
                break

        is_constant = decl_kind == "const"

        # Visit each declarator
        for child in node.children:
            if child.type == "variable_declarator":
                self._extract_variable_declarator(child, ctx, is_constant)

    def _extract_variable_declarator(
        self,
        node: Node,
        ctx: 'JSVisitorContext',
        is_constant: bool = False,
    ) -> None:
        """Extract a single variable declarator."""
        name_node = node.children[0] if node.children else None
        if not name_node:
            return

        # Handle different binding patterns
        if name_node.type == "identifier":
            name = self._get_node_text(name_node, ctx.source_bytes)
            location = self._get_location_range(name_node, ctx.file_path)

            # Check if there's an initializer that's a function
            value_node = None
            for child in node.children:
                if child.type == "=":
                    continue
                if child != name_node:
                    value_node = child
                    break

            # Set pending name for arrow function
            if value_node and value_node.type in ("arrow_function", "function"):
                ctx.pending_var_name = name

            # Create symbol
            symbol = Symbol(
                name=name,
                kind=SymbolKind.CONSTANT if is_constant else SymbolKind.VARIABLE,
                location=location,
                is_public=not name.startswith("_"),
            )
            ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

            # Visit initializer
            if value_node:
                self._visit_node(value_node, ctx)

        elif name_node.type == "object_pattern":
            # Destructuring: const { a, b } = obj
            self._extract_destructuring_pattern(name_node, ctx, is_constant)
            # Visit initializer
            for child in node.children:
                if child not in (name_node,) and child.type != "=":
                    self._visit_node(child, ctx)

        elif name_node.type == "array_pattern":
            # Array destructuring: const [a, b] = arr
            self._extract_destructuring_pattern(name_node, ctx, is_constant)
            # Visit initializer
            for child in node.children:
                if child not in (name_node,) and child.type != "=":
                    self._visit_node(child, ctx)

    def _extract_destructuring_pattern(
        self,
        node: Node,
        ctx: 'JSVisitorContext',
        is_constant: bool = False,
    ) -> None:
        """Extract variables from destructuring pattern."""
        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, ctx.source_bytes)
                location = self._get_location_range(child, ctx.file_path)
                symbol = Symbol(
                    name=name,
                    kind=SymbolKind.CONSTANT if is_constant else SymbolKind.VARIABLE,
                    location=location,
                    is_public=not name.startswith("_"),
                )
                ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

            elif child.type in ("shorthand_property_identifier", "shorthand_property_identifier_pattern"):
                name = self._get_node_text(child, ctx.source_bytes)
                location = self._get_location_range(child, ctx.file_path)
                symbol = Symbol(
                    name=name,
                    kind=SymbolKind.CONSTANT if is_constant else SymbolKind.VARIABLE,
                    location=location,
                )
                ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

            elif child.type in ("pair_pattern", "assignment_pattern"):
                # { key: value } or { key = default }
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = self._get_node_text(subchild, ctx.source_bytes)
                        location = self._get_location_range(subchild, ctx.file_path)
                        symbol = Symbol(
                            name=name,
                            kind=SymbolKind.CONSTANT if is_constant else SymbolKind.VARIABLE,
                            location=location,
                        )
                        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)
                        break

            elif child.type == "rest_pattern":
                # ...rest
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = self._get_node_text(subchild, ctx.source_bytes)
                        location = self._get_location_range(subchild, ctx.file_path)
                        symbol = Symbol(
                            name=name,
                            kind=SymbolKind.CONSTANT if is_constant else SymbolKind.VARIABLE,
                            location=location,
                        )
                        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

            elif child.type in ("object_pattern", "array_pattern"):
                # Nested destructuring
                self._extract_destructuring_pattern(child, ctx, is_constant)

    # ======================
    # Imports/Exports
    # ======================

    def _visit_import_statement(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit import statement."""
        module = None
        location = self._get_location_range(node, ctx.file_path)

        # Find the module specifier
        for child in node.children:
            if child.type == "string":
                module = self._get_node_text(child, ctx.source_bytes).strip("'\"")
                break

        if not module:
            return

        # Process different import types
        for child in node.children:
            if child.type == "import_clause":
                self._extract_import_clause(child, ctx, module, location)
            elif child.type == "namespace_import":
                # import * as name from "module"
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = self._get_node_text(subchild, ctx.source_bytes)
                        symbol = Symbol(
                            name=name,
                            kind=SymbolKind.IMPORT,
                            location=location,
                            imported_from=module,
                            metadata={"namespace_import": True},
                        )
                        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

    def _extract_import_clause(
        self,
        node: Node,
        ctx: 'JSVisitorContext',
        module: str,
        location: LocationRange,
    ) -> None:
        """Extract imports from import clause."""
        for child in node.children:
            if child.type == "identifier":
                # Default import
                name = self._get_node_text(child, ctx.source_bytes)
                symbol = Symbol(
                    name=name,
                    kind=SymbolKind.IMPORT,
                    location=location,
                    imported_from=f"{module}.default",
                    metadata={"default_import": True},
                )
                ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

            elif child.type == "named_imports":
                # { a, b as c }
                for spec in child.children:
                    if spec.type == "import_specifier":
                        original_name = None
                        alias = None
                        for subchild in spec.children:
                            if subchild.type == "identifier":
                                if original_name is None:
                                    original_name = self._get_node_text(subchild, ctx.source_bytes)
                                else:
                                    alias = self._get_node_text(subchild, ctx.source_bytes)

                        if original_name:
                            symbol = Symbol(
                                name=alias or original_name,
                                kind=SymbolKind.IMPORT,
                                location=location,
                                imported_from=f"{module}.{original_name}",
                                import_alias=alias,
                            )
                            ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

            elif child.type == "namespace_import":
                # * as name
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = self._get_node_text(subchild, ctx.source_bytes)
                        symbol = Symbol(
                            name=name,
                            kind=SymbolKind.IMPORT,
                            location=location,
                            imported_from=module,
                            metadata={"namespace_import": True},
                        )
                        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

    def _visit_export_statement(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit export statement."""
        location = self._get_location_range(node, ctx.file_path)

        for child in node.children:
            if child.type == "declaration":
                # export function/class/const/let/var
                self._visit_node(child, ctx)

            elif child.type == "function_declaration":
                symbol = self._extract_function(child, ctx)
                if symbol:
                    symbol.is_exported = True

            elif child.type == "class_declaration":
                symbol = self._extract_class(child, ctx)
                if symbol:
                    symbol.is_exported = True

            elif child.type == "lexical_declaration":
                self._extract_variable_declaration(child, ctx)

            elif child.type == "export_clause":
                # export { a, b as c }
                for spec in child.children:
                    if spec.type == "export_specifier":
                        for subchild in spec.children:
                            if subchild.type == "identifier":
                                name = self._get_node_text(subchild, ctx.source_bytes)
                                # Mark existing symbol as exported
                                existing = ctx.symbol_table.resolve_name(name, ctx.current_scope_id)
                                if existing:
                                    existing.is_exported = True

    def _visit_export_default_declaration(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit export default statement."""
        for child in node.children:
            if child.type == "function_declaration":
                symbol = self._extract_function(child, ctx)
                if symbol:
                    symbol.is_exported = True
                    symbol.metadata["default_export"] = True
            elif child.type == "class_declaration":
                symbol = self._extract_class(child, ctx)
                if symbol:
                    symbol.is_exported = True
                    symbol.metadata["default_export"] = True
            elif child.type not in ("export", "default"):
                self._visit_node(child, ctx)

    # ======================
    # Identifiers and Calls
    # ======================

    def _visit_identifier(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit identifier reference."""
        name = self._get_node_text(node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Skip if this is in a definition context
        parent = node.parent
        if parent and parent.type in ("variable_declarator", "function_declaration",
                                       "class_declaration", "method_definition",
                                       "formal_parameters", "import_specifier"):
            # Check if this is the name being defined
            if parent.children and parent.children[0] == node:
                return

        # Skip certain keywords
        if name in ("undefined", "null", "true", "false", "this", "super",
                    "arguments", "NaN", "Infinity"):
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

    def _visit_call_expression(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit function call."""
        func_node = node.children[0] if node.children else None
        if not func_node:
            return

        callee_name = self._get_node_text(func_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Count arguments
        args_node = self._find_child_by_type(node, "arguments")
        arg_count = 0
        has_spread = False

        if args_node:
            for child in args_node.children:
                if child.type == "spread_element":
                    has_spread = True
                    arg_count += 1
                elif child.type not in ("(", ")", ","):
                    arg_count += 1

        # Try to resolve callee
        base_name = callee_name.split(".")[0]
        resolved = ctx.symbol_table.resolve_name(base_name, ctx.current_scope_id)

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
                has_star_args=has_spread,
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

    def _visit_new_expression(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit new expression (constructor call)."""
        # Similar to call expression
        self._visit_call_expression(node, ctx)

    # ======================
    # Helper Methods
    # ======================

    def _extract_parameters(self, params_node: Node, ctx: 'JSVisitorContext') -> List[Symbol]:
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

            elif child.type == "assignment_pattern":
                # parameter = default
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = self._get_node_text(subchild, ctx.source_bytes)
                        location = self._get_location_range(subchild, ctx.file_path)
                        params.append(Symbol(
                            name=name,
                            kind=SymbolKind.PARAMETER,
                            location=location,
                        ))
                        break

            elif child.type == "rest_pattern":
                # ...rest
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = self._get_node_text(subchild, ctx.source_bytes)
                        location = self._get_location_range(subchild, ctx.file_path)
                        params.append(Symbol(
                            name=name,
                            kind=SymbolKind.PARAMETER,
                            location=location,
                            metadata={"rest": True},
                        ))

            elif child.type in ("object_pattern", "array_pattern"):
                # Destructuring parameter
                location = self._get_location_range(child, ctx.file_path)
                params.append(Symbol(
                    name=ctx.get_anon_name("param"),
                    kind=SymbolKind.PARAMETER,
                    location=location,
                    metadata={"destructured": True},
                ))

            elif child.type == "required_parameter":
                # TypeScript required parameter
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = self._get_node_text(subchild, ctx.source_bytes)
                        location = self._get_location_range(subchild, ctx.file_path)
                        type_info = self._extract_type_annotation(child, ctx)
                        params.append(Symbol(
                            name=name,
                            kind=SymbolKind.PARAMETER,
                            location=location,
                            type_info=type_info,
                        ))
                        break

            elif child.type == "optional_parameter":
                # TypeScript optional parameter
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name = self._get_node_text(subchild, ctx.source_bytes)
                        location = self._get_location_range(subchild, ctx.file_path)
                        type_info = self._extract_type_annotation(child, ctx)
                        params.append(Symbol(
                            name=name,
                            kind=SymbolKind.PARAMETER,
                            location=location,
                            type_info=type_info,
                            metadata={"optional": True},
                        ))
                        break

        return params

    def _extract_type_annotation(self, node: Node, ctx: 'JSVisitorContext') -> Optional[TypeInfo]:
        """Extract TypeScript type annotation."""
        for child in node.children:
            if child.type == "type_annotation":
                type_text = self._get_node_text(child, ctx.source_bytes)
                if type_text.startswith(":"):
                    type_text = type_text[1:].strip()
                return TypeInfo(type_string=type_text)
        return None

    def _extract_return_type(self, node: Node, ctx: 'JSVisitorContext') -> Optional[TypeInfo]:
        """Extract return type from function."""
        for child in node.children:
            if child.type == "type_annotation":
                type_text = self._get_node_text(child, ctx.source_bytes)
                if type_text.startswith(":"):
                    type_text = type_text[1:].strip()
                return TypeInfo(type_string=type_text, is_callable=True)
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


class JSVisitorContext:
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

        # For tracking variable name when extracting arrow functions
        self.pending_var_name: Optional[str] = None

        # Counter for anonymous names
        self._anon_counter: Dict[str, int] = {}

    def get_anon_name(self, prefix: str) -> str:
        """Generate a unique anonymous name."""
        count = self._anon_counter.get(prefix, 0)
        self._anon_counter[prefix] = count + 1
        return f"<{prefix}_{count}>"


# TypeScript-specific visitor
class TypeScriptSymbolVisitor(JavaScriptSymbolVisitor):
    """Visitor for TypeScript code."""

    def __init__(self):
        super().__init__(language="typescript")

    def _visit_interface_declaration(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit TypeScript interface declaration."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Get extends
        extends_clause = self._find_child_by_type(node, "extends_type_clause")
        base_classes = []
        if extends_clause:
            for child in extends_clause.children:
                if child.type == "type_identifier":
                    base_classes.append(self._get_node_text(child, ctx.source_bytes))

        symbol = Symbol(
            name=name,
            kind=SymbolKind.INTERFACE,
            location=location,
            is_public=not name.startswith("_"),
            base_classes=base_classes,
        )

        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

        # Visit body for properties
        body_node = self._find_child_by_type(node, "object_type")
        if body_node:
            interface_scope = ctx.symbol_table.create_scope(
                kind=ScopeKind.CLASS,  # Treat interface as class-like scope
                name=name,
                range=self._get_location_range(body_node, ctx.file_path),
                parent_id=ctx.current_scope_id,
            )

            old_scope = ctx.current_scope_id
            old_class = ctx.current_class
            ctx.current_scope_id = interface_scope.id
            ctx.current_class = symbol.qualified_name

            for child in body_node.children:
                self._visit_node(child, ctx)

            ctx.current_scope_id = old_scope
            ctx.current_class = old_class

    def _visit_type_alias_declaration(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit TypeScript type alias declaration."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        # Get the type definition
        type_value = None
        for child in node.children:
            if child.type not in ("type", "type_identifier", "=", ";"):
                type_value = self._get_node_text(child, ctx.source_bytes)
                break

        symbol = Symbol(
            name=name,
            kind=SymbolKind.TYPE_ALIAS,
            location=location,
            is_public=not name.startswith("_"),
            type_info=TypeInfo(type_string=type_value) if type_value else None,
        )

        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)

    def _visit_enum_declaration(self, node: Node, ctx: 'JSVisitorContext') -> None:
        """Visit TypeScript enum declaration."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return

        name = self._get_node_text(name_node, ctx.source_bytes)
        location = self._get_location_range(node, ctx.file_path)

        symbol = Symbol(
            name=name,
            kind=SymbolKind.ENUM,
            location=location,
            is_public=not name.startswith("_"),
        )

        ctx.symbol_table.add_symbol(symbol, ctx.current_scope_id)
        qualified_name = symbol.qualified_name

        # Visit enum body for members
        body_node = self._find_child_by_type(node, "enum_body")
        if body_node:
            enum_scope = ctx.symbol_table.create_scope(
                kind=ScopeKind.CLASS,
                name=name,
                range=self._get_location_range(body_node, ctx.file_path),
                parent_id=ctx.current_scope_id,
            )

            old_scope = ctx.current_scope_id
            ctx.current_scope_id = enum_scope.id

            for child in body_node.children:
                if child.type == "enum_assignment":
                    member_name_node = self._find_child_by_type(child, "property_identifier")
                    if member_name_node:
                        member_name = self._get_node_text(member_name_node, ctx.source_bytes)
                        member_location = self._get_location_range(child, ctx.file_path)
                        member_symbol = Symbol(
                            name=member_name,
                            kind=SymbolKind.ENUM_MEMBER,
                            location=member_location,
                            parent_symbol=qualified_name,
                        )
                        ctx.symbol_table.add_symbol(member_symbol, enum_scope.id)

            ctx.current_scope_id = old_scope
