"""
JavaScript/TypeScript Parser - Tree-sitter based AST parser for JS/TS code.

Extracts:
- Function declarations and expressions
- Arrow functions
- Class definitions with methods
- Import/export statements
- JSDoc comments
- TypeScript interfaces and type aliases
"""

from typing import List, Optional

import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Tree

from .base_parser import BaseParser
from .models import (
    SemanticInfo,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    ParameterInfo,
    DocumentationInfo,
    Visibility,
)


class JavaScriptParser(BaseParser):
    """Tree-sitter based parser for JavaScript."""

    @property
    def language_name(self) -> str:
        return "javascript"

    def _get_language(self) -> Language:
        return Language(tsjavascript.language())

    def _extract_semantics(self, tree: Tree, source_bytes: bytes) -> SemanticInfo:
        """Extract semantic information from JavaScript AST."""
        functions = []
        classes = []
        imports = []

        root = tree.root_node

        for child in root.children:
            self._process_node(child, source_bytes, functions, classes, imports)

        return SemanticInfo(
            functions=functions,
            classes=classes,
            imports=imports,
        )

    def _process_node(
        self,
        node: Node,
        source_bytes: bytes,
        functions: List[FunctionInfo],
        classes: List[ClassInfo],
        imports: List[ImportInfo],
        parent_class: str = None
    ):
        """Process a node and extract semantic info."""
        if node.type == "function_declaration":
            func = self._extract_function(node, source_bytes)
            if func:
                functions.append(func)

        elif node.type == "class_declaration":
            cls = self._extract_class(node, source_bytes)
            if cls:
                classes.append(cls)

        elif node.type == "import_statement":
            imp = self._extract_import(node, source_bytes)
            if imp:
                imports.append(imp)

        elif node.type == "export_statement":
            # Handle export statements that contain declarations
            for child in node.children:
                self._process_node(child, source_bytes, functions, classes, imports)

        elif node.type == "lexical_declaration":
            # Handle const/let declarations that might be arrow functions
            for child in node.children:
                if child.type == "variable_declarator":
                    func = self._extract_variable_function(child, source_bytes)
                    if func:
                        functions.append(func)

        elif node.type == "variable_declaration":
            # Handle var declarations
            for child in node.children:
                if child.type == "variable_declarator":
                    func = self._extract_variable_function(child, source_bytes)
                    if func:
                        functions.append(func)

    def _extract_function(
        self,
        node: Node,
        source_bytes: bytes,
        parent_class: str = None
    ) -> Optional[FunctionInfo]:
        """Extract function from function_declaration node."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Check for async
        is_async = any(child.type == "async" for child in node.children)

        # Check for generator
        is_generator = any(child.type == "generator_function" for child in node.children)
        if not is_generator:
            # Check if function keyword is followed by *
            for i, child in enumerate(node.children):
                if child.type == "function":
                    if i + 1 < len(node.children) and node.children[i + 1].type == "*":
                        is_generator = True
                        break

        # Get parameters
        params_node = self._find_child_by_type(node, "formal_parameters")
        parameters = self._extract_parameters(params_node, source_bytes) if params_node else []

        # Get JSDoc comment
        documentation = self._find_jsdoc(node, source_bytes)

        start_line, end_line, start_col, end_col = self._get_location(node)

        return FunctionInfo(
            name=name,
            parameters=parameters,
            is_async=is_async,
            is_generator=is_generator,
            is_method=parent_class is not None,
            visibility=Visibility.PUBLIC,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
            parent_class=parent_class,
        )

    def _extract_variable_function(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[FunctionInfo]:
        """Extract function from variable_declarator (arrow function or function expression)."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Find arrow function or function expression
        arrow_func = self._find_child_by_type(node, "arrow_function")
        func_expr = self._find_child_by_type(node, "function_expression") if not arrow_func else None

        func_node = arrow_func or func_expr
        if not func_node:
            return None

        # Check for async
        is_async = any(child.type == "async" for child in func_node.children)

        # Get parameters
        params_node = self._find_child_by_type(func_node, "formal_parameters")
        parameters = self._extract_parameters(params_node, source_bytes) if params_node else []

        # Get JSDoc comment
        documentation = self._find_jsdoc(node, source_bytes)

        start_line, end_line, start_col, end_col = self._get_location(node)

        return FunctionInfo(
            name=name,
            parameters=parameters,
            is_async=is_async,
            visibility=Visibility.PUBLIC,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
        )

    def _extract_parameters(
        self,
        node: Node,
        source_bytes: bytes
    ) -> List[ParameterInfo]:
        """Extract parameters from formal_parameters node."""
        params = []

        for child in node.children:
            if child.type == "identifier":
                params.append(ParameterInfo(
                    name=self._get_node_text(child, source_bytes)
                ))

            elif child.type == "assignment_pattern":
                # Parameter with default value
                name_node = self._find_child_by_type(child, "identifier")
                if name_node:
                    # Get default value
                    default_val = None
                    for i, c in enumerate(child.children):
                        if c.type == "=" and i + 1 < len(child.children):
                            default_val = self._get_node_text(child.children[i + 1], source_bytes)
                            break

                    params.append(ParameterInfo(
                        name=self._get_node_text(name_node, source_bytes),
                        default_value=default_val,
                    ))

            elif child.type == "rest_pattern":
                # ...rest
                name_node = self._find_child_by_type(child, "identifier")
                if name_node:
                    params.append(ParameterInfo(
                        name=self._get_node_text(name_node, source_bytes),
                        is_variadic=True,
                    ))

            elif child.type == "object_pattern":
                # Destructuring parameter
                params.append(ParameterInfo(
                    name=self._get_node_text(child, source_bytes)
                ))

            elif child.type == "array_pattern":
                # Array destructuring
                params.append(ParameterInfo(
                    name=self._get_node_text(child, source_bytes)
                ))

        return params

    def _extract_class(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[ClassInfo]:
        """Extract class from class_declaration node."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Get base class (extends)
        bases = []
        heritage = self._find_child_by_type(node, "class_heritage")
        if heritage:
            for child in heritage.children:
                if child.type == "identifier":
                    bases.append(self._get_node_text(child, source_bytes))

        # Get methods
        methods = []
        body_node = self._find_child_by_type(node, "class_body")
        if body_node:
            for child in body_node.children:
                if child.type == "method_definition":
                    method = self._extract_method(child, source_bytes, name)
                    if method:
                        methods.append(method)

                elif child.type == "field_definition":
                    # Check if it's an arrow function field
                    func = self._extract_field_function(child, source_bytes, name)
                    if func:
                        methods.append(func)

        # Get JSDoc
        documentation = self._find_jsdoc(node, source_bytes)

        start_line, end_line, _, _ = self._get_location(node)

        return ClassInfo(
            name=name,
            kind="class",
            bases=bases,
            methods=methods,
            visibility=Visibility.PUBLIC,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_method(
        self,
        node: Node,
        source_bytes: bytes,
        parent_class: str
    ) -> Optional[FunctionInfo]:
        """Extract method from method_definition node."""
        name_node = self._find_child_by_type(node, "property_identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Check for async/generator/static
        is_async = any(child.type == "async" for child in node.children)
        is_generator = any(child.type == "*" for child in node.children)
        is_static = any(child.type == "static" for child in node.children)

        # Determine visibility from name
        visibility = Visibility.PRIVATE if name.startswith("_") or name.startswith("#") else Visibility.PUBLIC

        # Get parameters
        params_node = self._find_child_by_type(node, "formal_parameters")
        parameters = self._extract_parameters(params_node, source_bytes) if params_node else []

        # Get JSDoc
        documentation = self._find_jsdoc(node, source_bytes)

        start_line, end_line, start_col, end_col = self._get_location(node)

        return FunctionInfo(
            name=name,
            parameters=parameters,
            is_async=is_async,
            is_generator=is_generator,
            is_method=True,
            is_static=is_static,
            visibility=visibility,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
            parent_class=parent_class,
        )

    def _extract_field_function(
        self,
        node: Node,
        source_bytes: bytes,
        parent_class: str
    ) -> Optional[FunctionInfo]:
        """Extract arrow function from class field."""
        name_node = self._find_child_by_type(node, "property_identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Find arrow function
        arrow_func = self._find_child_by_type(node, "arrow_function")
        if not arrow_func:
            return None

        is_async = any(child.type == "async" for child in arrow_func.children)
        is_static = any(child.type == "static" for child in node.children)

        params_node = self._find_child_by_type(arrow_func, "formal_parameters")
        parameters = self._extract_parameters(params_node, source_bytes) if params_node else []

        start_line, end_line, start_col, end_col = self._get_location(node)

        return FunctionInfo(
            name=name,
            parameters=parameters,
            is_async=is_async,
            is_method=True,
            is_static=is_static,
            visibility=Visibility.PRIVATE if name.startswith("_") else Visibility.PUBLIC,
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
            parent_class=parent_class,
        )

    def _extract_import(self, node: Node, source_bytes: bytes) -> Optional[ImportInfo]:
        """Extract import statement."""
        module = ""
        names = []
        alias = None
        is_wildcard = False

        for child in node.children:
            if child.type == "string":
                # Module path
                module = self._get_node_text(child, source_bytes).strip("'\"")

            elif child.type == "import_clause":
                for import_child in child.children:
                    if import_child.type == "identifier":
                        # Default import
                        names.append(self._get_node_text(import_child, source_bytes))

                    elif import_child.type == "namespace_import":
                        # import * as X
                        is_wildcard = True
                        alias_node = self._find_child_by_type(import_child, "identifier")
                        if alias_node:
                            alias = self._get_node_text(alias_node, source_bytes)

                    elif import_child.type == "named_imports":
                        # import { a, b }
                        for spec in import_child.children:
                            if spec.type == "import_specifier":
                                name_node = self._find_child_by_type(spec, "identifier")
                                if name_node:
                                    names.append(self._get_node_text(name_node, source_bytes))

        start_line, end_line, _, _ = self._get_location(node)
        return ImportInfo(
            module=module,
            names=names,
            alias=alias,
            is_wildcard=is_wildcard,
            start_line=start_line,
            end_line=end_line,
        )

    def _find_jsdoc(self, node: Node, source_bytes: bytes) -> Optional[DocumentationInfo]:
        """Find JSDoc comment preceding a node."""
        # Look for comment in previous sibling
        if node.prev_sibling and node.prev_sibling.type == "comment":
            comment_text = self._get_node_text(node.prev_sibling, source_bytes)
            if comment_text.startswith("/**"):
                # Strip comment markers
                content = comment_text[3:-2].strip() if comment_text.endswith("*/") else comment_text[3:].strip()
                # Clean up * at start of lines
                lines = content.split("\n")
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line.startswith("*"):
                        line = line[1:].strip()
                    cleaned_lines.append(line)

                start_line, end_line, _, _ = self._get_location(node.prev_sibling)
                return DocumentationInfo(
                    content="\n".join(cleaned_lines),
                    format="jsdoc",
                    start_line=start_line,
                    end_line=end_line,
                )

        return None


class TypeScriptParser(JavaScriptParser):
    """Tree-sitter based parser for TypeScript."""

    @property
    def language_name(self) -> str:
        return "typescript"

    def _get_language(self) -> Language:
        return Language(tstypescript.language_typescript())

    def _extract_semantics(self, tree: Tree, source_bytes: bytes) -> SemanticInfo:
        """Extract semantic information from TypeScript AST."""
        # Get base JS semantics
        semantic = super()._extract_semantics(tree, source_bytes)

        # Add TypeScript-specific: interfaces and type aliases
        for child in tree.root_node.children:
            if child.type == "interface_declaration":
                interface = self._extract_interface(child, source_bytes)
                if interface:
                    semantic.classes.append(interface)

            elif child.type == "type_alias_declaration":
                # Could add to a separate types list, but for now treat as class-like
                type_alias = self._extract_type_alias(child, source_bytes)
                if type_alias:
                    semantic.classes.append(type_alias)

            elif child.type == "export_statement":
                # Check for exported interfaces/types
                for export_child in child.children:
                    if export_child.type == "interface_declaration":
                        interface = self._extract_interface(export_child, source_bytes)
                        if interface:
                            semantic.classes.append(interface)
                    elif export_child.type == "type_alias_declaration":
                        type_alias = self._extract_type_alias(export_child, source_bytes)
                        if type_alias:
                            semantic.classes.append(type_alias)

        return semantic

    def _extract_interface(self, node: Node, source_bytes: bytes) -> Optional[ClassInfo]:
        """Extract TypeScript interface."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Get extends
        bases = []
        extends_clause = self._find_child_by_type(node, "extends_type_clause")
        if extends_clause:
            for child in extends_clause.children:
                if child.type == "type_identifier":
                    bases.append(self._get_node_text(child, source_bytes))

        # Get methods from interface body
        methods = []
        body = self._find_child_by_type(node, "interface_body") or self._find_child_by_type(node, "object_type")
        if body:
            for child in body.children:
                if child.type == "method_signature":
                    method = self._extract_method_signature(child, source_bytes, name)
                    if method:
                        methods.append(method)

        start_line, end_line, _, _ = self._get_location(node)

        return ClassInfo(
            name=name,
            kind="interface",
            bases=bases,
            methods=methods,
            visibility=Visibility.PUBLIC,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_type_alias(self, node: Node, source_bytes: bytes) -> Optional[ClassInfo]:
        """Extract TypeScript type alias."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)
        start_line, end_line, _, _ = self._get_location(node)

        return ClassInfo(
            name=name,
            kind="type",
            visibility=Visibility.PUBLIC,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_method_signature(
        self,
        node: Node,
        source_bytes: bytes,
        parent_class: str
    ) -> Optional[FunctionInfo]:
        """Extract method signature from interface."""
        name_node = self._find_child_by_type(node, "property_identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Get parameters
        params_node = self._find_child_by_type(node, "formal_parameters")
        parameters = self._extract_parameters(params_node, source_bytes) if params_node else []

        # Get return type
        return_type = None
        type_annotation = self._find_child_by_type(node, "type_annotation")
        if type_annotation:
            for child in type_annotation.children:
                if child.type not in (":", ):
                    return_type = self._get_node_text(child, source_bytes)
                    break

        start_line, end_line, start_col, end_col = self._get_location(node)

        return FunctionInfo(
            name=name,
            parameters=parameters,
            return_type=return_type,
            is_method=True,
            visibility=Visibility.PUBLIC,
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
            parent_class=parent_class,
        )

    def _extract_parameters(
        self,
        node: Node,
        source_bytes: bytes
    ) -> List[ParameterInfo]:
        """Extract parameters with TypeScript type annotations."""
        params = super()._extract_parameters(node, source_bytes)

        # Enhance with type annotations
        idx = 0
        for child in node.children:
            if child.type == "required_parameter" or child.type == "optional_parameter":
                name_node = self._find_child_by_type(child, "identifier")
                type_annotation = self._find_child_by_type(child, "type_annotation")

                if name_node and idx < len(params):
                    if type_annotation:
                        for type_child in type_annotation.children:
                            if type_child.type not in (":", ):
                                params[idx].type_annotation = self._get_node_text(type_child, source_bytes)
                                break
                    idx += 1

        return params
