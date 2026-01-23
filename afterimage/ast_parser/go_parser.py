"""
Go Parser - Tree-sitter based AST parser for Go code.

Extracts:
- Function definitions (func)
- Method definitions (func with receiver)
- Struct definitions
- Interface definitions
- Import statements
- Documentation comments
"""

from typing import List, Optional

import tree_sitter_go as tsgo
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


class GoParser(BaseParser):
    """Tree-sitter based parser for Go."""

    @property
    def language_name(self) -> str:
        return "go"

    def _get_language(self) -> Language:
        return Language(tsgo.language())

    def _extract_semantics(self, tree: Tree, source_bytes: bytes) -> SemanticInfo:
        """Extract semantic information from Go AST."""
        functions = []
        classes = []  # structs, interfaces
        imports = []
        module_doc = None

        root = tree.root_node

        # Check for package documentation (comment before package declaration)
        for i, child in enumerate(root.children):
            if child.type == "comment":
                # Check if this is a doc comment (before package)
                content = self._get_node_text(child, source_bytes)
                if content.startswith("//") or content.startswith("/*"):
                    module_doc = self._extract_doc_comment(child, source_bytes)
            elif child.type == "package_clause":
                break

        # Process all top-level declarations
        for child in root.children:
            if child.type == "function_declaration":
                func = self._extract_function(child, source_bytes)
                if func:
                    functions.append(func)

            elif child.type == "method_declaration":
                func = self._extract_method(child, source_bytes)
                if func:
                    functions.append(func)

            elif child.type == "type_declaration":
                # Handle type declarations (struct, interface, type alias)
                for spec in child.children:
                    if spec.type == "type_spec":
                        cls = self._extract_type_spec(spec, source_bytes)
                        if cls:
                            classes.append(cls)

            elif child.type == "import_declaration":
                imps = self._extract_imports(child, source_bytes)
                imports.extend(imps)

        return SemanticInfo(
            functions=functions,
            classes=classes,
            imports=imports,
            module_doc=module_doc,
        )

    def _extract_function(
        self,
        node: Node,
        source_bytes: bytes,
        receiver_type: str = None
    ) -> Optional[FunctionInfo]:
        """Extract function information from a function_declaration node."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Get parameters
        params_node = self._find_child_by_type(node, "parameter_list")
        parameters = self._extract_parameters(params_node, source_bytes) if params_node else []

        # Get return type
        return_type = None
        result_node = self._find_child_by_type(node, "result")
        if result_node is None:
            # Check for simple return type (single type without parentheses)
            for child in node.children:
                if child.type in ("type_identifier", "pointer_type", "slice_type",
                                  "map_type", "channel_type", "qualified_type"):
                    return_type = self._get_node_text(child, source_bytes)
                    break
        else:
            return_type = self._get_node_text(result_node, source_bytes)

        # Get documentation (comment preceding the function)
        documentation = self._get_preceding_doc(node, source_bytes)

        # Determine visibility (Go uses capitalization)
        visibility = Visibility.PUBLIC if name[0].isupper() else Visibility.PRIVATE

        start_line, end_line, start_col, end_col = self._get_location(node)

        return FunctionInfo(
            name=name,
            parameters=parameters,
            return_type=return_type,
            is_async=False,  # Go doesn't have async keyword
            is_generator=False,
            is_method=receiver_type is not None,
            visibility=visibility,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
            parent_class=receiver_type,
        )

    def _extract_method(self, node: Node, source_bytes: bytes) -> Optional[FunctionInfo]:
        """Extract method information from a method_declaration node."""
        # Get receiver type
        receiver_type = None
        receiver_node = self._find_child_by_type(node, "parameter_list")
        if receiver_node:
            # First parameter list is the receiver
            for child in receiver_node.children:
                if child.type == "parameter_declaration":
                    type_node = self._find_child_by_type(child, "type_identifier")
                    if type_node is None:
                        # Check for pointer receiver
                        pointer_node = self._find_child_by_type(child, "pointer_type")
                        if pointer_node:
                            type_node = self._find_child_by_type(pointer_node, "type_identifier")
                    if type_node:
                        receiver_type = self._get_node_text(type_node, source_bytes)
                    break

        # Get function name
        name_node = self._find_child_by_type(node, "field_identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Get parameters (second parameter_list is the actual params)
        param_lists = self._find_children_by_type(node, "parameter_list")
        parameters = []
        if len(param_lists) >= 2:
            parameters = self._extract_parameters(param_lists[1], source_bytes)

        # Get return type
        return_type = None
        result_node = self._find_child_by_type(node, "result")
        if result_node is None:
            for child in node.children:
                if child.type in ("type_identifier", "pointer_type", "slice_type",
                                  "map_type", "channel_type", "qualified_type"):
                    return_type = self._get_node_text(child, source_bytes)
                    break
        else:
            return_type = self._get_node_text(result_node, source_bytes)

        # Get documentation
        documentation = self._get_preceding_doc(node, source_bytes)

        # Determine visibility
        visibility = Visibility.PUBLIC if name[0].isupper() else Visibility.PRIVATE

        start_line, end_line, start_col, end_col = self._get_location(node)

        return FunctionInfo(
            name=name,
            parameters=parameters,
            return_type=return_type,
            is_method=True,
            visibility=visibility,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
            parent_class=receiver_type,
        )

    def _extract_parameters(
        self,
        node: Node,
        source_bytes: bytes
    ) -> List[ParameterInfo]:
        """Extract parameters from a parameter_list node."""
        params = []

        for child in node.children:
            if child.type == "parameter_declaration":
                # Get parameter names and type
                names = []
                type_str = None

                for param_child in child.children:
                    if param_child.type == "identifier":
                        names.append(self._get_node_text(param_child, source_bytes))
                    elif param_child.type in ("type_identifier", "pointer_type", "slice_type",
                                               "map_type", "channel_type", "qualified_type",
                                               "array_type", "struct_type", "interface_type",
                                               "function_type"):
                        type_str = self._get_node_text(param_child, source_bytes)
                    elif param_child.type == "variadic_parameter_declaration":
                        # Handle ...T
                        name_node = self._find_child_by_type(param_child, "identifier")
                        if name_node:
                            params.append(ParameterInfo(
                                name=self._get_node_text(name_node, source_bytes),
                                type_annotation="..." + (type_str or ""),
                                is_variadic=True,
                            ))
                        continue

                # Create parameter for each name
                for name in names:
                    params.append(ParameterInfo(
                        name=name,
                        type_annotation=type_str,
                    ))

                # Handle case where type is specified without name
                if not names and type_str:
                    params.append(ParameterInfo(
                        name="_",
                        type_annotation=type_str,
                    ))

            elif child.type == "variadic_parameter_declaration":
                name_node = self._find_child_by_type(child, "identifier")
                type_node = None
                for c in child.children:
                    if c.type in ("type_identifier", "qualified_type"):
                        type_node = c
                        break

                if name_node:
                    params.append(ParameterInfo(
                        name=self._get_node_text(name_node, source_bytes),
                        type_annotation="..." + (self._get_node_text(type_node, source_bytes) if type_node else ""),
                        is_variadic=True,
                    ))

        return params

    def _extract_type_spec(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[ClassInfo]:
        """Extract type specification (struct, interface, etc.)."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Determine type kind
        kind = "type"
        fields = []
        methods = []

        for child in node.children:
            if child.type == "struct_type":
                kind = "struct"
                fields = self._extract_struct_fields(child, source_bytes)

            elif child.type == "interface_type":
                kind = "interface"
                methods = self._extract_interface_methods(child, source_bytes)

        # Get documentation
        documentation = self._get_preceding_doc(node.parent, source_bytes)

        # Determine visibility
        visibility = Visibility.PUBLIC if name[0].isupper() else Visibility.PRIVATE

        start_line, end_line, _, _ = self._get_location(node)

        return ClassInfo(
            name=name,
            kind=kind,
            methods=methods,
            fields=fields,
            visibility=visibility,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_struct_fields(self, node: Node, source_bytes: bytes) -> List[dict]:
        """Extract fields from a struct_type node."""
        fields = []

        for child in node.children:
            if child.type == "field_declaration_list":
                for field_decl in child.children:
                    if field_decl.type == "field_declaration":
                        field_names = []
                        field_type = None
                        tag = None

                        for fc in field_decl.children:
                            if fc.type == "field_identifier":
                                field_names.append(self._get_node_text(fc, source_bytes))
                            elif fc.type in ("type_identifier", "pointer_type", "slice_type",
                                             "map_type", "qualified_type", "array_type"):
                                field_type = self._get_node_text(fc, source_bytes)
                            elif fc.type == "raw_string_literal" or fc.type == "interpreted_string_literal":
                                tag = self._get_node_text(fc, source_bytes)

                        for fname in field_names:
                            fields.append({
                                "name": fname,
                                "type": field_type,
                                "tag": tag,
                                "visibility": "public" if fname[0].isupper() else "private",
                            })

        return fields

    def _extract_interface_methods(self, node: Node, source_bytes: bytes) -> List[FunctionInfo]:
        """Extract method signatures from an interface_type node."""
        methods = []

        for child in node.children:
            if child.type == "method_elem":
                # Method signature in interface
                name_node = self._find_child_by_type(child, "field_identifier")
                if name_node:
                    name = self._get_node_text(name_node, source_bytes)

                    # Get parameters
                    params_node = self._find_child_by_type(child, "parameter_list")
                    parameters = self._extract_parameters(params_node, source_bytes) if params_node else []

                    # Get return type
                    return_type = None
                    result_node = self._find_child_by_type(child, "result")
                    if result_node:
                        return_type = self._get_node_text(result_node, source_bytes)

                    start_line, end_line, start_col, end_col = self._get_location(child)

                    methods.append(FunctionInfo(
                        name=name,
                        parameters=parameters,
                        return_type=return_type,
                        is_method=True,
                        visibility=Visibility.PUBLIC if name[0].isupper() else Visibility.PRIVATE,
                        start_line=start_line,
                        end_line=end_line,
                    ))

        return methods

    def _extract_imports(self, node: Node, source_bytes: bytes) -> List[ImportInfo]:
        """Extract imports from an import_declaration node."""
        imports = []

        for child in node.children:
            if child.type == "import_spec":
                imp = self._extract_import_spec(child, source_bytes)
                if imp:
                    imports.append(imp)
            elif child.type == "import_spec_list":
                for spec in child.children:
                    if spec.type == "import_spec":
                        imp = self._extract_import_spec(spec, source_bytes)
                        if imp:
                            imports.append(imp)

        return imports

    def _extract_import_spec(self, node: Node, source_bytes: bytes) -> Optional[ImportInfo]:
        """Extract a single import specification."""
        path_node = self._find_child_by_type(node, "interpreted_string_literal")
        if not path_node:
            return None

        module = self._get_node_text(path_node, source_bytes).strip('"')

        # Check for alias
        alias = None
        alias_node = self._find_child_by_type(node, "package_identifier")
        if alias_node:
            alias = self._get_node_text(alias_node, source_bytes)

        # Check for dot import
        is_wildcard = False
        for child in node.children:
            if child.type == "dot":
                is_wildcard = True
                break

        start_line, end_line, _, _ = self._get_location(node)

        return ImportInfo(
            module=module,
            alias=alias,
            is_wildcard=is_wildcard,
            start_line=start_line,
            end_line=end_line,
        )

    def _get_preceding_doc(self, node: Node, source_bytes: bytes) -> Optional[DocumentationInfo]:
        """Get documentation comment preceding a node."""
        if node.prev_sibling and node.prev_sibling.type == "comment":
            return self._extract_doc_comment(node.prev_sibling, source_bytes)
        return None

    def _extract_doc_comment(self, node: Node, source_bytes: bytes) -> DocumentationInfo:
        """Extract documentation from a comment node."""
        content = self._get_node_text(node, source_bytes)

        # Clean up comment syntax
        if content.startswith("//"):
            content = content[2:].strip()
        elif content.startswith("/*") and content.endswith("*/"):
            content = content[2:-2].strip()

        start_line, end_line, _, _ = self._get_location(node)

        return DocumentationInfo(
            content=content,
            format="godoc",
            start_line=start_line,
            end_line=end_line,
        )
