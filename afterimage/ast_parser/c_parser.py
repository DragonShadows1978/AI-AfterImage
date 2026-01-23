"""
C and C++ Parsers - Tree-sitter based AST parsers for C/C++ code.

Extracts:
- Function definitions and declarations
- Struct, class, and union definitions
- Include directives
- Documentation comments
- Enum definitions
"""

from typing import List, Optional

import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
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


class CParser(BaseParser):
    """Tree-sitter based parser for C."""

    @property
    def language_name(self) -> str:
        return "c"

    def _get_language(self) -> Language:
        return Language(tsc.language())

    def _extract_semantics(self, tree: Tree, source_bytes: bytes) -> SemanticInfo:
        """Extract semantic information from C AST."""
        functions = []
        classes = []  # structs, unions, enums
        imports = []  # includes
        module_doc = None

        root = tree.root_node

        # Check for file-level documentation comment at the start
        for child in root.children:
            if child.type == "comment":
                content = self._get_node_text(child, source_bytes)
                if content.startswith("/*") or content.startswith("//"):
                    module_doc = self._extract_doc_comment(child, source_bytes)
                    break
            elif child.type != "comment":
                break

        # Process all top-level nodes
        for child in root.children:
            if child.type == "function_definition":
                func = self._extract_function(child, source_bytes)
                if func:
                    functions.append(func)

            elif child.type == "declaration":
                # Could be a function declaration or variable declaration
                func = self._extract_function_declaration(child, source_bytes)
                if func:
                    functions.append(func)
                else:
                    # Check for struct/union/enum declaration
                    cls = self._extract_type_declaration(child, source_bytes)
                    if cls:
                        classes.append(cls)

            elif child.type == "struct_specifier":
                cls = self._extract_struct(child, source_bytes)
                if cls:
                    classes.append(cls)

            elif child.type == "enum_specifier":
                cls = self._extract_enum(child, source_bytes)
                if cls:
                    classes.append(cls)

            elif child.type == "preproc_include":
                imp = self._extract_include(child, source_bytes)
                if imp:
                    imports.append(imp)

            elif child.type == "type_definition":
                cls = self._extract_typedef(child, source_bytes)
                if cls:
                    classes.append(cls)

        return SemanticInfo(
            functions=functions,
            classes=classes,
            imports=imports,
            module_doc=module_doc,
        )

    def _extract_function(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[FunctionInfo]:
        """Extract function information from a function_definition node."""
        # Get declarator which contains function name and parameters
        declarator = self._find_child_by_type(node, "function_declarator")
        if declarator is None:
            # Try to find it nested
            for child in node.children:
                if child.type == "function_declarator":
                    declarator = child
                    break
                elif child.type == "pointer_declarator":
                    declarator = self._find_child_by_type(child, "function_declarator")
                    if declarator:
                        break

        if not declarator:
            return None

        # Get function name
        name_node = self._find_child_by_type(declarator, "identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Get parameters
        params_node = self._find_child_by_type(declarator, "parameter_list")
        parameters = self._extract_parameters(params_node, source_bytes) if params_node else []

        # Get return type (everything before the declarator)
        return_type = self._extract_return_type(node, declarator, source_bytes)

        # Get documentation
        documentation = self._get_preceding_doc(node, source_bytes)

        # Determine visibility (static = private in C context)
        visibility = Visibility.PUBLIC
        for child in node.children:
            if child.type == "storage_class_specifier":
                specifier = self._get_node_text(child, source_bytes)
                if specifier == "static":
                    visibility = Visibility.PRIVATE
                    break

        start_line, end_line, start_col, end_col = self._get_location(node)

        return FunctionInfo(
            name=name,
            parameters=parameters,
            return_type=return_type,
            visibility=visibility,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
        )

    def _extract_function_declaration(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[FunctionInfo]:
        """Extract function declaration (prototype)."""
        # Look for function_declarator in declaration
        declarator = None
        for child in node.children:
            if child.type == "function_declarator":
                declarator = child
                break
            elif child.type == "init_declarator":
                declarator = self._find_child_by_type(child, "function_declarator")
                if declarator:
                    break

        if not declarator:
            return None

        name_node = self._find_child_by_type(declarator, "identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        params_node = self._find_child_by_type(declarator, "parameter_list")
        parameters = self._extract_parameters(params_node, source_bytes) if params_node else []

        return_type = self._extract_return_type(node, declarator, source_bytes)
        documentation = self._get_preceding_doc(node, source_bytes)

        start_line, end_line, start_col, end_col = self._get_location(node)

        return FunctionInfo(
            name=name,
            parameters=parameters,
            return_type=return_type,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
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
                name = None
                type_str = None

                # Find parameter name (identifier or declarator)
                for pc in child.children:
                    if pc.type == "identifier":
                        name = self._get_node_text(pc, source_bytes)
                    elif pc.type == "pointer_declarator":
                        id_node = self._find_child_by_type(pc, "identifier")
                        if id_node:
                            name = self._get_node_text(id_node, source_bytes)
                    elif pc.type == "array_declarator":
                        id_node = self._find_child_by_type(pc, "identifier")
                        if id_node:
                            name = self._get_node_text(id_node, source_bytes)

                # Get type (everything except the name)
                type_parts = []
                for pc in child.children:
                    if pc.type in ("primitive_type", "type_identifier", "sized_type_specifier",
                                   "struct_specifier", "enum_specifier", "type_qualifier"):
                        type_parts.append(self._get_node_text(pc, source_bytes))

                type_str = " ".join(type_parts) if type_parts else None

                if name or type_str:
                    params.append(ParameterInfo(
                        name=name or "_",
                        type_annotation=type_str,
                    ))

            elif child.type == "variadic_parameter":
                params.append(ParameterInfo(
                    name="...",
                    is_variadic=True,
                ))

        return params

    def _extract_return_type(
        self,
        func_node: Node,
        declarator: Node,
        source_bytes: bytes
    ) -> Optional[str]:
        """Extract return type from function definition/declaration."""
        type_parts = []
        for child in func_node.children:
            if child == declarator:
                break
            if child.type in ("primitive_type", "type_identifier", "sized_type_specifier",
                              "struct_specifier", "enum_specifier", "type_qualifier",
                              "storage_class_specifier"):
                text = self._get_node_text(child, source_bytes)
                if text not in ("static", "extern", "inline"):
                    type_parts.append(text)

        return " ".join(type_parts) if type_parts else None

    def _extract_struct(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[ClassInfo]:
        """Extract struct definition."""
        # Get struct name
        name = None
        name_node = self._find_child_by_type(node, "type_identifier")
        if name_node:
            name = self._get_node_text(name_node, source_bytes)

        if not name:
            return None

        # Get fields
        fields = []
        field_list = self._find_child_by_type(node, "field_declaration_list")
        if field_list:
            fields = self._extract_struct_fields(field_list, source_bytes)

        documentation = self._get_preceding_doc(node, source_bytes)

        start_line, end_line, _, _ = self._get_location(node)

        return ClassInfo(
            name=name,
            kind="struct",
            fields=fields,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_struct_fields(self, node: Node, source_bytes: bytes) -> List[dict]:
        """Extract fields from a field_declaration_list node."""
        fields = []

        for child in node.children:
            if child.type == "field_declaration":
                field_names = []
                field_type = None

                for fc in child.children:
                    if fc.type == "field_identifier":
                        field_names.append(self._get_node_text(fc, source_bytes))
                    elif fc.type in ("primitive_type", "type_identifier", "sized_type_specifier"):
                        field_type = self._get_node_text(fc, source_bytes)
                    elif fc.type == "pointer_declarator":
                        id_node = self._find_child_by_type(fc, "field_identifier")
                        if id_node:
                            field_names.append(self._get_node_text(id_node, source_bytes))

                for fname in field_names:
                    fields.append({
                        "name": fname,
                        "type": field_type,
                    })

        return fields

    def _extract_enum(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[ClassInfo]:
        """Extract enum definition."""
        name = None
        name_node = self._find_child_by_type(node, "type_identifier")
        if name_node:
            name = self._get_node_text(name_node, source_bytes)

        if not name:
            return None

        # Get enum values
        fields = []
        enumerator_list = self._find_child_by_type(node, "enumerator_list")
        if enumerator_list:
            for child in enumerator_list.children:
                if child.type == "enumerator":
                    id_node = self._find_child_by_type(child, "identifier")
                    if id_node:
                        fields.append({
                            "name": self._get_node_text(id_node, source_bytes),
                            "type": "enum_value",
                        })

        documentation = self._get_preceding_doc(node, source_bytes)

        start_line, end_line, _, _ = self._get_location(node)

        return ClassInfo(
            name=name,
            kind="enum",
            fields=fields,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_type_declaration(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[ClassInfo]:
        """Extract struct/union/enum from a declaration."""
        for child in node.children:
            if child.type == "struct_specifier":
                return self._extract_struct(child, source_bytes)
            elif child.type == "enum_specifier":
                return self._extract_enum(child, source_bytes)
            elif child.type == "union_specifier":
                return self._extract_union(child, source_bytes)
        return None

    def _extract_union(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[ClassInfo]:
        """Extract union definition."""
        name = None
        name_node = self._find_child_by_type(node, "type_identifier")
        if name_node:
            name = self._get_node_text(name_node, source_bytes)

        if not name:
            return None

        fields = []
        field_list = self._find_child_by_type(node, "field_declaration_list")
        if field_list:
            fields = self._extract_struct_fields(field_list, source_bytes)

        documentation = self._get_preceding_doc(node, source_bytes)

        start_line, end_line, _, _ = self._get_location(node)

        return ClassInfo(
            name=name,
            kind="union",
            fields=fields,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_typedef(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[ClassInfo]:
        """Extract typedef declaration."""
        # Find the type identifier being defined
        name = None
        for child in node.children:
            if child.type == "type_identifier":
                name = self._get_node_text(child, source_bytes)

        if not name:
            return None

        documentation = self._get_preceding_doc(node, source_bytes)

        start_line, end_line, _, _ = self._get_location(node)

        return ClassInfo(
            name=name,
            kind="typedef",
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_include(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[ImportInfo]:
        """Extract #include directive."""
        path = None

        for child in node.children:
            if child.type == "string_literal":
                path = self._get_node_text(child, source_bytes).strip('"')
            elif child.type == "system_lib_string":
                path = self._get_node_text(child, source_bytes).strip('<>')

        if not path:
            return None

        start_line, end_line, _, _ = self._get_location(node)

        return ImportInfo(
            module=path,
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
            format="plain",
            start_line=start_line,
            end_line=end_line,
        )


class CppParser(CParser):
    """Tree-sitter based parser for C++."""

    @property
    def language_name(self) -> str:
        return "cpp"

    def _get_language(self) -> Language:
        return Language(tscpp.language())

    def _extract_semantics(self, tree: Tree, source_bytes: bytes) -> SemanticInfo:
        """Extract semantic information from C++ AST."""
        # Start with C extraction
        semantic = super()._extract_semantics(tree, source_bytes)

        # Add C++ specific extraction
        root = tree.root_node
        additional_classes = []

        for child in root.children:
            if child.type == "class_specifier":
                cls = self._extract_class(child, source_bytes)
                if cls:
                    additional_classes.append(cls)

            elif child.type == "namespace_definition":
                # Extract classes/functions from namespace
                ns_classes, ns_functions = self._extract_namespace_contents(child, source_bytes)
                additional_classes.extend(ns_classes)
                semantic.functions.extend(ns_functions)

            elif child.type == "template_declaration":
                # Extract template class or function
                tmpl_result = self._extract_template(child, source_bytes)
                if isinstance(tmpl_result, ClassInfo):
                    additional_classes.append(tmpl_result)
                elif isinstance(tmpl_result, FunctionInfo):
                    semantic.functions.append(tmpl_result)

        semantic.classes.extend(additional_classes)
        return semantic

    def _extract_class(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[ClassInfo]:
        """Extract C++ class definition."""
        name = None
        name_node = self._find_child_by_type(node, "type_identifier")
        if name_node:
            name = self._get_node_text(name_node, source_bytes)

        if not name:
            return None

        # Get base classes
        bases = []
        base_clause = self._find_child_by_type(node, "base_class_clause")
        if base_clause:
            for child in base_clause.children:
                if child.type == "type_identifier":
                    bases.append(self._get_node_text(child, source_bytes))
                elif child.type == "qualified_identifier":
                    bases.append(self._get_node_text(child, source_bytes))

        # Get methods and fields
        methods = []
        fields = []
        field_list = self._find_child_by_type(node, "field_declaration_list")
        if field_list:
            methods, fields = self._extract_class_members(field_list, source_bytes, name)

        documentation = self._get_preceding_doc(node, source_bytes)

        # Check for abstract class (has pure virtual functions)
        is_abstract = any(getattr(m, 'is_abstract', False) for m in methods)

        start_line, end_line, _, _ = self._get_location(node)

        return ClassInfo(
            name=name,
            kind="class",
            bases=bases,
            methods=methods,
            fields=fields,
            documentation=documentation,
            is_abstract=is_abstract,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_class_members(
        self,
        node: Node,
        source_bytes: bytes,
        class_name: str
    ) -> tuple:
        """Extract class members (methods and fields)."""
        methods = []
        fields = []
        current_visibility = Visibility.PRIVATE  # C++ default

        for child in node.children:
            if child.type == "access_specifier":
                spec = self._get_node_text(child, source_bytes).rstrip(":")
                if spec == "public":
                    current_visibility = Visibility.PUBLIC
                elif spec == "protected":
                    current_visibility = Visibility.PROTECTED
                elif spec == "private":
                    current_visibility = Visibility.PRIVATE

            elif child.type == "function_definition":
                func = self._extract_function(child, source_bytes)
                if func:
                    func.is_method = True
                    func.visibility = current_visibility
                    func.parent_class = class_name
                    methods.append(func)

            elif child.type == "declaration":
                # Could be a method declaration or field
                func = self._extract_function_declaration(child, source_bytes)
                if func:
                    func.is_method = True
                    func.visibility = current_visibility
                    func.parent_class = class_name
                    methods.append(func)
                else:
                    # Field declaration
                    field = self._extract_field_declaration(child, source_bytes, current_visibility)
                    if field:
                        fields.append(field)

        return methods, fields

    def _extract_field_declaration(
        self,
        node: Node,
        source_bytes: bytes,
        visibility: Visibility
    ) -> Optional[dict]:
        """Extract a class field declaration."""
        name = None
        type_str = None

        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source_bytes)
            elif child.type in ("primitive_type", "type_identifier", "template_type"):
                type_str = self._get_node_text(child, source_bytes)
            elif child.type == "init_declarator":
                id_node = self._find_child_by_type(child, "identifier")
                if id_node:
                    name = self._get_node_text(id_node, source_bytes)

        if name:
            return {
                "name": name,
                "type": type_str,
                "visibility": visibility.value,
            }
        return None

    def _extract_namespace_contents(
        self,
        node: Node,
        source_bytes: bytes
    ) -> tuple:
        """Extract classes and functions from a namespace."""
        classes = []
        functions = []

        body = self._find_child_by_type(node, "declaration_list")
        if body:
            for child in body.children:
                if child.type == "class_specifier":
                    cls = self._extract_class(child, source_bytes)
                    if cls:
                        classes.append(cls)
                elif child.type == "function_definition":
                    func = self._extract_function(child, source_bytes)
                    if func:
                        functions.append(func)
                elif child.type == "struct_specifier":
                    cls = self._extract_struct(child, source_bytes)
                    if cls:
                        classes.append(cls)

        return classes, functions

    def _extract_template(self, node: Node, source_bytes: bytes):
        """Extract template class or function."""
        for child in node.children:
            if child.type == "class_specifier":
                return self._extract_class(child, source_bytes)
            elif child.type == "function_definition":
                return self._extract_function(child, source_bytes)
            elif child.type == "struct_specifier":
                return self._extract_struct(child, source_bytes)
        return None
