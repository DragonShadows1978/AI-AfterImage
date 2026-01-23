"""
Rust Parser - Tree-sitter based AST parser for Rust code.

Extracts:
- Function definitions (fn, async fn, const fn)
- Struct and enum definitions
- Impl blocks and trait implementations
- Use statements
- Doc comments (/// and //!)
"""

from typing import List, Optional

import tree_sitter_rust as tsrust
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


class RustParser(BaseParser):
    """Tree-sitter based parser for Rust."""

    @property
    def language_name(self) -> str:
        return "rust"

    def _get_language(self) -> Language:
        return Language(tsrust.language())

    def _extract_semantics(self, tree: Tree, source_bytes: bytes) -> SemanticInfo:
        """Extract semantic information from Rust AST."""
        functions = []
        classes = []
        imports = []
        module_doc = None

        root = tree.root_node

        # Check for module-level doc comments (//! comments at start)
        for child in root.children:
            if child.type == "line_comment":
                comment = self._get_node_text(child, source_bytes)
                if comment.startswith("//!"):
                    if module_doc is None:
                        module_doc = DocumentationInfo(
                            content=comment[3:].strip(),
                            format="rustdoc",
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                        )
                    else:
                        module_doc.content += "\n" + comment[3:].strip()
                        module_doc.end_line = child.end_point[0] + 1
                else:
                    break
            elif child.type not in ("line_comment", "block_comment"):
                break

        # Process items
        for child in root.children:
            if child.type == "function_item":
                func = self._extract_function(child, source_bytes)
                if func:
                    functions.append(func)

            elif child.type == "struct_item":
                struct = self._extract_struct(child, source_bytes)
                if struct:
                    classes.append(struct)

            elif child.type == "enum_item":
                enum = self._extract_enum(child, source_bytes)
                if enum:
                    classes.append(enum)

            elif child.type == "impl_item":
                impl_methods = self._extract_impl(child, source_bytes)
                functions.extend(impl_methods)

            elif child.type == "trait_item":
                trait = self._extract_trait(child, source_bytes)
                if trait:
                    classes.append(trait)

            elif child.type == "use_declaration":
                imp = self._extract_use(child, source_bytes)
                if imp:
                    imports.append(imp)

            elif child.type == "mod_item":
                # Module declaration
                pass  # Could extract sub-modules

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
        parent_type: str = None
    ) -> Optional[FunctionInfo]:
        """Extract function from function_item node."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Check modifiers
        is_async = any(child.type == "async" for child in node.children)
        is_const = any(child.type == "const" for child in node.children)
        is_unsafe = any(child.type == "unsafe" for child in node.children)

        # Get visibility
        visibility = self._extract_visibility(node, source_bytes)

        # Get parameters
        params_node = self._find_child_by_type(node, "parameters")
        parameters = self._extract_parameters(params_node, source_bytes) if params_node else []

        # Get return type
        return_type = None
        for i, child in enumerate(node.children):
            if child.type == "->":
                if i + 1 < len(node.children):
                    return_type = self._get_node_text(node.children[i + 1], source_bytes)
                break

        # Get doc comments
        documentation = self._find_doc_comment(node, source_bytes)

        # Decorators (attributes)
        decorators = self._extract_attributes(node, source_bytes)

        start_line, end_line, start_col, end_col = self._get_location(node)

        return FunctionInfo(
            name=name,
            parameters=parameters,
            return_type=return_type,
            is_async=is_async,
            is_method=parent_type is not None,
            visibility=visibility,
            decorators=decorators,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
            parent_class=parent_type,
        )

    def _extract_parameters(
        self,
        node: Node,
        source_bytes: bytes
    ) -> List[ParameterInfo]:
        """Extract parameters from parameters node."""
        params = []

        for child in node.children:
            if child.type == "parameter":
                # Regular parameter
                pattern = self._find_child_by_type(child, "identifier")
                type_node = self._find_child_by_type(child, "type_identifier")

                if pattern:
                    name = self._get_node_text(pattern, source_bytes)
                    type_ann = None

                    # Find type after colon
                    for i, c in enumerate(child.children):
                        if c.type == ":":
                            if i + 1 < len(child.children):
                                type_ann = self._get_node_text(child.children[i + 1], source_bytes)
                            break

                    params.append(ParameterInfo(
                        name=name,
                        type_annotation=type_ann,
                    ))

            elif child.type == "self_parameter":
                # self, &self, &mut self
                self_text = self._get_node_text(child, source_bytes)
                params.append(ParameterInfo(
                    name=self_text,
                ))

        return params

    def _extract_struct(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[ClassInfo]:
        """Extract struct from struct_item node."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)
        visibility = self._extract_visibility(node, source_bytes)

        # Get fields
        fields = []
        field_list = self._find_child_by_type(node, "field_declaration_list")
        if field_list:
            for child in field_list.children:
                if child.type == "field_declaration":
                    field_name_node = self._find_child_by_type(child, "field_identifier")
                    if field_name_node:
                        field_name = self._get_node_text(field_name_node, source_bytes)
                        field_type = None
                        for i, c in enumerate(child.children):
                            if c.type == ":":
                                if i + 1 < len(child.children):
                                    field_type = self._get_node_text(child.children[i + 1], source_bytes)
                                break
                        fields.append({
                            "name": field_name,
                            "type": field_type,
                            "visibility": self._extract_visibility(child, source_bytes).value,
                        })

        # Get type parameters
        type_params = self._extract_type_parameters(node, source_bytes)

        # Get doc comment
        documentation = self._find_doc_comment(node, source_bytes)

        # Get attributes
        decorators = self._extract_attributes(node, source_bytes)

        start_line, end_line, _, _ = self._get_location(node)

        return ClassInfo(
            name=name,
            kind="struct",
            fields=fields,
            visibility=visibility,
            decorators=decorators,
            documentation=documentation,
            type_parameters=type_params,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_enum(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[ClassInfo]:
        """Extract enum from enum_item node."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)
        visibility = self._extract_visibility(node, source_bytes)

        # Get variants as fields
        fields = []
        variant_list = self._find_child_by_type(node, "enum_variant_list")
        if variant_list:
            for child in variant_list.children:
                if child.type == "enum_variant":
                    variant_name_node = self._find_child_by_type(child, "identifier")
                    if variant_name_node:
                        variant_name = self._get_node_text(variant_name_node, source_bytes)
                        fields.append({
                            "name": variant_name,
                            "kind": "variant",
                        })

        documentation = self._find_doc_comment(node, source_bytes)
        decorators = self._extract_attributes(node, source_bytes)

        start_line, end_line, _, _ = self._get_location(node)

        return ClassInfo(
            name=name,
            kind="enum",
            fields=fields,
            visibility=visibility,
            decorators=decorators,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_trait(
        self,
        node: Node,
        source_bytes: bytes
    ) -> Optional[ClassInfo]:
        """Extract trait from trait_item node."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)
        visibility = self._extract_visibility(node, source_bytes)

        # Get trait methods
        methods = []
        body = self._find_child_by_type(node, "declaration_list")
        if body:
            for child in body.children:
                if child.type == "function_item" or child.type == "function_signature_item":
                    method = self._extract_function(child, source_bytes, parent_type=name)
                    if method:
                        methods.append(method)

        # Get super traits
        bases = []
        bounds = self._find_child_by_type(node, "trait_bounds")
        if bounds:
            for child in bounds.children:
                if child.type == "type_identifier":
                    bases.append(self._get_node_text(child, source_bytes))

        documentation = self._find_doc_comment(node, source_bytes)

        start_line, end_line, _, _ = self._get_location(node)

        return ClassInfo(
            name=name,
            kind="trait",
            bases=bases,
            methods=methods,
            visibility=visibility,
            documentation=documentation,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_impl(
        self,
        node: Node,
        source_bytes: bytes
    ) -> List[FunctionInfo]:
        """Extract methods from impl block."""
        methods = []

        # Get the type being implemented
        impl_type = None
        for child in node.children:
            if child.type == "type_identifier":
                impl_type = self._get_node_text(child, source_bytes)
                break
            elif child.type == "generic_type":
                type_id = self._find_child_by_type(child, "type_identifier")
                if type_id:
                    impl_type = self._get_node_text(type_id, source_bytes)
                break

        # Get methods
        body = self._find_child_by_type(node, "declaration_list")
        if body:
            for child in body.children:
                if child.type == "function_item":
                    method = self._extract_function(child, source_bytes, parent_type=impl_type)
                    if method:
                        methods.append(method)

        return methods

    def _extract_use(self, node: Node, source_bytes: bytes) -> Optional[ImportInfo]:
        """Extract use declaration."""
        # Get the use tree
        use_tree = None
        for child in node.children:
            if child.type in ("use_tree", "use_as_clause", "scoped_identifier", "identifier"):
                use_tree = child
                break

        if not use_tree:
            return None

        module_path = self._get_node_text(use_tree, source_bytes)

        # Parse the path to extract module and names
        # e.g., "std::collections::{HashMap, HashSet}" or "crate::module::function"
        names = []
        alias = None
        is_wildcard = "*" in module_path

        # Handle use groups
        if "{" in module_path:
            # Extract base path and names
            base_end = module_path.find("::{")
            if base_end > 0:
                module = module_path[:base_end]
                names_part = module_path[base_end + 3:-1]  # Remove ::{ and }
                names = [n.strip() for n in names_part.split(",")]
            else:
                module = module_path
        else:
            # Simple use
            module = module_path
            if "::" in module:
                parts = module.rsplit("::", 1)
                if len(parts) == 2 and not parts[1].startswith("*"):
                    names = [parts[1]]
                    module = parts[0]

        # Check for alias
        if " as " in module:
            parts = module.split(" as ")
            module = parts[0].strip()
            alias = parts[1].strip()

        start_line, end_line, _, _ = self._get_location(node)

        return ImportInfo(
            module=module,
            names=names,
            alias=alias,
            is_wildcard=is_wildcard,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_visibility(self, node: Node, source_bytes: bytes) -> Visibility:
        """Extract visibility modifier from a node."""
        for child in node.children:
            if child.type == "visibility_modifier":
                vis_text = self._get_node_text(child, source_bytes)
                if "pub" in vis_text:
                    return Visibility.PUBLIC
        return Visibility.PRIVATE

    def _extract_type_parameters(self, node: Node, source_bytes: bytes) -> List[str]:
        """Extract generic type parameters."""
        params = []
        type_params = self._find_child_by_type(node, "type_parameters")
        if type_params:
            for child in type_params.children:
                if child.type == "type_identifier":
                    params.append(self._get_node_text(child, source_bytes))
                elif child.type == "constrained_type_parameter":
                    param_name = self._find_child_by_type(child, "type_identifier")
                    if param_name:
                        params.append(self._get_node_text(param_name, source_bytes))
        return params

    def _extract_attributes(self, node: Node, source_bytes: bytes) -> List[str]:
        """Extract attributes (#[...]) from a node."""
        attrs = []
        for child in node.children:
            if child.type == "attribute_item":
                attr_text = self._get_node_text(child, source_bytes)
                # Remove #[ and ]
                if attr_text.startswith("#[") and attr_text.endswith("]"):
                    attrs.append(attr_text[2:-1])
        return attrs

    def _find_doc_comment(self, node: Node, source_bytes: bytes) -> Optional[DocumentationInfo]:
        """Find doc comment (/// or //!) preceding a node."""
        doc_lines = []
        current = node.prev_sibling

        while current:
            if current.type == "line_comment":
                comment = self._get_node_text(current, source_bytes)
                if comment.startswith("///"):
                    doc_lines.insert(0, comment[3:].strip())
                elif comment.startswith("//!"):
                    doc_lines.insert(0, comment[3:].strip())
                else:
                    break
            elif current.type == "attribute_item":
                # Skip attributes, they're before doc comments
                current = current.prev_sibling
                continue
            else:
                break
            current = current.prev_sibling

        if doc_lines:
            return DocumentationInfo(
                content="\n".join(doc_lines),
                format="rustdoc",
                start_line=node.start_point[0] + 1 - len(doc_lines),
                end_line=node.start_point[0],
            )

        return None
