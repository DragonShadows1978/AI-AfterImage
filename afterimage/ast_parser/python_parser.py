"""
Python Parser - Tree-sitter based AST parser for Python code.

Extracts:
- Function definitions (def, async def)
- Class definitions with methods and inheritance
- Import statements (import, from...import)
- Docstrings (module, class, function level)
- Decorators
"""

from typing import List, Optional

import tree_sitter_python as tspython
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


class PythonParser(BaseParser):
    """Tree-sitter based parser for Python."""

    @property
    def language_name(self) -> str:
        return "python"

    def _get_language(self) -> Language:
        return Language(tspython.language())

    def _extract_semantics(self, tree: Tree, source_bytes: bytes) -> SemanticInfo:
        """Extract semantic information from Python AST."""
        functions = []
        classes = []
        imports = []
        module_doc = None

        root = tree.root_node

        # Check for module docstring (first string in module)
        for child in root.children:
            if child.type == "expression_statement":
                string_node = self._find_child_by_type(child, "string")
                if string_node:
                    module_doc = self._extract_docstring(string_node, source_bytes)
                break
            elif child.type not in ("comment",):
                break

        # Process top-level nodes
        for child in root.children:
            if child.type == "function_definition":
                func = self._extract_function(child, source_bytes)
                if func:
                    functions.append(func)

            elif child.type == "class_definition":
                cls = self._extract_class(child, source_bytes)
                if cls:
                    classes.append(cls)

            elif child.type == "import_statement":
                imp = self._extract_import(child, source_bytes)
                if imp:
                    imports.append(imp)

            elif child.type == "import_from_statement":
                imp = self._extract_import_from(child, source_bytes)
                if imp:
                    imports.append(imp)

            elif child.type == "decorated_definition":
                # Handle decorated functions/classes
                decorated = self._extract_decorated(child, source_bytes)
                if isinstance(decorated, FunctionInfo):
                    functions.append(decorated)
                elif isinstance(decorated, ClassInfo):
                    classes.append(decorated)

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
        decorators: List[str] = None,
        parent_class: str = None
    ) -> Optional[FunctionInfo]:
        """Extract function information from a function_definition node."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Check if async
        is_async = any(
            child.type == "async" for child in node.children
        )

        # Get parameters
        params_node = self._find_child_by_type(node, "parameters")
        parameters = self._extract_parameters(params_node, source_bytes) if params_node else []

        # Get return type annotation
        return_type = None
        for child in node.children:
            if child.type == "type":
                return_type = self._get_node_text(child, source_bytes)
                break

        # Get docstring
        documentation = None
        body_node = self._find_child_by_type(node, "block")
        if body_node and body_node.children:
            first_stmt = body_node.children[0]
            if first_stmt.type == "expression_statement":
                string_node = self._find_child_by_type(first_stmt, "string")
                if string_node:
                    documentation = self._extract_docstring(string_node, source_bytes)

        # Determine method properties
        is_method = parent_class is not None
        is_static = decorators and "staticmethod" in decorators
        is_class_method = decorators and "classmethod" in decorators

        # Check if generator
        is_generator = self._contains_yield(body_node) if body_node else False

        # Determine visibility
        visibility = Visibility.PRIVATE if name.startswith("_") else Visibility.PUBLIC
        if name.startswith("__") and not name.endswith("__"):
            visibility = Visibility.PRIVATE

        start_line, end_line, start_col, end_col = self._get_location(node)

        return FunctionInfo(
            name=name,
            parameters=parameters,
            return_type=return_type,
            is_async=is_async,
            is_generator=is_generator,
            is_method=is_method,
            is_static=is_static,
            is_class_method=is_class_method,
            visibility=visibility,
            decorators=decorators or [],
            documentation=documentation,
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
        """Extract parameters from a parameters node."""
        params = []

        for child in node.children:
            if child.type == "identifier":
                # Simple parameter
                params.append(ParameterInfo(
                    name=self._get_node_text(child, source_bytes)
                ))

            elif child.type == "typed_parameter":
                name_node = self._find_child_by_type(child, "identifier")
                type_node = self._find_child_by_type(child, "type")

                if name_node:
                    params.append(ParameterInfo(
                        name=self._get_node_text(name_node, source_bytes),
                        type_annotation=self._get_node_text(type_node, source_bytes) if type_node else None,
                    ))

            elif child.type == "default_parameter":
                name_node = self._find_child_by_type(child, "identifier")
                if name_node:
                    # Find default value (everything after '=')
                    default_val = None
                    for i, c in enumerate(child.children):
                        if c.type == "=" and i + 1 < len(child.children):
                            default_val = self._get_node_text(child.children[i + 1], source_bytes)
                            break

                    params.append(ParameterInfo(
                        name=self._get_node_text(name_node, source_bytes),
                        default_value=default_val,
                    ))

            elif child.type == "typed_default_parameter":
                name_node = self._find_child_by_type(child, "identifier")
                type_node = self._find_child_by_type(child, "type")

                if name_node:
                    # Find default value
                    default_val = None
                    for i, c in enumerate(child.children):
                        if c.type == "=" and i + 1 < len(child.children):
                            default_val = self._get_node_text(child.children[i + 1], source_bytes)
                            break

                    params.append(ParameterInfo(
                        name=self._get_node_text(name_node, source_bytes),
                        type_annotation=self._get_node_text(type_node, source_bytes) if type_node else None,
                        default_value=default_val,
                    ))

            elif child.type == "list_splat_pattern":
                # *args
                name_node = self._find_child_by_type(child, "identifier")
                if name_node:
                    params.append(ParameterInfo(
                        name=self._get_node_text(name_node, source_bytes),
                        is_variadic=True,
                    ))

            elif child.type == "dictionary_splat_pattern":
                # **kwargs
                name_node = self._find_child_by_type(child, "identifier")
                if name_node:
                    params.append(ParameterInfo(
                        name=self._get_node_text(name_node, source_bytes),
                        is_keyword_variadic=True,
                    ))

        return params

    def _extract_class(
        self,
        node: Node,
        source_bytes: bytes,
        decorators: List[str] = None
    ) -> Optional[ClassInfo]:
        """Extract class information from a class_definition node."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None

        name = self._get_node_text(name_node, source_bytes)

        # Get base classes
        bases = []
        arg_list = self._find_child_by_type(node, "argument_list")
        if arg_list:
            for child in arg_list.children:
                if child.type == "identifier":
                    bases.append(self._get_node_text(child, source_bytes))
                elif child.type == "attribute":
                    bases.append(self._get_node_text(child, source_bytes))

        # Get methods and class docstring
        methods = []
        documentation = None
        body_node = self._find_child_by_type(node, "block")

        if body_node:
            first_checked = False
            for child in body_node.children:
                # Check for class docstring
                if not first_checked and child.type == "expression_statement":
                    string_node = self._find_child_by_type(child, "string")
                    if string_node:
                        documentation = self._extract_docstring(string_node, source_bytes)
                    first_checked = True
                elif child.type not in ("comment",):
                    first_checked = True

                if child.type == "function_definition":
                    method = self._extract_function(child, source_bytes, parent_class=name)
                    if method:
                        methods.append(method)

                elif child.type == "decorated_definition":
                    # Get decorators
                    method_decorators = []
                    for deco_child in child.children:
                        if deco_child.type == "decorator":
                            deco_text = self._get_node_text(deco_child, source_bytes)
                            # Remove @ prefix
                            method_decorators.append(deco_text.lstrip("@").strip())

                    # Get the function
                    func_node = self._find_child_by_type(child, "function_definition")
                    if func_node:
                        method = self._extract_function(
                            func_node, source_bytes,
                            decorators=method_decorators,
                            parent_class=name
                        )
                        if method:
                            methods.append(method)

        # Check for abstract class
        is_abstract = decorators and any("abstractmethod" in d or "ABC" in d for d in decorators)

        start_line, end_line, _, _ = self._get_location(node)

        return ClassInfo(
            name=name,
            kind="class",
            bases=bases,
            methods=methods,
            visibility=Visibility.PRIVATE if name.startswith("_") else Visibility.PUBLIC,
            decorators=decorators or [],
            documentation=documentation,
            is_abstract=is_abstract,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_decorated(
        self,
        node: Node,
        source_bytes: bytes
    ):
        """Extract decorated function or class."""
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                deco_text = self._get_node_text(child, source_bytes)
                decorators.append(deco_text.lstrip("@").strip())

        # Find the actual definition
        func_node = self._find_child_by_type(node, "function_definition")
        if func_node:
            return self._extract_function(func_node, source_bytes, decorators=decorators)

        class_node = self._find_child_by_type(node, "class_definition")
        if class_node:
            return self._extract_class(class_node, source_bytes, decorators=decorators)

        return None

    def _extract_import(self, node: Node, source_bytes: bytes) -> Optional[ImportInfo]:
        """Extract import statement (import x, import x as y)."""
        # Find dotted_name or aliased_import
        for child in node.children:
            if child.type == "dotted_name":
                module = self._get_node_text(child, source_bytes)
                start_line, end_line, _, _ = self._get_location(node)
                return ImportInfo(
                    module=module,
                    start_line=start_line,
                    end_line=end_line,
                )

            elif child.type == "aliased_import":
                name_node = self._find_child_by_type(child, "dotted_name")
                alias_node = self._find_child_by_type(child, "identifier")
                if name_node:
                    module = self._get_node_text(name_node, source_bytes)
                    alias = self._get_node_text(alias_node, source_bytes) if alias_node else None
                    start_line, end_line, _, _ = self._get_location(node)
                    return ImportInfo(
                        module=module,
                        alias=alias,
                        start_line=start_line,
                        end_line=end_line,
                    )

        return None

    def _extract_import_from(self, node: Node, source_bytes: bytes) -> Optional[ImportInfo]:
        """Extract from...import statement."""
        module = ""
        names = []
        is_wildcard = False

        for child in node.children:
            if child.type == "dotted_name":
                module = self._get_node_text(child, source_bytes)

            elif child.type == "relative_import":
                module = self._get_node_text(child, source_bytes)

            elif child.type == "wildcard_import":
                is_wildcard = True

            elif child.type == "import_list":
                for import_child in child.children:
                    if import_child.type == "dotted_name":
                        names.append(self._get_node_text(import_child, source_bytes))
                    elif import_child.type == "aliased_import":
                        name_node = self._find_child_by_type(import_child, "dotted_name")
                        if name_node:
                            names.append(self._get_node_text(name_node, source_bytes))

        start_line, end_line, _, _ = self._get_location(node)
        return ImportInfo(
            module=module,
            names=names,
            is_wildcard=is_wildcard,
            start_line=start_line,
            end_line=end_line,
        )

    def _extract_docstring(self, node: Node, source_bytes: bytes) -> DocumentationInfo:
        """Extract docstring content."""
        text = self._get_node_text(node, source_bytes)

        # Remove quotes
        if text.startswith('"""') or text.startswith("'''"):
            text = text[3:-3]
        elif text.startswith('"') or text.startswith("'"):
            text = text[1:-1]

        start_line, end_line, _, _ = self._get_location(node)

        return DocumentationInfo(
            content=text.strip(),
            format="docstring",
            start_line=start_line,
            end_line=end_line,
        )

    def _contains_yield(self, node: Node) -> bool:
        """Check if a block contains yield statements."""
        if node is None:
            return False

        if node.type in ("yield", "yield_statement"):
            return True

        for child in node.children:
            if self._contains_yield(child):
                return True

        return False
