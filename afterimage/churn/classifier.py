"""
Change classifier for analyzing code modifications.

Uses Python AST for Python files, regex fallback for other languages.
Determines whether changes are additions, modifications, or deletions.
"""

import ast
import re
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from difflib import SequenceMatcher

from .storage import ChangeType, ChangeResult, FunctionInfo


class ChangeClassifier:
    """
    Classifies code changes to determine the type and scope of modifications.

    Supports:
    - Python: Full AST analysis for accurate function/class detection
    - JavaScript/TypeScript: Regex-based detection
    - Other languages: Generic regex fallback
    """

    # Language-specific function patterns
    PATTERNS = {
        ".py": {
            "function": r"^\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)",
            "class": r"^\s*class\s+(\w+)\s*(?:\([^)]*\))?:",
            "method": r"^\s+(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)",
        },
        ".js": {
            "function": r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>|(\w+)\s*:\s*(?:async\s*)?function)",
            "class": r"class\s+(\w+)",
            "method": r"^\s+(?:async\s+)?(\w+)\s*\([^)]*\)\s*{",
        },
        ".ts": {
            "function": r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>|(\w+)\s*:\s*(?:async\s*)?function)",
            "class": r"class\s+(\w+)",
            "method": r"^\s+(?:async\s+)?(\w+)\s*\([^)]*\)\s*(?::\s*\w+)?\s*{",
        },
        ".go": {
            "function": r"func\s+(\w+)\s*\([^)]*\)",
            "class": r"type\s+(\w+)\s+struct",
            "method": r"func\s+\([^)]+\)\s+(\w+)\s*\(",
        },
        ".rs": {
            "function": r"fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(",
            "class": r"(?:struct|enum|trait)\s+(\w+)",
            "method": r"^\s+(?:pub\s+)?fn\s+(\w+)",
        },
        ".java": {
            "function": r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+)?\s*{",
            "class": r"(?:public|private)?\s*class\s+(\w+)",
            "method": r"^\s+(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)",
        },
    }

    def __init__(self):
        self._cache: Dict[str, List[FunctionInfo]] = {}

    def classify_change(
        self,
        file_path: str,
        old_code: Optional[str],
        new_code: str
    ) -> ChangeResult:
        """
        Classify what kind of change this is.

        Compares old and new code to determine:
        - Which functions/classes were added
        - Which were modified
        - Which were deleted

        Args:
            file_path: Path to the file being modified
            old_code: Previous content (None for new files)
            new_code: New content being written

        Returns:
            ChangeResult with change type and function-level details
        """
        # New file - everything is an addition
        if old_code is None:
            new_functions = self.extract_functions(new_code, file_path)
            return ChangeResult(
                change_type=ChangeType.ADD,
                functions_added=new_functions,
            )

        # Extract functions from both versions
        old_functions = self.extract_functions(old_code, file_path)
        new_functions = self.extract_functions(new_code, file_path)

        # Build signature hash maps for comparison
        old_by_hash = {f.signature_hash(): f for f in old_functions}
        new_by_hash = {f.signature_hash(): f for f in new_functions}

        # Build name maps for fuzzy matching
        old_by_name = {f.name: f for f in old_functions}
        new_by_name = {f.name: f for f in new_functions}

        functions_added = []
        functions_modified = []
        functions_deleted = []

        # Find added and modified functions
        for sig_hash, func in new_by_hash.items():
            if sig_hash in old_by_hash:
                # Signature exists in old - check if body changed
                old_func = old_by_hash[sig_hash]
                if self._function_body_changed(old_code, new_code, old_func, func):
                    functions_modified.append(func)
            elif func.name in old_by_name:
                # Name exists but signature changed - modified
                functions_modified.append(func)
            else:
                # Truly new function
                functions_added.append(func)

        # Find deleted functions
        for sig_hash, func in old_by_hash.items():
            if sig_hash not in new_by_hash and func.name not in new_by_name:
                functions_deleted.append(func)

        # Determine overall change type
        if len(functions_deleted) > len(functions_added) + len(functions_modified):
            change_type = ChangeType.DELETE
        elif len(functions_added) > 0 and len(functions_modified) == 0:
            change_type = ChangeType.ADD
        elif len(functions_modified) > 0:
            change_type = ChangeType.MODIFY
        elif self._is_refactor(old_code, new_code):
            change_type = ChangeType.REFACTOR
        else:
            change_type = ChangeType.MODIFY

        return ChangeResult(
            change_type=change_type,
            functions_added=functions_added,
            functions_modified=functions_modified,
            functions_deleted=functions_deleted,
        )

    def extract_functions(self, code: str, file_path: str) -> List[FunctionInfo]:
        """
        Extract function/class/method definitions from code.

        Uses AST for Python, regex for other languages.

        Args:
            code: Source code
            file_path: Path (used to determine language)

        Returns:
            List of FunctionInfo objects
        """
        ext = Path(file_path).suffix.lower()

        if ext == ".py":
            return self._extract_python_ast(code, file_path)
        else:
            return self._extract_regex(code, file_path, ext)

    def _extract_python_ast(self, code: str, file_path: str) -> List[FunctionInfo]:
        """Extract Python functions/classes using AST."""
        functions = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fall back to regex if AST fails
            return self._extract_regex(code, file_path, ".py")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Get signature
                sig = self._get_python_signature(node)
                kind = "method" if self._is_method(node, tree) else "function"

                functions.append(FunctionInfo(
                    name=node.name,
                    file_path=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    kind=kind,
                    signature=sig,
                ))

            elif isinstance(node, ast.ClassDef):
                sig = f"class {node.name}"
                if node.bases:
                    bases = [self._get_name(b) for b in node.bases]
                    sig += f"({', '.join(bases)})"

                functions.append(FunctionInfo(
                    name=node.name,
                    file_path=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    kind="class",
                    signature=sig,
                ))

        return functions

    def _get_python_signature(self, node: ast.FunctionDef) -> str:
        """Build signature string from Python function AST node."""
        args = []

        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_name(arg.annotation)}"
            args.append(arg_str)

        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        sig = f"{prefix}def {node.name}({', '.join(args)})"

        if node.returns:
            sig += f" -> {self._get_name(node.returns)}"

        return sig

    def _get_name(self, node) -> str:
        """Get name from AST node (handles Name, Attribute, etc.)."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_name(node.slice)}]"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return "..."

    def _is_method(self, node: ast.FunctionDef, tree: ast.Module) -> bool:
        """Check if function is a method (inside a class)."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                for child in ast.iter_child_nodes(parent):
                    if child is node:
                        return True
        return False

    def _extract_regex(self, code: str, file_path: str, ext: str) -> List[FunctionInfo]:
        """Extract functions using regex patterns."""
        functions = []
        patterns = self.PATTERNS.get(ext, self.PATTERNS.get(".py"))

        lines = code.split("\n")

        for i, line in enumerate(lines):
            # Check function pattern
            if patterns.get("function"):
                match = re.search(patterns["function"], line)
                if match:
                    name = next(g for g in match.groups() if g)
                    functions.append(FunctionInfo(
                        name=name,
                        file_path=file_path,
                        line_start=i + 1,
                        line_end=i + 1,  # Can't determine end with regex
                        kind="function",
                        signature=line.strip(),
                    ))
                    continue

            # Check class pattern
            if patterns.get("class"):
                match = re.search(patterns["class"], line)
                if match:
                    name = match.group(1)
                    functions.append(FunctionInfo(
                        name=name,
                        file_path=file_path,
                        line_start=i + 1,
                        line_end=i + 1,
                        kind="class",
                        signature=line.strip(),
                    ))
                    continue

            # Check method pattern
            if patterns.get("method"):
                match = re.search(patterns["method"], line)
                if match:
                    name = next(g for g in match.groups() if g)
                    functions.append(FunctionInfo(
                        name=name,
                        file_path=file_path,
                        line_start=i + 1,
                        line_end=i + 1,
                        kind="method",
                        signature=line.strip(),
                    ))

        return functions

    def _function_body_changed(
        self,
        old_code: str,
        new_code: str,
        old_func: FunctionInfo,
        new_func: FunctionInfo
    ) -> bool:
        """Check if a function's body has changed."""
        old_lines = old_code.split("\n")
        new_lines = new_code.split("\n")

        # Extract function bodies
        old_body = "\n".join(old_lines[old_func.line_start - 1:old_func.line_end])
        new_body = "\n".join(new_lines[new_func.line_start - 1:new_func.line_end])

        # Normalize and compare
        old_normalized = self._normalize_code(old_body)
        new_normalized = self._normalize_code(new_body)

        return old_normalized != new_normalized

    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison (remove comments, normalize whitespace)."""
        # Remove single-line comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)

        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

        # Normalize whitespace
        code = " ".join(code.split())

        return code

    def _is_refactor(self, old_code: str, new_code: str) -> bool:
        """Check if change is primarily a refactoring (structure change, logic preserved)."""
        # Compare normalized code similarity
        old_normalized = self._normalize_code(old_code)
        new_normalized = self._normalize_code(new_code)

        # High similarity suggests refactoring
        ratio = SequenceMatcher(None, old_normalized, new_normalized).ratio()
        return ratio > 0.8

    def hash_signature(self, signature: str) -> str:
        """Create stable hash of function signature."""
        # Normalize: lowercase, strip, collapse whitespace
        normalized = " ".join(signature.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get_modified_function_names(
        self,
        old_code: Optional[str],
        new_code: str,
        file_path: str
    ) -> Set[str]:
        """Get set of function names that were modified."""
        result = self.classify_change(file_path, old_code, new_code)
        return {f.name for f in result.functions_modified}

    def is_purely_additive(
        self,
        old_code: Optional[str],
        new_code: str,
        file_path: str
    ) -> bool:
        """Check if change only adds new code without modifying existing."""
        result = self.classify_change(file_path, old_code, new_code)
        return result.is_purely_additive
