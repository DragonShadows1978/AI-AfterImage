"""
Semantic Code Chunker: Parse code into meaningful semantic units.

Instead of returning raw file contents, this module breaks code into
functions, classes, and logical blocks that can be independently
retrieved and scored for relevance.

Part of AfterImage Semantic Chunking v0.3.0.
"""

import re
import ast
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ChunkType(Enum):
    """Types of semantic code chunks."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    IMPORT_BLOCK = "imports"
    CONSTANT_BLOCK = "constants"
    CODE_BLOCK = "block"
    DOCSTRING = "docstring"
    COMMENT_BLOCK = "comments"
    UNKNOWN = "unknown"


@dataclass
class CodeChunk:
    """
    A semantic unit of code.

    Represents a meaningful piece of code (function, class, block)
    that can be independently retrieved and scored.
    """
    chunk_type: ChunkType
    name: str
    code: str
    start_line: int
    end_line: int
    parent_name: Optional[str] = None  # For methods in classes
    signature: Optional[str] = None  # Function/method signature
    docstring: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # Referenced symbols
    token_count: int = 0  # Estimated tokens

    def __hash__(self):
        return hash((self.chunk_type.value, self.name, self.start_line))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_type": self.chunk_type.value,
            "name": self.name,
            "code": self.code,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parent_name": self.parent_name,
            "signature": self.signature,
            "docstring": self.docstring,
            "dependencies": self.dependencies,
            "token_count": self.token_count,
        }


class SemanticChunker:
    """
    Parses source code into semantic chunks.

    Uses AST parsing for Python files and regex-based parsing for other
    languages. Returns meaningful code units rather than raw file contents.
    """

    def __init__(self, max_chunk_tokens: int = 500):
        """
        Initialize the chunker.

        Args:
            max_chunk_tokens: Maximum tokens per chunk (large items get split)
        """
        self.max_chunk_tokens = max_chunk_tokens

    def chunk_code(
        self,
        code: str,
        file_path: Optional[str] = None
    ) -> List[CodeChunk]:
        """
        Parse code into semantic chunks.

        Args:
            code: Source code to parse
            file_path: Optional file path for language detection

        Returns:
            List of CodeChunk objects representing semantic units
        """
        if not code or not code.strip():
            return []

        # Detect language from file path
        language = self._detect_language(file_path) if file_path else None

        # Use appropriate parser
        if language == "python":
            chunks = self._chunk_python(code)
        elif language in ("javascript", "typescript", "jsx", "tsx"):
            chunks = self._chunk_javascript(code)
        elif language in ("rust", "go", "c", "cpp"):
            chunks = self._chunk_c_like(code)
        else:
            chunks = self._chunk_generic(code)

        # Estimate tokens for each chunk
        for chunk in chunks:
            chunk.token_count = self._estimate_tokens(chunk.code)

        # Split oversized chunks
        final_chunks = []
        for chunk in chunks:
            if chunk.token_count > self.max_chunk_tokens:
                final_chunks.extend(self._split_chunk(chunk))
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "jsx",
            ".tsx": "tsx",
            ".rs": "rust",
            ".go": "go",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".java": "java",
            ".rb": "ruby",
            ".php": "php",
        }
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext)

    def _chunk_python(self, code: str) -> List[CodeChunk]:
        """Parse Python code using AST."""
        chunks = []
        lines = code.split("\n")

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fall back to generic parsing if AST fails
            return self._chunk_generic(code)

        # Track import blocks
        import_lines = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if hasattr(node, 'lineno'):
                    import_lines.append(node.lineno)

            elif isinstance(node, ast.FunctionDef):
                chunk = self._extract_python_function(node, lines, None)
                if chunk:
                    chunks.append(chunk)

            elif isinstance(node, ast.AsyncFunctionDef):
                chunk = self._extract_python_function(node, lines, None, is_async=True)
                if chunk:
                    chunks.append(chunk)

            elif isinstance(node, ast.ClassDef):
                # Extract the class and its methods
                class_chunks = self._extract_python_class(node, lines)
                chunks.extend(class_chunks)

        # Create import block if any
        if import_lines:
            import_start = min(import_lines)
            import_end = max(import_lines)
            import_code = "\n".join(lines[import_start-1:import_end])
            chunks.insert(0, CodeChunk(
                chunk_type=ChunkType.IMPORT_BLOCK,
                name="imports",
                code=import_code,
                start_line=import_start,
                end_line=import_end
            ))

        # Handle module-level constants and other code
        chunks.extend(self._extract_python_module_level(tree, lines, chunks))

        return chunks

    def _extract_python_function(
        self,
        node: ast.FunctionDef,
        lines: List[str],
        parent_class: Optional[str],
        is_async: bool = False
    ) -> Optional[CodeChunk]:
        """Extract a Python function/method as a chunk."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Get function code
        func_code = "\n".join(lines[start_line-1:end_line])

        # Build signature
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"

        prefix = "async def" if is_async else "def"
        signature = f"{prefix} {node.name}({', '.join(args)}){returns}"

        # Get docstring
        docstring = ast.get_docstring(node)

        # Determine chunk type
        chunk_type = ChunkType.METHOD if parent_class else ChunkType.FUNCTION

        return CodeChunk(
            chunk_type=chunk_type,
            name=node.name,
            code=func_code,
            start_line=start_line,
            end_line=end_line,
            parent_name=parent_class,
            signature=signature,
            docstring=docstring,
            dependencies=self._extract_python_dependencies(node)
        )

    def _extract_python_class(
        self,
        node: ast.ClassDef,
        lines: List[str]
    ) -> List[CodeChunk]:
        """Extract a Python class and its methods as chunks."""
        chunks = []

        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # First, create a chunk for the class skeleton
        class_header_lines = []
        class_header_lines.append(lines[start_line - 1])  # class definition line

        # Add docstring if present
        docstring = ast.get_docstring(node)
        if docstring:
            # Find docstring end
            for i, stmt in enumerate(node.body):
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    docstring_end = stmt.end_lineno
                    class_header_lines.extend(lines[start_line:docstring_end])
                    break

        # Create class overview chunk
        class_code = "\n".join(lines[start_line-1:end_line])
        chunks.append(CodeChunk(
            chunk_type=ChunkType.CLASS,
            name=node.name,
            code=class_code,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            dependencies=self._extract_python_dependencies(node)
        ))

        # Extract individual methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                chunk = self._extract_python_function(item, lines, node.name)
                if chunk:
                    chunks.append(chunk)
            elif isinstance(item, ast.AsyncFunctionDef):
                chunk = self._extract_python_function(item, lines, node.name, is_async=True)
                if chunk:
                    chunks.append(chunk)

        return chunks

    def _extract_python_module_level(
        self,
        tree: ast.Module,
        lines: List[str],
        existing_chunks: List[CodeChunk]
    ) -> List[CodeChunk]:
        """Extract module-level constants and assignments."""
        chunks = []
        covered_lines = set()

        for chunk in existing_chunks:
            covered_lines.update(range(chunk.start_line, chunk.end_line + 1))

        constant_block = []
        constant_start = None

        for node in tree.body:
            # Skip already covered nodes
            if hasattr(node, 'lineno') and node.lineno in covered_lines:
                continue

            # Module-level assignments (constants)
            if isinstance(node, ast.Assign):
                if constant_start is None:
                    constant_start = node.lineno
                constant_block.append(lines[node.lineno - 1])
            elif isinstance(node, ast.AnnAssign):
                if constant_start is None:
                    constant_start = node.lineno
                line_end = node.end_lineno or node.lineno
                constant_block.append("\n".join(lines[node.lineno-1:line_end]))

        if constant_block and constant_start:
            chunks.append(CodeChunk(
                chunk_type=ChunkType.CONSTANT_BLOCK,
                name="constants",
                code="\n".join(constant_block),
                start_line=constant_start,
                end_line=constant_start + len(constant_block) - 1
            ))

        return chunks

    def _extract_python_dependencies(self, node: ast.AST) -> List[str]:
        """Extract symbols referenced by a Python AST node."""
        deps = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                deps.add(child.id)
            elif isinstance(child, ast.Attribute):
                # Get the root name of attribute chains
                current = child
                while isinstance(current, ast.Attribute):
                    current = current.value
                if isinstance(current, ast.Name):
                    deps.add(current.id)
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    deps.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    deps.add(child.func.attr)

        # Filter out Python builtins
        builtins = {'print', 'len', 'range', 'str', 'int', 'float', 'list',
                    'dict', 'set', 'tuple', 'bool', 'None', 'True', 'False',
                    'self', 'cls', 'super', 'type', 'isinstance', 'hasattr',
                    'getattr', 'setattr', 'delattr', 'open', 'file', 'input',
                    'map', 'filter', 'reduce', 'zip', 'enumerate', 'sorted',
                    'min', 'max', 'sum', 'abs', 'round', 'any', 'all'}

        return list(deps - builtins)

    def _chunk_javascript(self, code: str) -> List[CodeChunk]:
        """Parse JavaScript/TypeScript code using regex patterns."""
        chunks = []
        lines = code.split("\n")

        # Function patterns
        func_pattern = re.compile(
            r'^(\s*)(export\s+)?(async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{',
            re.MULTILINE
        )

        # Arrow function pattern
        arrow_pattern = re.compile(
            r'^(\s*)(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*\{',
            re.MULTILINE
        )

        # Class pattern
        class_pattern = re.compile(
            r'^(\s*)(export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{',
            re.MULTILINE
        )

        # Find functions
        for match in func_pattern.finditer(code):
            start_pos = match.start()
            start_line = code[:start_pos].count('\n') + 1
            func_name = match.group(4)

            # Find matching closing brace
            end_line = self._find_closing_brace(lines, start_line - 1)
            func_code = "\n".join(lines[start_line-1:end_line])

            chunks.append(CodeChunk(
                chunk_type=ChunkType.FUNCTION,
                name=func_name,
                code=func_code,
                start_line=start_line,
                end_line=end_line,
                signature=match.group(0).strip().rstrip('{').strip()
            ))

        # Find arrow functions
        for match in arrow_pattern.finditer(code):
            start_pos = match.start()
            start_line = code[:start_pos].count('\n') + 1
            func_name = match.group(4)

            end_line = self._find_closing_brace(lines, start_line - 1)
            func_code = "\n".join(lines[start_line-1:end_line])

            chunks.append(CodeChunk(
                chunk_type=ChunkType.FUNCTION,
                name=func_name,
                code=func_code,
                start_line=start_line,
                end_line=end_line
            ))

        # Find classes
        for match in class_pattern.finditer(code):
            start_pos = match.start()
            start_line = code[:start_pos].count('\n') + 1
            class_name = match.group(3)

            end_line = self._find_closing_brace(lines, start_line - 1)
            class_code = "\n".join(lines[start_line-1:end_line])

            chunks.append(CodeChunk(
                chunk_type=ChunkType.CLASS,
                name=class_name,
                code=class_code,
                start_line=start_line,
                end_line=end_line
            ))

        return chunks

    def _chunk_c_like(self, code: str) -> List[CodeChunk]:
        """Parse C/C++/Rust/Go code using regex patterns."""
        chunks = []
        lines = code.split("\n")

        # Function pattern for C-like languages
        func_pattern = re.compile(
            r'^(\s*)(?:pub\s+)?(?:async\s+)?(?:fn|func|void|int|char|float|double|bool|auto|\w+(?:\s*\*)?)\s+(\w+)\s*\([^)]*\)\s*(?:->[\s\w<>]+)?\s*\{',
            re.MULTILINE
        )

        # Struct/impl pattern
        struct_pattern = re.compile(
            r'^(\s*)(?:pub\s+)?(?:struct|impl|type)\s+(\w+)(?:<[^>]+>)?(?:\s+\w+)?\s*\{',
            re.MULTILINE
        )

        for match in func_pattern.finditer(code):
            start_pos = match.start()
            start_line = code[:start_pos].count('\n') + 1
            func_name = match.group(2)

            end_line = self._find_closing_brace(lines, start_line - 1)
            func_code = "\n".join(lines[start_line-1:end_line])

            chunks.append(CodeChunk(
                chunk_type=ChunkType.FUNCTION,
                name=func_name,
                code=func_code,
                start_line=start_line,
                end_line=end_line
            ))

        for match in struct_pattern.finditer(code):
            start_pos = match.start()
            start_line = code[:start_pos].count('\n') + 1
            struct_name = match.group(2)

            end_line = self._find_closing_brace(lines, start_line - 1)
            struct_code = "\n".join(lines[start_line-1:end_line])

            chunks.append(CodeChunk(
                chunk_type=ChunkType.CLASS,
                name=struct_name,
                code=struct_code,
                start_line=start_line,
                end_line=end_line
            ))

        return chunks

    def _chunk_generic(self, code: str) -> List[CodeChunk]:
        """
        Generic chunking for unknown languages.

        Uses simple heuristics like blank lines and indentation to
        identify logical code blocks.
        """
        chunks = []
        lines = code.split("\n")

        current_block = []
        block_start = 1

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Empty line might indicate block boundary
            if not stripped:
                if current_block:
                    block_code = "\n".join(current_block)
                    chunks.append(CodeChunk(
                        chunk_type=ChunkType.CODE_BLOCK,
                        name=f"block_{len(chunks) + 1}",
                        code=block_code,
                        start_line=block_start,
                        end_line=i - 1
                    ))
                    current_block = []
                    block_start = i + 1
            else:
                current_block.append(line)

        # Don't forget the last block
        if current_block:
            block_code = "\n".join(current_block)
            chunks.append(CodeChunk(
                chunk_type=ChunkType.CODE_BLOCK,
                name=f"block_{len(chunks) + 1}",
                code=block_code,
                start_line=block_start,
                end_line=len(lines)
            ))

        return chunks

    def _find_closing_brace(self, lines: List[str], start_idx: int) -> int:
        """Find the line number of the closing brace for a block."""
        brace_count = 0
        in_string = False
        string_char = None

        for i in range(start_idx, len(lines)):
            line = lines[i]
            j = 0
            while j < len(line):
                char = line[j]

                # Handle string literals
                if char in ('"', "'", '`') and (j == 0 or line[j-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None

                # Count braces outside strings
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            return i + 1

                j += 1

        # If we couldn't find closing brace, return end of file
        return len(lines)

    def _split_chunk(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Split an oversized chunk into smaller pieces."""
        lines = chunk.code.split("\n")
        chunks = []

        # Calculate lines per chunk to stay under token limit
        tokens_per_line = chunk.token_count / max(len(lines), 1)
        lines_per_chunk = int(self.max_chunk_tokens / max(tokens_per_line, 1))
        lines_per_chunk = max(lines_per_chunk, 5)  # Minimum 5 lines per chunk

        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunk_code = "\n".join(chunk_lines)

            part_num = i // lines_per_chunk + 1
            total_parts = (len(lines) + lines_per_chunk - 1) // lines_per_chunk

            chunks.append(CodeChunk(
                chunk_type=chunk.chunk_type,
                name=f"{chunk.name}_part{part_num}of{total_parts}",
                code=chunk_code,
                start_line=chunk.start_line + i,
                end_line=chunk.start_line + i + len(chunk_lines) - 1,
                parent_name=chunk.parent_name,
                token_count=self._estimate_tokens(chunk_code)
            ))

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses approximation: ~4 characters per token for code.
        """
        return len(text) // 4 + 1


def chunk_code_file(file_path: str, max_chunk_tokens: int = 500) -> List[CodeChunk]:
    """
    Convenience function to chunk a code file.

    Args:
        file_path: Path to the code file
        max_chunk_tokens: Maximum tokens per chunk

    Returns:
        List of CodeChunk objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    chunker = SemanticChunker(max_chunk_tokens=max_chunk_tokens)
    return chunker.chunk_code(code, file_path)
