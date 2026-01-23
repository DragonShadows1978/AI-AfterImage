"""
Base Parser - Abstract base class for tree-sitter based language parsers.

Provides common functionality for:
- Parser initialization and caching
- Node traversal and counting
- Error detection and confidence scoring
- Incremental parsing state management with proper tree.edit() support
"""

import difflib
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from tree_sitter import Language, Parser, Node, Tree

from .models import (
    ASTResult,
    ErrorInfo,
    SemanticInfo,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    DocumentationInfo,
)


@dataclass
class EditDescriptor:
    """Describes a single edit operation for incremental parsing."""
    start_byte: int
    old_end_byte: int
    new_end_byte: int
    start_point: Tuple[int, int]  # (row, col)
    old_end_point: Tuple[int, int]  # (row, col)
    new_end_point: Tuple[int, int]  # (row, col)


@dataclass
class ParseState:
    """State for incremental parsing."""
    tree: Tree
    source_bytes: bytes
    source_hash: str
    timestamp: float
    file_path: Optional[str] = None


def _byte_offset_to_point(source_bytes: bytes, byte_offset: int) -> Tuple[int, int]:
    """Convert byte offset to (row, col) point."""
    text_before = source_bytes[:byte_offset].decode('utf-8', errors='replace')
    lines = text_before.split('\n')
    row = len(lines) - 1
    col = len(lines[-1].encode('utf-8')) if lines else 0
    return (row, col)


def _compute_edit_descriptors(
    old_source: bytes,
    new_source: bytes
) -> List[EditDescriptor]:
    """
    Compute edit descriptors between old and new source using difflib.

    Returns a list of EditDescriptor objects describing the changes
    needed to transform old_source into new_source.

    The key insight is that tree-sitter's tree.edit() needs to be called
    for each region that changed, with byte offsets and point coordinates.
    """
    # Use SequenceMatcher to find matching blocks
    matcher = difflib.SequenceMatcher(None, old_source, new_source, autojunk=False)

    edits = []

    # Get opcodes: 'equal', 'replace', 'insert', 'delete'
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue

        # For replace/insert/delete, we need to describe the edit
        start_byte = i1
        old_end_byte = i2
        new_end_byte = i1 + (j2 - j1)

        start_point = _byte_offset_to_point(old_source, start_byte)
        old_end_point = _byte_offset_to_point(old_source, old_end_byte)
        new_end_point = _byte_offset_to_point(new_source, j2)

        edits.append(EditDescriptor(
            start_byte=start_byte,
            old_end_byte=old_end_byte,
            new_end_byte=new_end_byte,
            start_point=start_point,
            old_end_point=old_end_point,
            new_end_point=new_end_point,
        ))

    return edits


class BaseParser(ABC):
    """
    Abstract base class for language-specific tree-sitter parsers.

    Subclasses must implement:
    - _get_language(): Return the tree-sitter Language object
    - _extract_semantics(): Extract semantic info from the parse tree
    """

    # Maximum source size to parse (10MB)
    MAX_SOURCE_SIZE = 10 * 1024 * 1024

    # Cache for incremental parsing states
    _parse_states: Dict[str, ParseState] = {}

    # Maximum cached states (LRU eviction when exceeded)
    MAX_CACHED_STATES = 100

    def __init__(self, enable_incremental: bool = True):
        """
        Initialize the parser.

        Args:
            enable_incremental: Enable incremental parsing for file updates
        """
        self.enable_incremental = enable_incremental
        self._parser: Optional[Parser] = None
        self._language: Optional[Language] = None

    @property
    def parser(self) -> Parser:
        """Lazy initialization of the tree-sitter parser."""
        if self._parser is None:
            self._language = self._get_language()
            self._parser = Parser(self._language)
        return self._parser

    @property
    def language(self) -> Language:
        """Get the tree-sitter Language object."""
        if self._language is None:
            self._language = self._get_language()
        return self._language

    @abstractmethod
    def _get_language(self) -> Language:
        """Return the tree-sitter Language for this parser."""
        pass

    @abstractmethod
    def _extract_semantics(self, tree: Tree, source_bytes: bytes) -> SemanticInfo:
        """Extract semantic information from the parse tree."""
        pass

    @property
    @abstractmethod
    def language_name(self) -> str:
        """Return the language name (e.g., 'python', 'javascript')."""
        pass

    def parse(
        self,
        source: str,
        file_path: Optional[str] = None,
        language_result: Optional[Any] = None,
    ) -> ASTResult:
        """
        Parse source code and return an ASTResult.

        Args:
            source: Source code string
            file_path: Optional file path for incremental parsing
            language_result: Optional LanguageResult from detection

        Returns:
            ASTResult with parsed AST and semantic information
        """
        # Check source size
        if len(source) > self.MAX_SOURCE_SIZE:
            return ASTResult(
                language=self.language_name,
                source_language_result=language_result,
                parse_confidence=0.0,
                errors=[ErrorInfo(
                    message=f"Source too large: {len(source)} bytes exceeds {self.MAX_SOURCE_SIZE}",
                    error_type="size_limit",
                )],
            )

        source_bytes = source.encode('utf-8')
        source_hash = hashlib.sha256(source_bytes).hexdigest()[:16]

        # Check for incremental parsing opportunity
        is_incremental = False
        old_tree = None
        old_source_bytes = None

        if self.enable_incremental and file_path:
            state = self._parse_states.get(file_path)
            if state and state.source_hash != source_hash:
                # Can do incremental parse - store old source for diff computation
                old_tree = state.tree
                old_source_bytes = state.source_bytes
                is_incremental = True

        # Parse the source
        if old_tree and old_source_bytes is not None:
            # Compute edit descriptors and apply them to the tree
            edits = _compute_edit_descriptors(old_source_bytes, source_bytes)
            for edit in edits:
                old_tree.edit(
                    start_byte=edit.start_byte,
                    old_end_byte=edit.old_end_byte,
                    new_end_byte=edit.new_end_byte,
                    start_point=edit.start_point,
                    old_end_point=edit.old_end_point,
                    new_end_point=edit.new_end_point,
                )
            tree = self.parser.parse(source_bytes, old_tree)
        else:
            tree = self.parser.parse(source_bytes)

        # Count nodes and errors
        total_nodes, error_count, errors = self._analyze_tree(tree, source_bytes)

        # Calculate confidence based on error ratio
        parse_confidence = self._calculate_confidence(total_nodes, error_count)

        # Extract semantic information
        semantic = self._extract_semantics(tree, source_bytes)

        # Store state for future incremental parsing
        if self.enable_incremental and file_path:
            self._store_parse_state(file_path, tree, source_bytes, source_hash)

        return ASTResult(
            language=self.language_name,
            source_language_result=language_result,
            parse_tree=tree,
            root_node=tree.root_node,
            semantic=semantic,
            parse_confidence=parse_confidence,
            total_nodes=total_nodes,
            error_count=error_count,
            errors=errors,
            is_incremental=is_incremental,
            source_bytes=source_bytes,
            source_hash=source_hash,
            file_path=file_path,
        )

    def _analyze_tree(
        self,
        tree: Tree,
        source_bytes: bytes
    ) -> Tuple[int, int, List[ErrorInfo]]:
        """
        Analyze parse tree for node count and errors.

        Returns:
            Tuple of (total_nodes, error_count, error_list)
        """
        total_nodes = 0
        error_count = 0
        errors = []

        def visit(node: Node):
            nonlocal total_nodes, error_count
            total_nodes += 1

            if node.is_error or node.is_missing:
                error_count += 1
                errors.append(self._create_error_info(node, source_bytes))

            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return total_nodes, error_count, errors

    def _create_error_info(self, node: Node, source_bytes: bytes) -> ErrorInfo:
        """Create ErrorInfo from an error node."""
        # Get context around the error
        start = max(0, node.start_byte - 20)
        end = min(len(source_bytes), node.end_byte + 20)
        context = source_bytes[start:end].decode('utf-8', errors='replace')

        error_type = "missing" if node.is_missing else "syntax"
        message = f"{error_type.capitalize()} error at line {node.start_point[0] + 1}"

        return ErrorInfo(
            message=message,
            error_type=error_type,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_column=node.start_point[1],
            end_column=node.end_point[1],
            context=context,
        )

    def _calculate_confidence(self, total_nodes: int, error_count: int) -> float:
        """Calculate parse confidence based on error ratio."""
        if total_nodes == 0:
            return 0.0

        error_ratio = error_count / total_nodes
        # Use exponential decay for confidence
        # 0 errors = 1.0, high errors = approaches 0
        confidence = max(0.0, 1.0 - (error_ratio * 5))
        return min(1.0, confidence)

    def _store_parse_state(
        self,
        file_path: str,
        tree: Tree,
        source_bytes: bytes,
        source_hash: str
    ):
        """Store parse state for incremental parsing."""
        # LRU eviction if cache is full
        if len(self._parse_states) >= self.MAX_CACHED_STATES:
            oldest_key = min(
                self._parse_states.keys(),
                key=lambda k: self._parse_states[k].timestamp
            )
            del self._parse_states[oldest_key]

        self._parse_states[file_path] = ParseState(
            tree=tree,
            source_bytes=source_bytes,
            source_hash=source_hash,
            timestamp=time.time(),
            file_path=file_path,
        )

    def clear_cache(self, file_path: Optional[str] = None):
        """Clear incremental parsing cache."""
        if file_path:
            self._parse_states.pop(file_path, None)
        else:
            self._parse_states.clear()

    # Helper methods for subclasses

    def _get_node_text(self, node: Node, source_bytes: bytes) -> str:
        """Get the text content of a node."""
        return source_bytes[node.start_byte:node.end_byte].decode('utf-8')

    def _find_child_by_type(self, node: Node, type_name: str) -> Optional[Node]:
        """Find the first child node of a given type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    def _find_children_by_type(self, node: Node, type_name: str) -> List[Node]:
        """Find all child nodes of a given type."""
        return [child for child in node.children if child.type == type_name]

    def _get_location(self, node: Node) -> Tuple[int, int, int, int]:
        """Get (start_line, end_line, start_col, end_col) for a node."""
        return (
            node.start_point[0] + 1,  # 1-indexed
            node.end_point[0] + 1,
            node.start_point[1],
            node.end_point[1],
        )

    def _traverse_tree(
        self,
        node: Node,
        callback: Callable[[Node], None],
        filter_types: Optional[List[str]] = None
    ):
        """Traverse tree and call callback on each node."""
        if filter_types is None or node.type in filter_types:
            callback(node)
        for child in node.children:
            self._traverse_tree(child, callback, filter_types)
