"""
Semantic Index - Project-level coordinator for semantic intelligence.

Manages symbol tables across multiple files and provides unified
interfaces for go-to-definition, find-references, and hover.
Supports incremental updates when files change.

Enhanced with:
- JavaScript/TypeScript support
- Type inference for Python
- Performance caching
- Parallel indexing
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Callable

from .models import (
    Symbol,
    Reference,
    Location,
    DefinitionResult,
    ReferenceResult,
    HoverInfo,
    CallSite,
    TypeInfo,
)
from .symbol_table import SymbolTable
from .call_graph import CallGraph
from .performance import (
    DefinitionCache,
    ParallelIndexer,
    IndexingTask,
    IndexingResult,
    compute_file_hash,
)
from .visitors.python_visitor import PythonSymbolVisitor
from .definition_resolver import DefinitionResolver
from .references_finder import ReferencesFinder
from .hover_provider import HoverProvider
from .type_inference import TypeInferencer, TypePropagator

# Import JavaScript visitor with fallback
try:
    from .visitors.javascript_visitor import (
        JavaScriptSymbolVisitor,
        TypeScriptSymbolVisitor,
    )
    HAS_JS_SUPPORT = True
except ImportError:
    HAS_JS_SUPPORT = False
    JavaScriptSymbolVisitor = None
    TypeScriptSymbolVisitor = None

# Import Rust visitor with fallback
try:
    from .visitors.rust_visitor import RustSymbolVisitor
    HAS_RUST_SUPPORT = True
except ImportError:
    HAS_RUST_SUPPORT = False
    RustSymbolVisitor = None


@dataclass
class FileState:
    """Tracks the state of an indexed file."""
    file_path: str
    source_hash: str
    symbol_table: SymbolTable
    module_name: Optional[str] = None
    last_modified: Optional[float] = None


@dataclass
class IndexStats:
    """Statistics about the semantic index."""
    total_files: int = 0
    total_symbols: int = 0
    total_references: int = 0
    total_scopes: int = 0
    total_call_sites: int = 0


class SemanticIndex:
    """
    Project-level semantic index coordinator.

    Provides:
    - Multi-file symbol table management
    - Unified go-to-definition, find-references, hover
    - Incremental updates when files change
    - Project-wide call graph
    - JavaScript/TypeScript support
    - Type inference for Python
    - Performance caching
    """

    # Supported file extensions and their languages
    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.mjs': 'javascript',
        '.cjs': 'javascript',
        '.rs': 'rust',
    }

    def __init__(
        self,
        project_root: Optional[str] = None,
        enable_caching: bool = True,
        enable_type_inference: bool = True,
    ):
        self.project_root = Path(project_root) if project_root else None

        # File tracking
        self.files: Dict[str, FileState] = {}

        # Core components
        self.definition_resolver = DefinitionResolver()
        self.references_finder = ReferencesFinder()
        self.hover_provider = HoverProvider()

        # Project-wide call graph
        self.call_graph = CallGraph()

        # File call graphs (for incremental updates)
        self._file_call_graphs: Dict[str, CallGraph] = {}

        # Dependency tracking for incremental updates
        self._dependents: Dict[str, Set[str]] = {}  # file -> files that import it
        self._dependencies: Dict[str, Set[str]] = {}  # file -> files it imports

        # Visitors for parsing different languages
        self._python_visitor = PythonSymbolVisitor()
        self._js_visitor = JavaScriptSymbolVisitor() if HAS_JS_SUPPORT else None
        self._ts_visitor = TypeScriptSymbolVisitor() if HAS_JS_SUPPORT else None
        self._rust_visitor = RustSymbolVisitor() if HAS_RUST_SUPPORT else None

        # Type inference
        self._enable_type_inference = enable_type_inference
        self._type_inferencer = TypeInferencer() if enable_type_inference else None
        self._type_propagator = TypePropagator(self._type_inferencer) if enable_type_inference else None

        # Performance caching
        self._enable_caching = enable_caching
        self._cache = DefinitionCache() if enable_caching else None

        # Parallel indexer (initialized lazily)
        self._parallel_indexer: Optional[ParallelIndexer] = None

    def _get_visitor_for_file(self, file_path: str):
        """Get the appropriate visitor for a file based on its extension."""
        ext = Path(file_path).suffix.lower()
        language = self.SUPPORTED_EXTENSIONS.get(ext)

        if language == 'python':
            return self._python_visitor
        elif language == 'javascript' and self._js_visitor:
            return self._js_visitor
        elif language == 'typescript' and self._ts_visitor:
            return self._ts_visitor
        elif language == 'rust' and self._rust_visitor:
            return self._rust_visitor
        elif language in ('javascript', 'typescript', 'rust'):
            return None  # Language support not available
        else:
            return self._python_visitor  # Default fallback

    def _get_language_for_file(self, file_path: str) -> str:
        """Get the language name for a file based on its extension."""
        ext = Path(file_path).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, 'python')

    def index_file(
        self,
        file_path: str,
        source: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """
        Index a single file.

        Args:
            file_path: Path to the file
            source: Optional source code (reads from file if not provided)
            force: Force re-indexing even if unchanged

        Returns:
            True if file was indexed/updated, False if unchanged
        """
        file_path = str(Path(file_path).resolve())

        # Get appropriate visitor
        visitor = self._get_visitor_for_file(file_path)
        if visitor is None:
            return False  # Unsupported file type

        # Read source if not provided
        if source is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
            except (IOError, UnicodeDecodeError) as e:
                return False

        # Check cache for unchanged files
        source_hash = self._compute_hash(source)
        if self._enable_caching and self._cache:
            if not self._cache.update_file_hash(file_path, source_hash) and not force:
                if file_path in self.files:
                    return False  # File unchanged

        # Check if file has changed (backup check without cache)
        if not force and file_path in self.files:
            if self.files[file_path].source_hash == source_hash:
                return False  # File unchanged

        # Parse the file with appropriate visitor
        symbol_table, file_call_graph = visitor.visit(source, file_path)
        if not symbol_table:
            return False

        # Run type inference for Python files
        language = self._get_language_for_file(file_path)
        if language == 'python' and self._enable_type_inference and self._type_inferencer:
            self._type_inferencer.infer_types(symbol_table)
            if self._type_propagator:
                self._type_propagator.propagate(symbol_table)

        # Determine module name
        module_name = self._compute_module_name(file_path)
        symbol_table.module_name = module_name

        # Remove old state if exists
        if file_path in self.files:
            self._unregister_file(file_path)

        # Register with all components
        self.definition_resolver.register_file(file_path, symbol_table, module_name)
        self.references_finder.register_file(file_path, symbol_table)
        self.hover_provider.register_file(file_path, symbol_table)

        # Update call graph
        self._file_call_graphs[file_path] = file_call_graph
        self._merge_call_graph(file_call_graph)

        # Track dependencies
        self._update_dependencies(file_path, symbol_table)

        # Store file state
        self.files[file_path] = FileState(
            file_path=file_path,
            source_hash=source_hash,
            symbol_table=symbol_table,
            module_name=module_name,
        )

        return True

    def index_directory(
        self,
        directory: str,
        pattern: str = "**/*.py",
        exclude_patterns: Optional[List[str]] = None,
    ) -> int:
        """
        Index all matching files in a directory.

        Args:
            directory: Root directory to index
            pattern: Glob pattern for files
            exclude_patterns: Patterns to exclude

        Returns:
            Number of files indexed
        """
        directory = Path(directory)
        exclude_patterns = exclude_patterns or []
        indexed = 0

        for file_path in directory.glob(pattern):
            # Check exclusions
            if any(file_path.match(ep) for ep in exclude_patterns):
                continue

            if file_path.is_file():
                if self.index_file(str(file_path)):
                    indexed += 1

        return indexed

    def update_file(
        self,
        file_path: str,
        source: Optional[str] = None,
    ) -> Tuple[bool, Set[str]]:
        """
        Incrementally update a file and return affected files.

        Args:
            file_path: Path to the file
            source: Optional new source code

        Returns:
            Tuple of (was_updated, set of affected file paths)
        """
        file_path = str(Path(file_path).resolve())

        # Invalidate cache for this file before re-indexing
        if self._enable_caching and self._cache:
            self._cache.invalidate_file(file_path)

        # Index the file
        updated = self.index_file(file_path, source, force=True)
        if not updated:
            return False, set()

        # Find affected files (files that depend on this one)
        affected = self._dependents.get(file_path, set()).copy()

        # Re-resolve references in affected files and invalidate their caches
        for affected_path in affected:
            if self._enable_caching and self._cache:
                self._cache.invalidate_file(affected_path)
            if affected_path in self.files:
                self._reindex_references(affected_path)

        return True, affected

    def remove_file(self, file_path: str) -> bool:
        """Remove a file from the index."""
        file_path = str(Path(file_path).resolve())
        if file_path not in self.files:
            return False

        self._unregister_file(file_path)
        del self.files[file_path]
        return True

    def go_to_definition(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> DefinitionResult:
        """
        Find the definition of the symbol at the given location.

        Args:
            file_path: Path to the source file
            line: Line number (1-indexed)
            column: Column number (0-indexed)

        Returns:
            DefinitionResult with the definition location
        """
        file_path = str(Path(file_path).resolve())

        # Check cache first
        if self._enable_caching and self._cache:
            cached = self._cache.get_by_location(file_path, line, column)
            if cached is not None:
                return cached

        # Perform resolution
        result = self.definition_resolver.go_to_definition(file_path, line, column)

        # Cache the result
        if self._enable_caching and self._cache and result.success:
            self._cache.set_by_location(file_path, line, column, result)

        return result

    def find_references(
        self,
        file_path: str,
        line: int,
        column: int,
        include_declaration: bool = True,
    ) -> ReferenceResult:
        """
        Find all references to the symbol at the given location.

        Args:
            file_path: Path to the source file
            line: Line number (1-indexed)
            column: Column number (0-indexed)
            include_declaration: Whether to include the declaration itself

        Returns:
            ReferenceResult with all found references
        """
        file_path = str(Path(file_path).resolve())
        return self.references_finder.find_references(
            file_path, line, column, include_declaration
        )

    def get_hover(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> Optional[HoverInfo]:
        """
        Get hover information for the symbol at the given location.

        Args:
            file_path: Path to the source file
            line: Line number (1-indexed)
            column: Column number (0-indexed)

        Returns:
            HoverInfo if a symbol is found, None otherwise
        """
        file_path = str(Path(file_path).resolve())
        return self.hover_provider.get_hover(file_path, line, column)

    def get_call_graph(self) -> CallGraph:
        """Get the project-wide call graph."""
        return self.call_graph

    def get_callers(self, function_name: str) -> List[str]:
        """Get all functions that call the given function."""
        return list(self.call_graph.get_direct_callers(function_name))

    def get_callees(self, function_name: str) -> List[str]:
        """Get all functions called by the given function."""
        return list(self.call_graph.get_direct_calls(function_name))

    def get_symbol(self, file_path: str, qualified_name: str) -> Optional[Symbol]:
        """Get a symbol by its qualified name in a file."""
        file_path = str(Path(file_path).resolve())
        if file_path not in self.files:
            return None
        return self.files[file_path].symbol_table.get_symbol(qualified_name)

    def get_all_symbols(self, file_path: str) -> List[Symbol]:
        """Get all symbols in a file."""
        file_path = str(Path(file_path).resolve())
        if file_path not in self.files:
            return []
        return list(self.files[file_path].symbol_table.get_all_symbols())

    def search_symbols(
        self,
        query: str,
        file_path: Optional[str] = None,
    ) -> List[Symbol]:
        """
        Search for symbols matching a query.

        Args:
            query: Search query (simple substring match)
            file_path: Optionally limit to a specific file

        Returns:
            List of matching symbols
        """
        results = []
        query_lower = query.lower()

        files = (
            [self.files[file_path]]
            if file_path and file_path in self.files
            else self.files.values()
        )

        for file_state in files:
            for symbol in file_state.symbol_table.get_all_symbols():
                if query_lower in symbol.name.lower():
                    results.append(symbol)

        return results

    def get_stats(self) -> IndexStats:
        """Get statistics about the index."""
        stats = IndexStats(total_files=len(self.files))

        for file_state in self.files.values():
            table = file_state.symbol_table
            stats.total_symbols += len(table.symbols)
            stats.total_references += len(table.references)
            stats.total_scopes += len(table.scopes)

        stats.total_call_sites = sum(
            len(node.calls) for node in self.call_graph.nodes.values()
        )

        return stats

    def get_unused_symbols(self, file_path: str) -> List[Symbol]:
        """Find symbols in a file that have no references."""
        file_path = str(Path(file_path).resolve())
        return self.references_finder.find_unused_symbols(file_path)

    def index_directory_parallel(
        self,
        directory: str,
        pattern: str = "**/*",
        exclude_patterns: Optional[List[str]] = None,
        worker_count: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[IndexingResult]:
        """
        Index all matching files in a directory using parallel processing.

        Args:
            directory: Root directory to index
            pattern: Glob pattern for files (will filter by supported extensions)
            exclude_patterns: Patterns to exclude
            worker_count: Number of worker threads (default: CPU count)
            progress_callback: Called with (completed, total, current_file)

        Returns:
            List of IndexingResult for each file
        """
        directory_path = Path(directory)
        exclude_patterns = exclude_patterns or []
        tasks = []

        # Collect files with supported extensions
        for file_path in directory_path.glob(pattern):
            if not file_path.is_file():
                continue
            if any(file_path.match(ep) for ep in exclude_patterns):
                continue
            ext = file_path.suffix.lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                continue
            tasks.append(IndexingTask(file_path=str(file_path)))

        if not tasks:
            return []

        # Initialize parallel indexer lazily
        if self._parallel_indexer is None:
            self._parallel_indexer = ParallelIndexer(
                worker_count=worker_count,
                index_func=lambda path, source: self.index_file(path, source),
            )

        return self._parallel_indexer.index_files(tasks, progress_callback)

    def get_cache_stats(self) -> Optional[Dict]:
        """Get statistics about the definition cache."""
        if not self._enable_caching or not self._cache:
            return None
        return self._cache.get_stats()

    def clear_cache(self) -> None:
        """Clear all caches."""
        if self._cache:
            self._cache.clear()
        if self._type_inferencer:
            self._type_inferencer.clear_cache()
        self.definition_resolver.clear_cache()

    def get_inferred_type(self, file_path: str, symbol_name: str) -> Optional[TypeInfo]:
        """Get the inferred type for a symbol."""
        if not self._type_inferencer:
            return None
        file_path = str(Path(file_path).resolve())
        if file_path not in self.files:
            return None
        return self._type_inferencer.get_cached_type(symbol_name)

    def has_js_support(self) -> bool:
        """Check if JavaScript/TypeScript support is available."""
        return HAS_JS_SUPPORT

    def has_rust_support(self) -> bool:
        """Check if Rust support is available."""
        return HAS_RUST_SUPPORT

    def _compute_hash(self, source: str) -> str:
        """Compute a hash of the source code."""
        return hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]

    def _compute_module_name(self, file_path: str) -> Optional[str]:
        """Compute a module name from a file path."""
        path = Path(file_path)

        if self.project_root:
            try:
                relative = path.relative_to(self.project_root)
                parts = list(relative.parts)
                # Remove .py extension from last part
                if parts and parts[-1].endswith('.py'):
                    parts[-1] = parts[-1][:-3]
                    if parts[-1] == '__init__':
                        parts = parts[:-1]
                return '.'.join(parts) if parts else None
            except ValueError:
                pass

        # Fallback: just use the filename
        name = path.stem
        return name if name != '__init__' else path.parent.name

    def _unregister_file(self, file_path: str) -> None:
        """Unregister a file from all components."""
        self.definition_resolver.unregister_file(file_path)
        self.references_finder.unregister_file(file_path)
        self.hover_provider.unregister_file(file_path)

        # Remove from call graph
        if file_path in self._file_call_graphs:
            # Remove all calls from this file
            file_cg = self._file_call_graphs[file_path]
            for func in list(file_cg.nodes.keys()):
                if func in self.call_graph.nodes:
                    del self.call_graph.nodes[func]
            del self._file_call_graphs[file_path]

        # Clean up dependencies
        if file_path in self._dependencies:
            for dep in self._dependencies[file_path]:
                if dep in self._dependents:
                    self._dependents[dep].discard(file_path)
            del self._dependencies[file_path]

        if file_path in self._dependents:
            del self._dependents[file_path]

    def _merge_call_graph(self, file_call_graph: CallGraph) -> None:
        """Merge a file's call graph into the project-wide graph."""
        # Use the merge method from CallGraph
        self.call_graph.merge(file_call_graph)

    def _update_dependencies(
        self,
        file_path: str,
        symbol_table: SymbolTable,
    ) -> None:
        """Update dependency tracking for a file."""
        # Find all imports in this file
        imported_modules = set()
        for symbol in symbol_table.get_all_symbols():
            if symbol.imported_from:
                # Extract module name
                parts = symbol.imported_from.split('.')
                module = parts[0]
                imported_modules.add(module)

        # Map modules to files
        dependencies = set()
        for module in imported_modules:
            if module in self.definition_resolver.module_paths:
                dep_file = self.definition_resolver.module_paths[module]
                dependencies.add(dep_file)

        # Update dependency maps
        self._dependencies[file_path] = dependencies
        for dep in dependencies:
            if dep not in self._dependents:
                self._dependents[dep] = set()
            self._dependents[dep].add(file_path)

    def _reindex_references(self, file_path: str) -> None:
        """Re-resolve references in a file after dependencies changed."""
        if file_path not in self.files:
            return

        file_state = self.files[file_path]
        symbol_table = file_state.symbol_table

        # Re-resolve each reference
        for ref in symbol_table.references:
            if ref.resolved_symbol:
                continue  # Already resolved

            # Try to resolve with updated context
            scope = symbol_table.get_scope_at(
                ref.location.start.line,
                ref.location.start.column,
            )
            resolved = symbol_table.resolve_name(
                ref.name,
                from_scope_id=scope.id if scope else None,
            )
            if resolved:
                ref.resolved_symbol = resolved
                ref.resolved_qualified_name = resolved.qualified_name
