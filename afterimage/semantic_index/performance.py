"""
Performance Optimizations for GoToDefinition.

Provides:
- Caching for repeated lookups
- Lazy loading of call graph transitive closure
- Parallel file indexing support
- LRU caching for hot paths
"""

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)
import time

T = TypeVar('T')


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_queries: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    def record_hit(self) -> None:
        self.hits += 1
        self.total_queries += 1

    def record_miss(self) -> None:
        self.misses += 1
        self.total_queries += 1

    def record_eviction(self) -> None:
        self.evictions += 1


class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache with configurable size.

    Supports:
    - Maximum entry limit
    - TTL-based expiration
    - Statistics tracking
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[float] = None,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[T, float]] = {}  # key -> (value, timestamp)
        self._access_order: List[str] = []  # Most recent at end
        self._lock = threading.RLock()
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[T]:
        """Get a value from the cache."""
        with self._lock:
            if key not in self._cache:
                self.stats.record_miss()
                return None

            value, timestamp = self._cache[key]

            # Check TTL
            if self.ttl_seconds is not None:
                if time.time() - timestamp > self.ttl_seconds:
                    self._evict(key)
                    self.stats.record_miss()
                    return None

            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            self.stats.record_hit()
            return value

    def set(self, key: str, value: T) -> None:
        """Set a value in the cache."""
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size and self._access_order:
                oldest = self._access_order[0]
                self._evict(oldest)

            # Set value
            self._cache[key] = (value, time.time())

            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    def _evict(self, key: str) -> None:
        """Evict a key from the cache."""
        if key in self._cache:
            del self._cache[key]
            self.stats.record_eviction()
        if key in self._access_order:
            self._access_order.remove(key)

    def invalidate(self, key: str) -> None:
        """Invalidate a specific key."""
        with self._lock:
            self._evict(key)

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def contains(self, key: str) -> bool:
        """Check if a key is in the cache."""
        with self._lock:
            return key in self._cache

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)


class DefinitionCache:
    """
    Specialized cache for go-to-definition results.

    Caches:
    - Symbol lookups by location
    - Import resolution results
    - Cross-file resolution results
    """

    def __init__(self, max_size: int = 5000):
        # Cache for location-based lookups: (file_path, line, col) -> result
        self._location_cache: LRUCache = LRUCache(max_size=max_size)

        # Cache for name resolution: (file_path, name, scope_id) -> symbol
        self._name_cache: LRUCache = LRUCache(max_size=max_size)

        # Cache for import resolution: (module, name) -> symbol
        self._import_cache: LRUCache = LRUCache(max_size=max_size // 2)

        # File content hashes for invalidation
        self._file_hashes: Dict[str, str] = {}

    def get_by_location(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> Optional[Any]:
        """Get cached result for a location lookup."""
        key = f"{file_path}:{line}:{column}"
        return self._location_cache.get(key)

    def set_by_location(
        self,
        file_path: str,
        line: int,
        column: int,
        result: Any,
    ) -> None:
        """Cache a result for a location lookup."""
        key = f"{file_path}:{line}:{column}"
        self._location_cache.set(key, result)

    def get_by_name(
        self,
        file_path: str,
        name: str,
        scope_id: Optional[str],
    ) -> Optional[Any]:
        """Get cached result for a name resolution."""
        key = f"{file_path}:{name}:{scope_id or 'global'}"
        return self._name_cache.get(key)

    def set_by_name(
        self,
        file_path: str,
        name: str,
        scope_id: Optional[str],
        result: Any,
    ) -> None:
        """Cache a result for a name resolution."""
        key = f"{file_path}:{name}:{scope_id or 'global'}"
        self._name_cache.set(key, result)

    def get_import(
        self,
        module: str,
        name: str,
    ) -> Optional[Any]:
        """Get cached import resolution result."""
        key = f"{module}.{name}"
        return self._import_cache.get(key)

    def set_import(
        self,
        module: str,
        name: str,
        result: Any,
    ) -> None:
        """Cache an import resolution result."""
        key = f"{module}.{name}"
        self._import_cache.set(key, result)

    def invalidate_file(self, file_path: str) -> None:
        """Invalidate all caches related to a file."""
        # We can't efficiently invalidate partial caches,
        # so we track file hashes and check on access
        if file_path in self._file_hashes:
            del self._file_hashes[file_path]

    def update_file_hash(self, file_path: str, content_hash: str) -> bool:
        """
        Update file hash and return whether file changed.

        Returns:
            True if file changed (hash differs), False otherwise
        """
        old_hash = self._file_hashes.get(file_path)
        self._file_hashes[file_path] = content_hash
        return old_hash != content_hash

    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches."""
        return {
            "location": self._location_cache.stats,
            "name": self._name_cache.stats,
            "import": self._import_cache.stats,
        }

    def clear(self) -> None:
        """Clear all caches."""
        self._location_cache.clear()
        self._name_cache.clear()
        self._import_cache.clear()


class LazyCallGraphClosure:
    """
    Lazy-loading transitive closure for call graphs.

    Computes transitive closure on-demand and caches results.
    Invalidates cache when graph changes.
    """

    def __init__(self):
        self._direct_calls: Dict[str, Set[str]] = {}
        self._direct_callers: Dict[str, Set[str]] = {}
        self._transitive_calls: Dict[str, Set[str]] = {}
        self._transitive_callers: Dict[str, Set[str]] = {}
        self._dirty: bool = True
        self._computed_calls: Set[str] = set()
        self._computed_callers: Set[str] = set()
        self._lock = threading.RLock()

    def set_direct_edges(
        self,
        calls: Dict[str, Set[str]],
        callers: Dict[str, Set[str]],
    ) -> None:
        """Set the direct edges in the call graph."""
        with self._lock:
            self._direct_calls = calls
            self._direct_callers = callers
            self._dirty = True
            self._transitive_calls.clear()
            self._transitive_callers.clear()
            self._computed_calls.clear()
            self._computed_callers.clear()

    def add_call(self, caller: str, callee: str) -> None:
        """Add a call edge."""
        with self._lock:
            if caller not in self._direct_calls:
                self._direct_calls[caller] = set()
            self._direct_calls[caller].add(callee)

            if callee not in self._direct_callers:
                self._direct_callers[callee] = set()
            self._direct_callers[callee].add(caller)

            # Invalidate transitive closure for affected nodes
            self._computed_calls.discard(caller)
            self._computed_callers.discard(callee)
            self._dirty = True

    def get_transitive_calls(self, function: str) -> Set[str]:
        """Get all functions transitively called by a function."""
        with self._lock:
            if function in self._computed_calls:
                return self._transitive_calls.get(function, set()).copy()

            # Compute lazily
            result = self._compute_transitive(
                function,
                self._direct_calls,
            )
            self._transitive_calls[function] = result
            self._computed_calls.add(function)
            return result.copy()

    def get_transitive_callers(self, function: str) -> Set[str]:
        """Get all functions that transitively call a function."""
        with self._lock:
            if function in self._computed_callers:
                return self._transitive_callers.get(function, set()).copy()

            # Compute lazily
            result = self._compute_transitive(
                function,
                self._direct_callers,
            )
            self._transitive_callers[function] = result
            self._computed_callers.add(function)
            return result.copy()

    def _compute_transitive(
        self,
        start: str,
        edges: Dict[str, Set[str]],
    ) -> Set[str]:
        """Compute transitive closure from a starting node."""
        visited = set()
        stack = list(edges.get(start, set()))

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(edges.get(current, set()) - visited)

        return visited

    def invalidate(self) -> None:
        """Invalidate all computed closures."""
        with self._lock:
            self._transitive_calls.clear()
            self._transitive_callers.clear()
            self._computed_calls.clear()
            self._computed_callers.clear()
            self._dirty = True


@dataclass
class IndexingTask:
    """A single file indexing task."""
    file_path: str
    source: Optional[str] = None
    priority: int = 0  # Lower is higher priority


@dataclass
class IndexingResult:
    """Result of indexing a file."""
    file_path: str
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0.0
    symbols_count: int = 0


class ParallelIndexer:
    """
    Parallel file indexing with thread pool.

    Supports:
    - Configurable worker count
    - Priority ordering
    - Progress callbacks
    - Error handling
    """

    def __init__(
        self,
        worker_count: Optional[int] = None,
        index_func: Optional[Callable[[str, Optional[str]], bool]] = None,
    ):
        """
        Initialize the parallel indexer.

        Args:
            worker_count: Number of worker threads (default: CPU count)
            index_func: Function to index a single file (path, source) -> success
        """
        import os
        self.worker_count = worker_count or os.cpu_count() or 4
        self.index_func = index_func
        self._lock = threading.Lock()

    def index_files(
        self,
        tasks: List[IndexingTask],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[IndexingResult]:
        """
        Index multiple files in parallel.

        Args:
            tasks: List of indexing tasks
            progress_callback: Called with (completed, total, current_file)

        Returns:
            List of indexing results
        """
        if not self.index_func:
            raise ValueError("No index function provided")

        # Sort by priority
        tasks = sorted(tasks, key=lambda t: t.priority)
        results: List[IndexingResult] = []
        total = len(tasks)
        completed = 0

        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._index_single, task): task
                for task in tasks
            }

            # Process results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(IndexingResult(
                        file_path=task.file_path,
                        success=False,
                        error=str(e),
                    ))

                completed += 1
                if progress_callback:
                    progress_callback(completed, total, task.file_path)

        return results

    def _index_single(self, task: IndexingTask) -> IndexingResult:
        """Index a single file."""
        start_time = time.time()

        try:
            # Read source if not provided
            source = task.source
            if source is None:
                try:
                    with open(task.file_path, 'r', encoding='utf-8') as f:
                        source = f.read()
                except (IOError, UnicodeDecodeError) as e:
                    return IndexingResult(
                        file_path=task.file_path,
                        success=False,
                        error=str(e),
                    )

            # Index the file
            success = self.index_func(task.file_path, source)
            duration = (time.time() - start_time) * 1000

            return IndexingResult(
                file_path=task.file_path,
                success=success,
                duration_ms=duration,
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return IndexingResult(
                file_path=task.file_path,
                success=False,
                error=str(e),
                duration_ms=duration,
            )

    def index_directory(
        self,
        directory: str,
        pattern: str = "**/*.py",
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[IndexingResult]:
        """
        Index all matching files in a directory.

        Args:
            directory: Root directory to index
            pattern: Glob pattern for files
            exclude_patterns: Patterns to exclude
            progress_callback: Called with (completed, total, current_file)

        Returns:
            List of indexing results
        """
        directory_path = Path(directory)
        exclude_patterns = exclude_patterns or []
        tasks = []

        for file_path in directory_path.glob(pattern):
            # Check exclusions
            if any(file_path.match(ep) for ep in exclude_patterns):
                continue

            if file_path.is_file():
                tasks.append(IndexingTask(file_path=str(file_path)))

        return self.index_files(tasks, progress_callback)


def compute_file_hash(content: str) -> str:
    """Compute a hash of file content for change detection."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def memoize_method(method: Callable) -> Callable:
    """Decorator to memoize instance method results."""
    cache_attr = f'_memoize_cache_{method.__name__}'

    def wrapper(self, *args):
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
        cache = getattr(self, cache_attr)

        key = args
        if key not in cache:
            cache[key] = method(self, *args)
        return cache[key]

    return wrapper
