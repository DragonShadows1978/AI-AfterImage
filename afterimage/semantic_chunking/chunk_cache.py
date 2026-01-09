"""
Chunk Cache: LRU caching layer for semantic chunking results.

Caches chunk results per file to avoid re-parsing the same code repeatedly.
Uses content hash to invalidate cache when file content changes.

Part of AfterImage Semantic Chunking v0.3.0.
"""

import hashlib
import threading
import time
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from collections import OrderedDict

from .chunker import CodeChunk


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached chunking result."""
    chunks: List[CodeChunk]
    content_hash: str
    file_path: str
    timestamp: float = field(default_factory=time.time)
    hits: int = 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "hit_rate": round(self.hit_rate, 3),
            "total_lookups": self.hits + self.misses,
        }


class ChunkCache:
    """
    LRU cache for semantic chunking results.

    Features:
    - Content-hash based invalidation (cache invalidates when code changes)
    - LRU eviction when capacity is exceeded
    - Thread-safe operations
    - TTL-based expiration
    - Detailed statistics for monitoring
    """

    def __init__(
        self,
        max_entries: int = 100,
        ttl_seconds: float = 3600,  # 1 hour default TTL
        enabled: bool = True
    ):
        """
        Initialize the chunk cache.

        Args:
            max_entries: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries (0 = no expiry)
            enabled: Whether caching is enabled
        """
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def _compute_content_hash(self, content: str) -> str:
        """Compute hash of content for cache invalidation."""
        return hashlib.sha256(content.encode('utf-8', errors='replace')).hexdigest()[:16]

    def _compute_cache_key(self, file_path: str, max_chunk_tokens: int) -> str:
        """Compute cache key from file path and chunk settings."""
        key_data = f"{file_path}:{max_chunk_tokens}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def get(
        self,
        file_path: str,
        content: str,
        max_chunk_tokens: int = 500
    ) -> Optional[List[CodeChunk]]:
        """
        Get cached chunks for a file.

        Args:
            file_path: Path to the file
            content: Current content of the file (for hash validation)
            max_chunk_tokens: Chunk token limit used

        Returns:
            List of chunks if cached and valid, None otherwise
        """
        if not self.enabled:
            self._stats.misses += 1
            return None

        cache_key = self._compute_cache_key(file_path, max_chunk_tokens)
        content_hash = self._compute_content_hash(content)

        with self._lock:
            entry = self._cache.get(cache_key)

            if entry is None:
                self._stats.misses += 1
                logger.debug(f"Cache miss (not found): {file_path}")
                return None

            # Check content hash (invalidate if content changed)
            if entry.content_hash != content_hash:
                self._stats.invalidations += 1
                self._stats.misses += 1
                del self._cache[cache_key]
                logger.debug(f"Cache invalidated (content changed): {file_path}")
                return None

            # Check TTL
            if self.ttl_seconds > 0 and entry.age_seconds > self.ttl_seconds:
                self._stats.evictions += 1
                self._stats.misses += 1
                del self._cache[cache_key]
                logger.debug(f"Cache expired (TTL): {file_path}")
                return None

            # Cache hit - move to end for LRU
            self._cache.move_to_end(cache_key)
            entry.hits += 1
            self._stats.hits += 1

            logger.debug(f"Cache hit: {file_path} (hits={entry.hits})")
            return entry.chunks

    def put(
        self,
        file_path: str,
        content: str,
        chunks: List[CodeChunk],
        max_chunk_tokens: int = 500
    ) -> None:
        """
        Store chunks in the cache.

        Args:
            file_path: Path to the file
            content: Content of the file (for hash calculation)
            chunks: Parsed chunks to cache
            max_chunk_tokens: Chunk token limit used
        """
        if not self.enabled:
            return

        cache_key = self._compute_cache_key(file_path, max_chunk_tokens)
        content_hash = self._compute_content_hash(content)

        entry = CacheEntry(
            chunks=chunks,
            content_hash=content_hash,
            file_path=file_path
        )

        with self._lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_entries:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1
                logger.debug(f"Cache eviction (LRU): removed oldest entry")

            self._cache[cache_key] = entry
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)

        logger.debug(f"Cache store: {file_path} ({len(chunks)} chunks)")

    def invalidate(self, file_path: str, max_chunk_tokens: int = 500) -> bool:
        """
        Invalidate cache entry for a file.

        Args:
            file_path: Path to invalidate
            max_chunk_tokens: Chunk token limit (must match)

        Returns:
            True if entry was found and removed
        """
        cache_key = self._compute_cache_key(file_path, max_chunk_tokens)

        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self._stats.invalidations += 1
                logger.debug(f"Cache invalidated (manual): {file_path}")
                return True
        return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.evictions += count
            logger.info(f"Cache cleared: {count} entries")
            return count

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = CacheStats()

    def get_summary(self) -> Dict[str, Any]:
        """Get cache summary for monitoring."""
        with self._lock:
            entries = list(self._cache.values())

            return {
                "enabled": self.enabled,
                "max_entries": self.max_entries,
                "current_entries": len(entries),
                "ttl_seconds": self.ttl_seconds,
                "stats": self._stats.to_dict(),
                "oldest_entry_age": max((e.age_seconds for e in entries), default=0),
                "most_hits_entry": max((e.hits for e in entries), default=0),
            }


# Global cache instance
_global_cache: Optional[ChunkCache] = None
_cache_lock = threading.Lock()


def get_chunk_cache(
    max_entries: int = 100,
    ttl_seconds: float = 3600
) -> ChunkCache:
    """
    Get or create the global chunk cache.

    Args:
        max_entries: Maximum cache entries
        ttl_seconds: TTL for entries

    Returns:
        Global ChunkCache instance
    """
    global _global_cache

    with _cache_lock:
        if _global_cache is None:
            _global_cache = ChunkCache(
                max_entries=max_entries,
                ttl_seconds=ttl_seconds
            )
        return _global_cache


def clear_global_cache() -> int:
    """Clear the global cache. Returns number of entries cleared."""
    global _global_cache

    with _cache_lock:
        if _global_cache is not None:
            return _global_cache.clear()
    return 0
