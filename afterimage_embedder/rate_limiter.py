"""
Token Bucket Rate Limiter for AfterImage Embedding Daemon.

Implements rate limiting to prevent overwhelming GPU/database during high-load periods.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterStats:
    """Statistics for rate limiter operations."""
    tokens_acquired: int = 0
    waits_total: int = 0
    timeouts_total: int = 0
    current_tokens: float = 0.0
    capacity: int = 0
    refill_rate: float = 0.0


class TokenBucket:
    """
    Token bucket rate limiter implementation.

    Allows controlled bursting up to capacity, with steady-state refill rate.
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum tokens (burst capacity)
            refill_rate: Tokens per second to refill
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)  # Start with full bucket
        self.last_refill = time.time()
        self._lock = threading.Lock()

        # Stats
        self._tokens_acquired = 0
        self._waits_total = 0
        self._timeouts_total = 0

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                self._tokens_acquired += tokens
                return True
            return False

    def wait_for_tokens(self, tokens: int = 1, timeout: float = 30.0) -> bool:
        """
        Block until tokens are available or timeout.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds

        Returns:
            True if tokens were acquired, False if timed out
        """
        deadline = time.time() + timeout
        poll_interval = 0.1  # 100ms polling
        waited = False

        while True:
            with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self._tokens_acquired += tokens
                    if waited:
                        self._waits_total += 1
                    return True

            # Check timeout
            remaining = deadline - time.time()
            if remaining <= 0:
                with self._lock:
                    self._timeouts_total += 1
                return False

            # Wait before retry
            waited = True
            time.sleep(min(poll_interval, remaining))

    def get_current_tokens(self) -> float:
        """Get current token count (after refill)."""
        with self._lock:
            self._refill()
            return self.tokens

    def get_stats(self) -> RateLimiterStats:
        """Get rate limiter statistics."""
        with self._lock:
            self._refill()
            return RateLimiterStats(
                tokens_acquired=self._tokens_acquired,
                waits_total=self._waits_total,
                timeouts_total=self._timeouts_total,
                current_tokens=self.tokens,
                capacity=self.capacity,
                refill_rate=self.refill_rate
            )

    def reset(self) -> None:
        """Reset bucket to full capacity."""
        with self._lock:
            self.tokens = float(self.capacity)
            self.last_refill = time.time()


class RateLimiter:
    """
    Composite rate limiter with separate buckets for GPU and database operations.
    """

    def __init__(
        self,
        gpu_capacity: int = 50,
        gpu_refill_rate: float = 10.0,
        db_capacity: int = 100,
        db_refill_rate: float = 50.0,
        enabled: bool = True
    ):
        """
        Initialize composite rate limiter.

        Args:
            gpu_capacity: Max GPU operations per burst
            gpu_refill_rate: GPU operations per second
            db_capacity: Max DB writes per burst
            db_refill_rate: DB writes per second
            enabled: Whether rate limiting is active
        """
        self.enabled = enabled
        self._gpu_bucket = TokenBucket(gpu_capacity, gpu_refill_rate)
        self._db_bucket = TokenBucket(db_capacity, db_refill_rate)

        logger.info(
            f"Rate limiter initialized: GPU={gpu_capacity}/{gpu_refill_rate}/s, "
            f"DB={db_capacity}/{db_refill_rate}/s, enabled={enabled}"
        )

    def acquire_gpu(self, tokens: int = 1, wait: bool = True, timeout: float = 30.0) -> bool:
        """
        Acquire tokens for GPU operation.

        Args:
            tokens: Number of tokens (usually 1 per batch)
            wait: Whether to block if tokens unavailable
            timeout: Maximum wait time in seconds

        Returns:
            True if tokens acquired, False otherwise
        """
        if not self.enabled:
            return True

        if wait:
            acquired = self._gpu_bucket.wait_for_tokens(tokens, timeout)
        else:
            acquired = self._gpu_bucket.acquire(tokens)

        if not acquired:
            logger.warning(f"GPU rate limit: could not acquire {tokens} tokens")

        return acquired

    def acquire_db(self, tokens: int = 1, wait: bool = True, timeout: float = 30.0) -> bool:
        """
        Acquire tokens for database operation.

        Args:
            tokens: Number of tokens (usually 1 per entry)
            wait: Whether to block if tokens unavailable
            timeout: Maximum wait time in seconds

        Returns:
            True if tokens acquired, False otherwise
        """
        if not self.enabled:
            return True

        if wait:
            acquired = self._db_bucket.wait_for_tokens(tokens, timeout)
        else:
            acquired = self._db_bucket.acquire(tokens)

        if not acquired:
            logger.warning(f"DB rate limit: could not acquire {tokens} tokens")

        return acquired

    def get_stats(self) -> dict:
        """Get combined rate limiter statistics."""
        gpu_stats = self._gpu_bucket.get_stats()
        db_stats = self._db_bucket.get_stats()

        return {
            "enabled": self.enabled,
            "gpu": {
                "tokens_acquired": gpu_stats.tokens_acquired,
                "waits_total": gpu_stats.waits_total,
                "timeouts_total": gpu_stats.timeouts_total,
                "current_tokens": round(gpu_stats.current_tokens, 1),
                "capacity": gpu_stats.capacity,
                "refill_rate": gpu_stats.refill_rate
            },
            "db": {
                "tokens_acquired": db_stats.tokens_acquired,
                "waits_total": db_stats.waits_total,
                "timeouts_total": db_stats.timeouts_total,
                "current_tokens": round(db_stats.current_tokens, 1),
                "capacity": db_stats.capacity,
                "refill_rate": db_stats.refill_rate
            },
            "total_waits": gpu_stats.waits_total + db_stats.waits_total,
            "total_timeouts": gpu_stats.timeouts_total + db_stats.timeouts_total
        }

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable rate limiting."""
        self.enabled = enabled
        logger.info(f"Rate limiter {'enabled' if enabled else 'disabled'}")

    def reset(self) -> None:
        """Reset both buckets to full capacity."""
        self._gpu_bucket.reset()
        self._db_bucket.reset()
        logger.info("Rate limiter buckets reset to full capacity")
