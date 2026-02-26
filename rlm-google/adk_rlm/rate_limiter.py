"""
Token bucket rate limiter for LLM API calls.

Enforces both concurrent and per-minute rate limits.
"""

import asyncio
import threading
import time
from contextlib import contextmanager
from typing import Optional


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter that enforces both concurrent and RPM limits.

    Order of operations:
    1. Acquire rate limit token (throttles frequency)
    2. Acquire concurrency semaphore (throttles simultaneous calls)
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        max_concurrent: int = 30,
        max_burst: int = 5,
    ):
        self.requests_per_minute = requests_per_minute
        self.max_concurrent = max_concurrent
        self.max_burst = max_burst

        # Token bucket for RPM limiting
        self._tokens = float(max_burst)
        self._last_refill = time.monotonic()
        self._token_lock = threading.Lock()

        # Concurrent request limiting
        self._semaphore = threading.BoundedSemaphore(max_concurrent)

        # Calculate refill rate (tokens per second)
        self._refill_rate = requests_per_minute / 60.0

    def _refill_tokens(self):
        """Add tokens based on time elapsed since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self._refill_rate

        self._tokens = min(float(self.max_burst), self._tokens + tokens_to_add)
        self._last_refill = now

    def _acquire_token(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token from the bucket.

        Returns True if token acquired, False if timeout.
        """
        start_time = time.monotonic()

        while True:
            with self._token_lock:
                self._refill_tokens()

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

                # Calculate wait time for next token
                tokens_needed = 1.0 - self._tokens
                wait_time = tokens_needed / self._refill_rate

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    return False

            # Wait before trying again
            time.sleep(min(wait_time, 0.1))  # Check at least every 100ms

    @contextmanager
    def acquire(self, timeout: Optional[float] = None):
        """
        Context manager to acquire both concurrent slot and RPM token.
        """
        acquired_semaphore = False
        acquired_token = False

        try:
            # STEP 1: Acquire token FIRST (limits rate)
            if not self._acquire_token(timeout=timeout):
                raise TimeoutError("Could not acquire rate limit token")
            acquired_token = True

            # STEP 2: Acquire semaphore SECOND (limits concurrency)
            if not self._semaphore.acquire(timeout=timeout):
                raise TimeoutError("Could not acquire concurrent request slot")
            acquired_semaphore = True

            yield

        finally:
            if acquired_semaphore:
                self._semaphore.release()

    async def acquire_async(self, timeout: Optional[float] = None):
        """
        Async context manager to acquire both concurrent slot and RPM token.
        """
        return _AsyncRateLimiterContext(self, timeout)


class _AsyncRateLimiterContext:
    """Helper class for async context manager protocol."""

    def __init__(self, limiter: TokenBucketRateLimiter, timeout: Optional[float]):
        self.limiter = limiter
        self.timeout = timeout
        self.acquired_semaphore = False
        self.acquired_token = False

    async def __aenter__(self):
        # STEP 1: Acquire token FIRST (limits rate)
        token_acquired = await asyncio.to_thread(
            self.limiter._acquire_token,
            timeout=self.timeout
        )
        if not token_acquired:
            raise TimeoutError("Could not acquire rate limit token")
        self.acquired_token = True

        # STEP 2: Acquire semaphore SECOND (limits concurrency)
        acquired = await asyncio.to_thread(
            self.limiter._semaphore.acquire,
            timeout=self.timeout
        )
        if not acquired:
            raise TimeoutError("Could not acquire concurrent request slot")
        self.acquired_semaphore = True

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.acquired_semaphore:
            self.limiter._semaphore.release()
        return False


# Global rate limiter registry
_rate_limiters: dict[str, TokenBucketRateLimiter] = {}
_registry_lock = threading.Lock()


def get_rate_limiter(provider: str = "default") -> TokenBucketRateLimiter:
    """Get or create a rate limiter for a specific provider using global config."""
    global _rate_limiters

    from adk_rlm.config import get_config
    config = get_config()

    with _registry_lock:
        if provider not in _rate_limiters:
            # Check for provider-specific config
            if provider in config.provider_configs:
                p_config = config.provider_configs[provider]
                rpm = p_config.requests_per_minute
                concurrent = p_config.max_concurrent
                burst = p_config.max_burst
            else:
                # Use default config
                rpm = config.requests_per_minute
                concurrent = config.max_concurrent_requests
                burst = config.max_burst

            _rate_limiters[provider] = TokenBucketRateLimiter(
                requests_per_minute=rpm,
                max_concurrent=concurrent,
                max_burst=burst,
            )
        return _rate_limiters[provider]


def reset_rate_limiters():
    """Reset all rate limiters."""
    global _rate_limiters
    with _registry_lock:
        _rate_limiters = {}


# Convenience functions
@contextmanager
def rate_limit(provider: str = "default", timeout: Optional[float] = None):
    """Context manager using global rate limiter registry."""
    limiter = get_rate_limiter(provider=provider)
    with limiter.acquire(timeout=timeout):
        yield


async def rate_limit_async(provider: str = "default", timeout: Optional[float] = None):
    """Async context manager using global rate limiter registry."""
    limiter = get_rate_limiter(provider=provider)
    return await limiter.acquire_async(timeout=timeout)
