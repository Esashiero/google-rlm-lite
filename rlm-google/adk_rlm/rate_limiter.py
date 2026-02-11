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

    For Gemini's 60rpm limit:
    - tokens_per_minute = 60
    - refill_rate = 60/60 = 1 token per second
    - max_burst = 5 (allow small bursts)
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
        self._tokens = max_burst
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

        self._tokens = min(self.max_burst, self._tokens + tokens_to_add)
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

                if self._tokens >= 1:
                    self._tokens -= 1
                    return True

                # Calculate wait time for next token
                tokens_needed = 1 - self._tokens
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

        Usage:
            with rate_limiter.acquire():
                # Make LLM call
                response = client.completion(...)
        """
        acquired_semaphore = False
        acquired_token = False

        try:
            # First acquire semaphore (concurrent limit)
            if not self._semaphore.acquire(timeout=timeout):
                raise TimeoutError("Could not acquire concurrent request slot")
            acquired_semaphore = True

            # Then acquire token (RPM limit)
            if not self._acquire_token(timeout=timeout):
                raise TimeoutError("Could not acquire rate limit token")
            acquired_token = True

            yield

        finally:
            # Release in reverse order
            if acquired_token:
                # Token is consumed, don't add back
                pass
            if acquired_semaphore:
                self._semaphore.release()

    async def acquire_async(self, timeout: Optional[float] = None):
        """
        Async context manager to acquire both concurrent slot and RPM token.

        Usage:
            async with rate_limiter.acquire_async():
                # Make async LLM call
                response = await client.acompletion(...)
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
        # Acquire semaphore in thread pool
        acquired = await asyncio.to_thread(
            self.limiter._semaphore.acquire,
            timeout=self.timeout
        )
        if not acquired:
            raise TimeoutError("Could not acquire concurrent request slot")
        self.acquired_semaphore = True

        # Acquire token in thread pool
        token_acquired = await asyncio.to_thread(
            self.limiter._acquire_token,
            timeout=self.timeout
        )
        if not token_acquired:
            # Release semaphore if token acquisition failed
            self.limiter._semaphore.release()
            self.acquired_semaphore = False
            raise TimeoutError("Could not acquire rate limit token")
        self.acquired_token = True

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.acquired_semaphore:
            self.limiter._semaphore.release()
        return False


# Global rate limiter instance
# 60 RPM for Gemini free tier, 30 concurrent, allow burst of 5
_default_rate_limiter: Optional[TokenBucketRateLimiter] = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter(
    requests_per_minute: int = 60,
    max_concurrent: int = 30,
    max_burst: int = 5,
) -> TokenBucketRateLimiter:
    """Get or create the global rate limiter instance."""
    global _default_rate_limiter

    with _rate_limiter_lock:
        if _default_rate_limiter is None:
            _default_rate_limiter = TokenBucketRateLimiter(
                requests_per_minute=requests_per_minute,
                max_concurrent=max_concurrent,
                max_burst=max_burst,
            )
        return _default_rate_limiter


def reset_rate_limiter():
    """Reset the global rate limiter (useful for testing)."""
    global _default_rate_limiter
    with _rate_limiter_lock:
        _default_rate_limiter = None


# Convenience functions using global limiter
@contextmanager
def rate_limit(timeout: Optional[float] = None):
    """Context manager using global rate limiter."""
    limiter = get_rate_limiter()
    with limiter.acquire(timeout=timeout):
        yield


async def rate_limit_async(timeout: Optional[float] = None):
    """Async context manager using global rate limiter."""
    limiter = get_rate_limiter()
    return await limiter.acquire_async(timeout=timeout)
