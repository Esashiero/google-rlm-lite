"""
LLM utilities and rate limiting.

This module provides a global rate limiter for limiting LLM calls
across all RLM components (agents, code executors, batched queries).
"""

import asyncio
from contextlib import contextmanager
from adk_rlm.rate_limiter import get_rate_limiter, rate_limit, rate_limit_async

# Re-exporting rate limiting functions for backwards compatibility
# The underlying implementation now uses TokenBucketRateLimiter

@contextmanager
def llm_rate_limit():
    """Context manager for rate-limiting LLM calls (sync version).

    Usage:
        with llm_rate_limit():
            response = client.models.generate_content(...)
    """
    with rate_limit():
        yield


async def llm_rate_limit_async():
    """Async context manager for rate-limiting LLM calls.

    Usage:
        async with llm_rate_limit_async():
            response = await client.aio.models.generate_content(...)
    """
    return await rate_limit_async()


class AsyncLLMRateLimiter:
    """Async context manager for rate-limiting LLM calls.

    Usage:
        async with AsyncLLMRateLimiter():
            response = await client.aio.models.generate_content(...)
    """

    async def __aenter__(self):
        self._ctx = await rate_limit_async()
        await self._ctx.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._ctx.__aexit__(exc_type, exc_val, exc_tb)
        return False
