# Google Jules Instructions: Add LiteLLM Support with Proper Rate Limiting to rlm-google

## Executive Summary

The current `rlm-google` project has a **critical rate limiting flaw**: it uses a `BoundedSemaphore(30)` which only limits **concurrent** requests, not **requests per minute (RPM)**. This means 30 requests can be fired simultaneously within 1 second, easily exceeding Gemini's 60rpm limit and causing rate limit errors.

This document provides a complete implementation plan to:
1. Add LiteLLM support to rlm-google (replacing direct Gemini SDK usage)
2. Implement proper token bucket rate limiting for 60rpm enforcement
3. Add retry logic with exponential backoff for rate limit errors

---

## Current State Analysis

### rlm-google (Current Implementation)

**File**: `adk_rlm/llm.py`
```python
# Current rate limiting - ONLY limits concurrency, NOT RPM
LLM_CONCURRENCY_LIMIT = 30
_llm_semaphore = threading.BoundedSemaphore(LLM_CONCURRENCY_LIMIT)
```

**Problem**: 
- 30 concurrent requests can ALL be fired in the same second
- Gemini's 60rpm limit means max 1 request/second
- With current code, you hit 60rpm in ~2 seconds, then get rate limited

**Where rate limiting is used**:
- `adk_rlm/llm.py` - Simple LLM calls use `with llm_rate_limit():`
- `adk_rlm/code_executor.py` line 220, 608 - Both sync and async paths

### rlm (Reference Implementation)

**File**: `rlm/rlm/clients/litellm.py`
- Has LiteLLM client but **NO rate limiting at all**
- Uses `litellm.completion()` and `litellm.acompletion()` directly
- No concurrency or RPM limiting

### Key Insight

The problem occurs in batched queries:
```python
# adk_rlm/code_executor.py ~line 684
tasks = [query_single(client, p, i) for i, p in enumerate(prompts)]
return await asyncio.gather(*tasks)  # ALL fire at once!
```

Even with semaphore, if 30 requests all acquire the semaphore in <1 second, you exceed 60rpm.

---

## Solution Architecture

### 1. Token Bucket Rate Limiter

Implement a proper token bucket algorithm that enforces:
- **RPM limit**: 60 requests per minute = 1 request per second
- **Concurrent limit**: 30 max concurrent (existing)
- **Burst handling**: Allow small bursts but smooth out over time

### 2. Retry with Exponential Backoff

When rate limit errors occur:
- Catch `429 Too Many Requests` errors
- Retry with exponential backoff (1s, 2s, 4s, 8s...)
- Max 5 retries before giving up

### 3. LiteLLM Integration

Replace direct `google-genai` SDK calls with LiteLLM:
- Unified interface for multiple providers
- Built-in model routing
- Easier to support non-Gemini models

---

## Implementation Plan

### Phase 1: Create New Rate Limiter Module

**New File**: `adk_rlm/rate_limiter.py`

```python
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
```

### Phase 2: Create LiteLLM Client Module

**New File**: `adk_rlm/litellm_client.py`

```python
"""
LiteLLM client wrapper with rate limiting and retry logic.
"""

import asyncio
import time
from typing import Any, Optional

import litellm
from litellm.exceptions import RateLimitError

from adk_rlm.rate_limiter import rate_limit, rate_limit_async


class LiteLLMClient:
    """
    LiteLLM client with built-in rate limiting and retry logic.
    
    Supports both sync and async operations with automatic retries
    for rate limit errors.
    """
    
    def __init__(
        self,
        model: str = "gemini/gemini-1.5-flash",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_retries: int = 5,
        base_retry_delay: float = 1.0,
    ):
        """
        Initialize LiteLLM client.
        
        Args:
            model: Model name (e.g., "gemini/gemini-1.5-flash", "openai/gpt-4o")
            api_key: API key (defaults to env var based on provider)
            api_base: Custom API base URL
            max_retries: Max retries for rate limit errors
            base_retry_delay: Initial retry delay in seconds (doubles each retry)
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
    
    def completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Make a completion call with rate limiting and retries.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            **kwargs: Additional arguments for litellm
            
        Returns:
            Response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        last_error = None
        
        for attempt in range(self.max_retries):
            with rate_limit(timeout=60):  # Wait up to 60s for rate limit
                try:
                    response = litellm.completion(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        api_key=self.api_key,
                        api_base=self.api_base,
                        **kwargs
                    )
                    return response.choices[0].message.content
                    
                except RateLimitError as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        delay = self.base_retry_delay * (2 ** attempt)
                        time.sleep(delay)
                    else:
                        raise
                except Exception:
                    raise
        
        # Should not reach here, but just in case
        raise last_error or Exception("Max retries exceeded")
    
    async def acompletion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Make an async completion call with rate limiting and retries.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            **kwargs: Additional arguments for litellm
            
        Returns:
            Response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        last_error = None
        
        for attempt in range(self.max_retries):
            async with await rate_limit_async(timeout=60):
                try:
                    response = await litellm.acompletion(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        api_key=self.api_key,
                        api_base=self.api_base,
                        **kwargs
                    )
                    return response.choices[0].message.content
                    
                except RateLimitError as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        delay = self.base_retry_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                    else:
                        raise
                except Exception:
                    raise
        
        raise last_error or Exception("Max retries exceeded")
    
    async def acompletion_batched(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> list[str]:
        """
        Make batched async completion calls with rate limiting.
        
        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt (applied to all)
            temperature: Sampling temperature
            **kwargs: Additional arguments for litellm
            
        Returns:
            List of response texts in same order as prompts
        """
        semaphore = asyncio.Semaphore(30)  # Limit concurrent within batch
        
        async def query_one(prompt: str, idx: int) -> tuple[int, str]:
            async with semaphore:
                try:
                    result = await self.acompletion(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        **kwargs
                    )
                    return (idx, result)
                except Exception as e:
                    return (idx, f"Error: {e}")
        
        # Create tasks for all prompts
        tasks = [query_one(p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        
        # Sort by index and return just the results
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
```

### Phase 3: Update Dependencies

**File**: `rlm-google/pyproject.toml`

Add to dependencies:
```toml
dependencies = [
    "google-adk>=1.0.0",
    "google-genai>=1.0.0",  # Keep for now, remove after full migration
    "rich>=13.0.0",
    "python-dotenv>=1.0.0",
    "litellm>=1.0.0",  # ADD THIS
]
```

### Phase 4: Modify Code Executor

**File**: `adk_rlm/code_executor.py`

Replace the `_simple_llm_call` method (around line 174) and `_create_llm_query_batched_fn` method (around line 536):

```python
from adk_rlm.litellm_client import LiteLLMClient
from adk_rlm.rate_limiter import get_rate_limiter

# ... in class definition, add:
# Use LiteLLM client instead of direct genai client
_sub_model: str = PrivateAttr(default="gemini/gemini-1.5-flash")

# ... replace _simple_llm_call method:
def _simple_llm_call(
    self,
    prompt: str,
    model: str,
    batch_index: int | None = None,
    batch_size: int | None = None,
) -> str:
    """Make a simple LLM call with rate limiting and retry logic."""
    
    # Emit start event
    self._emit_sub_llm_event(
        RLMEventType.SUB_LLM_START,
        model=model,
        prompt=prompt,
        batch_index=batch_index,
        batch_size=batch_size,
    )
    
    start_time = time.perf_counter()
    error_msg = None
    response_text = None
    
    try:
        # Create LiteLLM client
        client = LiteLLMClient(
            model=model,
            max_retries=5,
            base_retry_delay=1.0,
        )
        
        response_text = client.completion(prompt=prompt)
        
        # Track usage (if available from response)
        # Note: You'll need to update usage tracking based on LiteLLM response format
        
    except Exception as e:
        error_msg = str(e)
        response_text = f"Error: LLM query failed - {e}"
    
    execution_time_ms = (time.perf_counter() - start_time) * 1000
    
    # Emit end event
    self._emit_sub_llm_event(
        RLMEventType.SUB_LLM_END,
        model=model,
        response=response_text if not error_msg else None,
        error=error_msg,
        execution_time_ms=execution_time_ms,
        batch_index=batch_index,
        batch_size=batch_size,
    )
    
    # Log to JSONL
    self._log_simple_llm_call(
        prompt=prompt,
        response=response_text,
        model=model,
        execution_time_ms=execution_time_ms,
        batch_index=batch_index,
        batch_size=batch_size,
        error=error_msg,
    )
    
    return response_text

# ... replace async batch handling in _create_llm_query_batched_fn:
async def run_all():
    """Run all queries with proper rate limiting."""
    client = LiteLLMClient(
        model=target_model,
        max_retries=5,
        base_retry_delay=1.0,
    )
    
    # Use the batched completion which handles rate limiting internally
    return await client.acompletion_batched(
        prompts=prompts,
        temperature=0.7,
    )

# Run with proper event loop handling
try:
    asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, run_all())
        return future.result()
except RuntimeError:
    return asyncio.run(run_all())
```

### Phase 5: Configuration Support

**New File**: `adk_rlm/config.py`

```python
"""Configuration management for rlm-google."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class RLMConfig:
    """Configuration for RLM operations."""
    
    # Model settings
    default_model: str = "gemini/gemini-1.5-flash"
    
    # Rate limiting
    requests_per_minute: int = 60
    max_concurrent_requests: int = 30
    max_burst: int = 5
    
    # Retry settings
    max_retries: int = 5
    base_retry_delay: float = 1.0
    
    # API Keys (loaded from env vars)
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if self.gemini_api_key is None:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    @classmethod
    def from_env(cls) -> "RLMConfig":
        """Create config from environment variables."""
        return cls(
            default_model=os.getenv("RLM_DEFAULT_MODEL", "gemini/gemini-1.5-flash"),
            requests_per_minute=int(os.getenv("RLM_RPM", "60")),
            max_concurrent_requests=int(os.getenv("RLM_MAX_CONCURRENT", "30")),
            max_burst=int(os.getenv("RLM_MAX_BURST", "5")),
            max_retries=int(os.getenv("RLM_MAX_RETRIES", "5")),
            base_retry_delay=float(os.getenv("RLM_RETRY_DELAY", "1.0")),
        )


# Global config instance
_config: Optional[RLMConfig] = None


def get_config() -> RLMConfig:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = RLMConfig.from_env()
    return _config


def set_config(config: RLMConfig):
    """Set global config instance."""
    global _config
    _config = config
```

---

## Testing the Implementation

### Test File: `tests/test_rate_limiting.py`

```python
"""Tests for rate limiting functionality."""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch

from adk_rlm.rate_limiter import TokenBucketRateLimiter, get_rate_limiter, reset_rate_limiter


class TestTokenBucketRateLimiter:
    """Test token bucket rate limiter."""
    
    def test_initial_state(self):
        """Test initial token bucket state."""
        limiter = TokenBucketRateLimiter(
            requests_per_minute=60,
            max_concurrent=30,
            max_burst=5
        )
        assert limiter._tokens == 5
        assert limiter.max_concurrent == 30
    
    def test_token_refill(self):
        """Test token refill over time."""
        limiter = TokenBucketRateLimiter(
            requests_per_minute=60,  # 1 token per second
            max_concurrent=30,
            max_burst=1  # Start with 1 token
        )
        
        # Consume the token
        limiter._tokens = 0
        initial_time = time.monotonic()
        limiter._last_refill = initial_time
        
        # Wait a bit
        time.sleep(0.1)
        
        # Refill should add tokens
        limiter._refill_tokens()
        assert limiter._tokens > 0
        assert limiter._tokens <= limiter.max_burst
    
    def test_rate_limit_enforcement(self):
        """Test that rate limit is enforced."""
        limiter = TokenBucketRateLimiter(
            requests_per_minute=60,  # 1 token per second
            max_concurrent=30,
            max_burst=1
        )
        
        # First acquisition should succeed
        assert limiter._acquire_token(timeout=1) is True
        
        # Second acquisition should fail (no tokens left)
        assert limiter._acquire_token(timeout=0.1) is False
    
    def test_concurrent_limit(self):
        """Test concurrent request limit."""
        limiter = TokenBucketRateLimiter(
            requests_per_minute=6000,  # Very high RPM
            max_concurrent=2,
            max_burst=100
        )
        
        acquired = []
        
        def acquire():
            with limiter.acquire(timeout=1):
                acquired.append(1)
                time.sleep(0.1)  # Hold for 100ms
        
        # Start 3 threads trying to acquire
        import threading
        threads = [threading.Thread(target=acquire) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All 3 should succeed (2 concurrent + 1 queued)
        assert len(acquired) == 3


class TestRateLimiterIntegration:
    """Test rate limiter integration."""
    
    def test_global_rate_limiter_singleton(self):
        """Test that global rate limiter is a singleton."""
        reset_rate_limiter()
        
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        
        assert limiter1 is limiter2
    
    @pytest.mark.asyncio
    async def test_async_rate_limiting(self):
        """Test async rate limiting."""
        limiter = TokenBucketRateLimiter(
            requests_per_minute=60,
            max_concurrent=30,
            max_burst=5
        )
        
        results = []
        
        async def task(idx):
            async with await limiter.acquire_async():
                results.append(idx)
                await asyncio.sleep(0.01)
        
        # Run 10 tasks
        await asyncio.gather(*[task(i) for i in range(10)])
        
        assert len(results) == 10


class TestRetryLogic:
    """Test retry logic for rate limit errors."""
    
    @patch('adk_rlm.litellm_client.litellm.completion')
    def test_retry_on_rate_limit(self, mock_completion):
        """Test that retries happen on rate limit errors."""
        from litellm.exceptions import RateLimitError
        from adk_rlm.litellm_client import LiteLLMClient
        
        # Fail twice, then succeed
        mock_completion.side_effect = [
            RateLimitError("Rate limit exceeded", model="test", llm_provider="test"),
            RateLimitError("Rate limit exceeded", model="test", llm_provider="test"),
            Mock(choices=[Mock(message=Mock(content="Success"))])
        ]
        
        client = LiteLLMClient(
            model="gemini/test",
            max_retries=3,
            base_retry_delay=0.01  # Fast for testing
        )
        
        result = client.completion(prompt="Test")
        
        assert result == "Success"
        assert mock_completion.call_count == 3
    
    @patch('adk_rlm.litellm_client.litellm.completion')
    def test_max_retries_exceeded(self, mock_completion):
        """Test that max retries raises exception."""
        from litellm.exceptions import RateLimitError
        from adk_rlm.litellm_client import LiteLLMClient
        
        mock_completion.side_effect = RateLimitError(
            "Rate limit exceeded", model="test", llm_provider="test"
        )
        
        client = LiteLLMClient(
            model="gemini/test",
            max_retries=2,
            base_retry_delay=0.01
        )
        
        with pytest.raises(RateLimitError):
            client.completion(prompt="Test")
        
        assert mock_completion.call_count == 2
```

---

## Environment Variables

Add to `.env` file or environment:

```bash
# Model Configuration
RLM_DEFAULT_MODEL=gemini/gemini-1.5-flash

# Rate Limiting (adjust based on your API tier)
# Gemini free tier: 60 RPM
# Gemini paid tier: 1000+ RPM
RLM_RPM=60
RLM_MAX_CONCURRENT=30
RLM_MAX_BURST=5

# Retry Configuration
RLM_MAX_RETRIES=5
RLM_RETRY_DELAY=1.0

# API Keys
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # If using OpenAI through LiteLLM
ANTHROPIC_API_KEY=your_key_here  # If using Anthropic through LiteLLM
```

---

## Migration Checklist

- [ ] Create `adk_rlm/rate_limiter.py` with token bucket implementation
- [ ] Create `adk_rlm/litellm_client.py` with LiteLLM wrapper
- [ ] Create `adk_rlm/config.py` for configuration management
- [ ] Update `pyproject.toml` to add `litellm>=1.0.0` dependency
- [ ] Modify `adk_rlm/code_executor.py` to use new LiteLLM client
- [ ] Update `adk_rlm/llm.py` to use new rate limiter (or deprecate)
- [ ] Add tests in `tests/test_rate_limiting.py`
- [ ] Update environment variables in `.env`
- [ ] Test with batched queries to verify rate limiting works
- [ ] Test retry logic by temporarily lowering RPM limit

---

## Key Changes Summary

| Component | Before | After |
|-----------|--------|-------|
| **Rate Limiting** | Only concurrent (30 max), no RPM limit | Token bucket: 60rpm + 30 concurrent |
| **API Client** | Direct `google-genai` SDK | LiteLLM with retry logic |
| **Batched Queries** | Fire all at once with `asyncio.gather` | Rate-limited with semaphore + token bucket |
| **Error Handling** | No retry logic | Exponential backoff retry (5 attempts) |
| **Model Support** | Only Gemini | Any LiteLLM-supported model |

---

## Why This Solution Works

1. **Token Bucket Algorithm**: Smoothly enforces 60rpm by allowing bursts but averaging to 1 request/second over time
2. **Dual Limiting**: Both concurrent (30) and RPM (60) limits prevent overload at multiple levels
3. **Retry Logic**: Automatically handles transient rate limit errors with exponential backoff
4. **LiteLLM**: Unified interface supports multiple providers without code changes
5. **Backwards Compatible**: Can still use direct Gemini SDK alongside LiteLLM during migration

---

## Expected Behavior After Implementation

**Before** (60 requests in 2 seconds):
```
Request 1-30: Start simultaneously -> Rate limited immediately
Result: Most fail with 429 errors
```

**After** (60 requests spread over 60 seconds):
```
Request 1-5: Start immediately (burst allowance)
Request 6: Wait ~1s
Request 7: Wait ~2s
...
Request 60: Wait ~55s
Result: All succeed, no rate limit errors
```

---

## Troubleshooting

### Issue: Still getting rate limit errors
- Check `RLM_RPM` matches your API tier (free=60, paid=1000+)
- Verify rate limiter is being used (check logs)
- Ensure only one instance of the rate limiter exists (singleton pattern)

### Issue: Too slow
- Increase `RLM_RPM` if you have higher tier
- Increase `RLM_MAX_BURST` for more initial throughput
- Check if you're hitting concurrent limit vs RPM limit

### Issue: Retry not working
- Verify `litellm` is catching the right exception type
- Check that `base_retry_delay` isn't too high
- Ensure max_retries > 0

---

## References

- LiteLLM Documentation: https://docs.litellm.ai/
- Token Bucket Algorithm: https://en.wikipedia.org/wiki/Token_bucket
- Gemini API Rate Limits: https://ai.google.dev/gemini-api/docs/rate-limits
