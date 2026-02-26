"""Tests for rate limiting functionality."""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch

from adk_rlm.rate_limiter import TokenBucketRateLimiter, get_rate_limiter, reset_rate_limiters


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
        reset_rate_limiters()

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

        assert result.choices[0].message.content == "Success"
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

class TestProviderSpecificLimits:
    """Test provider-specific rate limit support."""

    def test_provider_registry(self):
        """Test that different providers get different limiters."""
        reset_rate_limiters()

        limiter1 = get_rate_limiter("gemini")
        limiter2 = get_rate_limiter("mistral")

        assert limiter1 is not limiter2
        assert limiter1.requests_per_minute == 60
        assert limiter2.requests_per_minute == 30

    def test_default_provider(self):
        """Test that unknown provider gets default limits."""
        reset_rate_limiters()

        limiter = get_rate_limiter("unknown")
        assert limiter.requests_per_minute == 60
