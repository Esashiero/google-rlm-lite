"""
LiteLLM client wrapper with rate limiting and retry logic.
"""

import asyncio
import time
import logging
from typing import Any, List, Optional, Dict

import litellm
from litellm.exceptions import RateLimitError

from adk_rlm.rate_limiter import rate_limit, rate_limit_async

logger = logging.getLogger(__name__)

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

        # Determine provider
        try:
            self.provider = litellm.get_llm_provider(model)[1]
        except Exception:
            self.provider = "default"

    def completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Any:
        """
        Make a completion call with rate limiting and retries.
        """
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        elif system_prompt:
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": system_prompt}] + messages

        last_error = None

        for attempt in range(self.max_retries):
            try:
                with rate_limit(provider=self.provider, timeout=120):
                    try:
                        response = litellm.completion(
                            model=self.model,
                            messages=messages,
                            temperature=temperature,
                            api_key=self.api_key,
                            api_base=self.api_base,
                            **kwargs
                        )
                        return response

                    except RateLimitError as e:
                        last_error = e
                        if attempt < self.max_retries - 1:
                            delay = self.base_retry_delay * (2 ** attempt)
                            logger.warning(f"Rate limit hit for {self.model}, retrying in {delay}s... (attempt {attempt+1})")
                            time.sleep(delay)
                        else:
                            raise
                    except Exception as e:
                        logger.error(f"Unexpected error in LiteLLMClient.completion: {e}")
                        raise
            except TimeoutError as e:
                logger.error(f"Rate limit acquisition timed out: {e}")
                raise

        raise last_error or Exception("Max retries exceeded")

    async def acompletion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Any:
        """
        Make an async completion call with rate limiting and retries.
        """
        from adk_rlm.rate_limiter import get_rate_limiter
        limiter = get_rate_limiter(self.provider)

        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        elif system_prompt:
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": system_prompt}] + messages

        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Acquiring rate limit for {self.provider} - tokens available: {limiter._tokens:.2f}")
                async with await limiter.acquire_async(timeout=120):
                    try:
                        response = await litellm.acompletion(
                            model=self.model,
                            messages=messages,
                            temperature=temperature,
                            api_key=self.api_key,
                            api_base=self.api_base,
                            **kwargs
                        )
                        return response

                    except RateLimitError as e:
                        last_error = e
                        if attempt < self.max_retries - 1:
                            delay = self.base_retry_delay * (2 ** attempt)
                            logger.warning(f"Rate limit hit for {self.model}, retrying in {delay}s... (attempt {attempt+1})")
                            await asyncio.sleep(delay)
                        else:
                            raise
                    except Exception as e:
                        logger.error(f"Unexpected error in LiteLLMClient.acompletion: {e}")
                        raise
            except TimeoutError as e:
                logger.error(f"Rate limit acquisition timed out: {e}")
                raise

        raise last_error or Exception("Max retries exceeded")

    async def acompletion_batched(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        stagger: Optional[float] = None,
        **kwargs
    ) -> list[str]:
        """
        Make batched async completion calls with rate limiting and staggering.
        """
        if stagger is None:
            from adk_rlm.rate_limiter import get_rate_limiter
            limiter = get_rate_limiter(self.provider)
            # Calculate stagger: 60 / RPM gives seconds per request
            # Use a slightly higher value (66 instead of 60) for safety
            stagger = 66.0 / limiter.requests_per_minute
            logger.debug(f"Calculated stagger for {self.provider}: {stagger:.2f}s (RPM: {limiter.requests_per_minute})")

        results = [None] * len(prompts)

        async def query_one(prompt: str, idx: int):
            # Add a small stagger based on index to prevent simultaneous bursts
            if stagger and stagger > 0:
                await asyncio.sleep(idx * stagger)

            try:
                response = await self.acompletion(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    **kwargs
                )
                results[idx] = response.choices[0].message.content
            except Exception as e:
                results[idx] = f"Error: {e}"

        # We still use gather but query_one now has an internal stagger
        tasks = [query_one(p, i) for i, p in enumerate(prompts)]
        await asyncio.gather(*tasks)

        return results
