"""
LiteLLM client wrapper with rate limiting and retry logic.
"""

import asyncio
import time
from typing import Any, List, Optional, Dict

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
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Any:
        """
        Make a completion call with rate limiting and retries.

        Args:
            prompt: User prompt (ignored if messages provided)
            messages: List of message dictionaries
            system_prompt: Optional system prompt (prepended to messages)
            temperature: Sampling temperature
            **kwargs: Additional arguments for litellm

        Returns:
            LiteLLM response object
        """
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        elif system_prompt:
            # Check if there's already a system message
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": system_prompt}] + messages

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
                    return response

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
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Any:
        """
        Make an async completion call with rate limiting and retries.

        Args:
            prompt: User prompt (ignored if messages provided)
            messages: List of message dictionaries
            system_prompt: Optional system prompt (prepended to messages)
            temperature: Sampling temperature
            **kwargs: Additional arguments for litellm

        Returns:
            LiteLLM response object
        """
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        elif system_prompt:
            # Check if there's already a system message
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": system_prompt}] + messages

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
                    return response

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
                    response = await self.acompletion(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        **kwargs
                    )
                    return (idx, response.choices[0].message.content)
                except Exception as e:
                    return (idx, f"Error: {e}")

        # Create tasks for all prompts
        tasks = [query_one(p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

        # Sort by index and return just the results
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
