# Rate Limit Debug Instructions for Jules

## Problem Statement
The rate limiting is still failing with Mistral. After running:
```bash
python scripts/test_web_client.py query "What's the best way to get rich through polymarket"
```

**Result**: Both iterations fail with:
```
[LLM query failed: litellm.RateLimitError: RateLimitError: MistralException - {"object":"error","message":"Rate limit exceeded","type":"rate_limited","param":null,"code":"1300"}]
```

**Elapsed time**: 47.83s (mostly wasted on retries)

---

## Root Cause Analysis

### Issue 1: Mistral Free Tier Rate Limits Are Stricter Than Expected

The current config assumes Mistral free tier = 30 RPM (0.5 req/sec):
```python
# adk_rlm/config.py line 52
"mistral": ProviderConfig(requests_per_minute=30, max_burst=1, max_concurrent=2),
```

**But the actual Mistral free tier limit is approximately 1 request per 2 seconds (30 RPM with burst of 1 is too aggressive).**

Evidence:
- Query has 2 iterations
- Both iterations get rate limited
- Total time is 47 seconds (wasted on retries with exponential backoff)

### Issue 2: Configuration Not Being Read from .env

The `.env` file has provider-specific limits commented out:
```bash
# .env line 14
# RLM_PROVIDER_LIMITS='{"mistral": {"requests_per_minute": 30, "max_burst": 1, "max_concurrent": 2}}'
```

But the code uses hardcoded defaults instead of reading from environment.

### Issue 3: Stagger Delay Too Short

In `litellm_client.py` line 131:
```python
stagger: Optional[float] = 0.1,  # 100ms between requests
```

With 30 RPM (0.5 req/sec), 100ms stagger is not enough breathing room.

---

## Required Fixes

### Fix 1: Update Mistral Default Config

**File**: `adk_rlm/config.py`
**Line**: 52

Change from:
```python
"mistral": ProviderConfig(requests_per_minute=30, max_burst=1, max_concurrent=2),
```

To:
```python
"mistral": ProviderConfig(requests_per_minute=20, max_burst=1, max_concurrent=1),
```

**Rationale**: 
- 20 RPM = 1 request every 3 seconds (safer for free tier)
- max_concurrent=1 ensures sequential processing
- max_burst=1 prevents any burst behavior

### Fix 2: Enable Provider-Specific Config from .env

**File**: `adk_rlm/config.py`
**Function**: `get_provider_config()` (lines 55-71)

The function reads from `RLM_PROVIDER_LIMITS` environment variable but it's not working correctly. Verify that:
1. The env var is being read
2. The JSON parsing works
3. The values override hardcoded defaults

**Debug step**: Add logging to see what config is actually being used:
```python
import logging
logger = logging.getLogger(__name__)

# In get_provider_config()
logger.info(f"Using config for {provider}: {config}")
```

### Fix 3: Increase Default Stagger for Batched Requests

**File**: `adk_rlm/litellm_client.py`
**Line**: 131

Change from:
```python
stagger: Optional[float] = 0.1,
```

To:
```python
stagger: Optional[float] = None,  # Will calculate from rate limiter
```

Then in the method, calculate stagger based on RPM:
```python
if stagger is None and rate_limiter:
    # Calculate stagger: 60 / RPM gives seconds per request
    stagger = 60.0 / rate_limiter.requests_per_minute
```

### Fix 4: Verify Rate Limiter is Actually Being Used

**File**: `adk_rlm/litellm_client.py`
**Method**: `acompletion()` (lines 136-177)

Add debug logging to confirm:
1. Rate limiter is acquired before API call
2. Token is available
3. Request is staggered properly

**Debug logging to add**:
```python
logger.debug(f"Acquiring rate limit for {self.provider} - tokens available: {rate_limiter.tokens}")
```

---

## Testing Procedure

After making changes, test with:

```bash
# Terminal 1: Start server with debug logging
export RLM_LOG_LEVEL=DEBUG
python -m adk_rlm.web

# Terminal 2: Run test query
python scripts/test_web_client.py query "What is 2 + 2?"
```

**Expected behavior**:
1. Query should complete in ~6-10 seconds (2 iterations × 3 seconds each)
2. No rate limit errors
3. Both iterations should show "Llm End" successfully
4. Final answer should contain actual content, not error message

**Success criteria**:
- [ ] No `RateLimitError` in logs
- [ ] Total time < 15 seconds for 2 iterations
- [ ] Final answer contains meaningful content
- [ ] Each request is spaced ~3 seconds apart

---

## Debugging Checklist

If it still fails:

1. **Check what rate limit config is actually loaded**:
   ```python
   from adk_rlm.config import get_provider_config
   config = get_provider_config("mistral")
   print(f"Config: {config}")
   ```

2. **Verify rate limiter state**:
   ```python
   from adk_rlm.rate_limiter import get_rate_limiter
   limiter = get_rate_limiter("mistral")
   print(f"Tokens: {limiter.tokens}, RPM: {limiter.requests_per_minute}")
   ```

3. **Enable verbose logging**:
   Add to `adk_rlm/__init__.py`:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Check LiteLLM retry behavior**:
   The error message suggests LiteLLM is retrying. Check if rate limiter is being bypassed on retries.

---

## Files to Modify

1. `adk_rlm/config.py` - Update mistral defaults
2. `adk_rlm/litellm_client.py` - Better stagger calculation
3. `adk_rlm/rate_limiter.py` - Add debug logging (optional)
4. `adk_rlm/agents/rlm_agent.py` - Verify client initialization

---

## Environment Variables

Ensure these are set in `.env`:
```bash
RLM_MODEL=mistral/mistral-large-latest
RLM_MAX_ITERATIONS=2  # Start low for testing
MISTRAL_API_KEY=your_key_here

# Optional: Override provider limits
RLM_PROVIDER_LIMITS='{"mistral": {"requests_per_minute": 20, "max_burst": 1, "max_concurrent": 1}}'
```

---

## Notes

- The current implementation has rate limiting code, but the defaults are too aggressive for Mistral free tier
- The stagger delay of 0.1s is not sufficient - needs to be based on actual RPM limits
- Consider adding a "safe mode" flag that uses very conservative rate limits for testing
- Free tier APIs are often more restrictive than documented - err on the side of caution
