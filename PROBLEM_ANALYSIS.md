# PROBLEM ANALYSIS: Rate Limit Failures in rlm-google

## Executive Summary

The rlm-google implementation is experiencing **RateLimitError** exceptions when using Mistral's API via LiteLLM. Despite having rate limiting code in place, the system is rapidly exhausting the API provider's rate limits because of fundamental architectural flaws in how rate limiting is implemented.

**Error Pattern Observed:**
```
RateLimitError: MistralException - {"object":"error","message":"Rate limit exceeded","type":"rate_limited","param":null,"code":"1300"}
```

**Impact:**
- LLM calls fail after 16+ seconds of retry attempts
- Recursive RLM execution is severely impacted
- System degrades to the point of unusability
- Each iteration takes 16+ seconds due to failed retries

---

## Detailed Problem Analysis

### 1. Root Cause: Misordered Rate Limiting Logic

**Location:** `adk_rlm/rate_limiter.py` - `TokenBucketRateLimiter.acquire()` method

**Current Implementation (FLAWED):**
```python
async def acquire(self, timeout: Optional[float] = None) -> bool:
    """Acquire a token from the bucket, blocking if necessary."""
    start_time = time.monotonic()
    
    # STEP 1: Acquire semaphore FIRST (limits concurrency)
    if not await self._acquire_semaphore(timeout):
        return False
    
    # STEP 2: Then try to get a token (limits rate)
    while True:
        if await self._try_get_token():
            return True
        
        # Check timeout
        if timeout is not None:
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                self.semaphore.release()  # Release semaphore on timeout
                return False
        
        await asyncio.sleep(0.01)  # Small delay before retry
```

**The Flaw:**
The code acquires the **semaphore BEFORE the token**. This means:
1. Up to 30 tasks can acquire the semaphore simultaneously
2. These 30 tasks then all compete for the 5 burst tokens
3. Tasks without tokens spin in a tight loop (`sleep(0.01)`), consuming CPU
4. When tokens become available, waiting tasks immediately consume them
5. This creates a "thundering herd" problem where requests still fire rapidly

**What Should Happen:**
```
Token Bucket FIRST → Semaphore SECOND
```

The token bucket should throttle the **rate** at which tasks even attempt to run. Only after a task has "paid" for its token should it be allowed to proceed to the concurrency semaphore.

---

### 2. Root Cause: Batched Requests Fire Simultaneously

**Location:** `adk_rlm/litellm_client.py` - `acompletion_batched()` method

**Current Implementation (FLAWED):**
```python
async def acompletion_batched(self, prompts: list[str], ...) -> list[str]:
    semaphore = asyncio.Semaphore(30)
    
    async def query_one(prompt: str, idx: int) -> tuple[int, str]:
        async with semaphore:  # Only limits concurrency
            response = await self.acompletion(...)  # Rate limit checked HERE
            return (idx, response.choices[0].message.content)
    
    # ALL tasks created immediately
    tasks = [query_one(p, i) for i, p in enumerate(prompts)]
    
    # ALL tasks started simultaneously via gather
    results = await asyncio.gather(*tasks)
```

**The Flaw:**
1. `asyncio.gather(*tasks)` starts ALL tasks at once
2. Each task immediately hits the semaphore and waits
3. As slots become available (tasks complete), waiting tasks immediately start
4. The rate limiter inside `self.acompletion()` only sees one task at a time
5. But tasks complete and new ones start so rapidly that the effective rate is much higher than 60 RPM

**Example Timeline (10 prompts, 60 RPM limit):**
```
T+0.0s: All 10 tasks created via list comprehension
T+0.0s: Task 1-5 acquire semaphore, others wait
T+0.0s: Task 1 acquires token, calls API (token bucket: 5→4)
T+0.1s: Task 2 acquires token, calls API (token bucket: 4→3)
T+0.2s: Task 3 acquires token, calls API (token bucket: 3→2)
T+0.3s: Task 4 acquires token, calls API (token bucket: 2→1)
T+0.4s: Task 5 acquires token, calls API (token bucket: 1→0)
T+0.8s: Task 6 acquires token (waited 0.8s for refill), calls API
T+1.2s: Task 7 acquires token, calls API
T+1.6s: Task 8 acquires token, calls API
T+2.0s: Task 9 acquires token, calls API
T+2.4s: Task 10 acquires token, calls API

Result: 10 requests in 2.4 seconds = 250 RPM (WAY over 60 RPM limit!)
```

**The Real Problem:**
Even though each individual request waits for a token, the burst of requests at the start (tasks 1-5 all fire within 0.4s) combined with rapid completion and slot turnover means the **effective request rate is much higher than the configured 60 RPM**.

With Mistral's free tier limit of **~1 request per second**, 10 requests in 2.4 seconds triggers rate limiting immediately.

---

### 3. Root Cause: Recursive Parallel Execution Multiplies the Problem

**Location:** `adk_rlm/code_executor.py` - `_run_parallel_recursive()` method

**Current Implementation:**
```python
def _run_parallel_recursive(
    prompts: list[str],
    parent_context: str,
    model: Optional[str],
    max_workers: int = 4,
) -> list[str]:
    """Run multiple recursive RLM agents in parallel using threading."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_single_recursive, prompt, parent_context, model)
            for prompt in prompts
        ]
        return [f.result() for f in futures]
```

**The Flaw:**
1. `_run_parallel_recursive` spawns multiple **threads** (not just async tasks)
2. Each thread creates its own event loop via `asyncio.new_event_loop()`
3. Each event loop has its own instance of the rate limiter
4. Multiple threads = multiple rate limiters = no global coordination

**Impact:**
- 4 threads × 5 batched requests each = 20 requests firing simultaneously
- Each thread's rate limiter independently allows 5 burst tokens
- Result: 20 requests within milliseconds of each other
- With Mistral's 1 req/sec limit, this immediately triggers rate limiting

---

### 4. Root Cause: No Per-Provider Rate Limit Configuration

**Location:** `adk_rlm/config.py` and hardcoded defaults

**Current Configuration:**
```python
# Default rate limits (gemini tier 1)
DEFAULT_RPM = 60  # Requests per minute
DEFAULT_MAX_CONCURRENT = 30
DEFAULT_MAX_BURST = 5
```

**The Flaw:**
1. Rate limits are **hardcoded to 60 RPM**
2. No provider-specific configuration
3. Mistral's free tier has **much stricter limits** (~1 req/sec = 60 RPM theoretical max, but bursts fail)
4. The code treats all providers the same

**Provider Rate Limits (Examples):**
- **Gemini:** 60 RPM (configurable)
- **Mistral Free Tier:** ~1 request/second with strict burst limits
- **OpenAI:** Tier-dependent, 60-3500 RPM
- **Anthropic:** Tier-dependent, 100-4000 RPM

Using a one-size-fits-all 60 RPM configuration doesn't work when providers have different actual limits.

---

### 5. Root Cause: Inadequate Retry Backoff Strategy

**Location:** `adk_rlm/litellm_client.py` - retry logic

**Current Implementation:**
```python
except RateLimitError as e:
    last_error = e
    if attempt < self.max_retries - 1:
        delay = self.base_retry_delay * (2 ** attempt)  # Exponential: 1, 2, 4, 8, 16s
        await asyncio.sleep(delay)
    else:
        raise
```

**The Flaw:**
1. When rate limits are hit, the retry uses exponential backoff
2. First retry: 1 second
3. Second retry: 2 seconds
4. Third retry: 4 seconds
5. Fourth retry: 8 seconds
6. Fifth retry: 16 seconds

**Impact:**
- With aggressive request firing, most requests hit rate limits
- Each failed request retries multiple times
- Total delay: 1+2+4+8+16 = 31 seconds of waiting
- But during these 31 seconds, OTHER requests are still being made
- This compounds the problem and makes the system unusable

**Observed Behavior:**
```
Iteration 1:
  - LLM Start at T+0s
  - Rate limit error, retries 1, 2, 4, 8s
  - LLM End at T+16.7s (FAILED)

Iteration 2:
  - LLM Start at T+16.7s
  - Rate limit error, retries 1, 2, 4, 8s
  - LLM End at T+32.4s (FAILED)

Total: 2 iterations, both failed, took 48+ seconds
```

---

## Evidence from Logs

### Console Log Timeline:
```
12:03:19 - First request
12:03:21 - Second request (2s later)
12:03:23 - Third request (2s later)
12:03:27 - Fourth request (4s later)
12:03:35 - Fifth request (8s later) ← FIRST RATE LIMIT ERROR
12:03:35 - Sixth request (immediate)
12:03:36 - Seventh request (1s later)
12:03:38 - Eighth request (2s later)
12:03:43 - Ninth request (5s later)
12:03:51 - Tenth request (8s later) ← SECOND RATE LIMIT ERROR
```

**Pattern Analysis:**
- Multiple requests fire rapidly (within seconds)
- Rate limit errors occur after ~8 seconds of rapid firing
- Error occurs at 12:03:35, 12:03:51, 12:04:07 (every ~16 seconds)
- This matches the retry pattern: 1+2+4+8 = 15 seconds of retries before giving up

### JSONL Log Analysis:
```json
{"type": "iteration", "iteration": 1, "timestamp": "2026-02-11T12:03:35.744954", ...}
{"type": "iteration", "iteration": 2, "timestamp": "2026-02-11T12:03:51.406489", ...}
{"type": "iteration", "iteration": 3, "timestamp": "2026-02-11T12:04:07.082898", ...}
```

- 3 iterations attempted
- Each iteration ~16 seconds apart
- All iterations failed due to rate limiting
- This matches the exponential backoff retry delays (1+2+4+8 = 15s)

---

## Summary of Issues

| Issue | Location | Impact | Severity |
|-------|----------|--------|----------|
| Misordered rate limiting | `rate_limiter.py:85-86` | Tasks acquire semaphore before token, allowing rapid-fire requests | **CRITICAL** |
| Batched gather fires all at once | `litellm_client.py:201` | `asyncio.gather(*tasks)` starts all tasks simultaneously | **CRITICAL** |
| Thread-based recursive execution | `code_executor.py:410-418` | Multiple threads = multiple rate limiters = no global coordination | **HIGH** |
| No provider-specific limits | `config.py` | Hardcoded 60 RPM doesn't match Mistral's actual limits | **HIGH** |
| Inadequate retry backoff | `litellm_client.py:156-158` | Exponential backoff doesn't prevent initial burst | **MEDIUM** |

---

## Why Current Rate Limiter Doesn't Work

The current `TokenBucketRateLimiter` has the right **idea** but wrong **implementation**:

**What It Tries to Do:**
- Limit concurrent requests to 30
- Limit rate to 60 RPM (1 token/second)
- Allow burst of 5 requests

**Why It Fails:**
1. **Semaphore before token** means 30 tasks can all be "in flight" at once
2. **Batched execution** via `asyncio.gather` starts all tasks immediately
3. **Thread-based recursion** creates multiple independent rate limiters
4. **No per-provider configuration** uses wrong limits for Mistral

**The Fundamental Problem:**
> Rate limiting must happen **BEFORE** tasks are even created, not **AFTER** they're already running.

The current design allows tasks to be created and started, then tries to throttle them. By that point, it's too late—the API has already been overwhelmed.

---

## Conclusion

The rate limit failures are caused by a **systematic architectural flaw** in how rate limiting is implemented:

1. **Order of operations is wrong:** Semaphore acquired before token
2. **Batch execution is too aggressive:** `asyncio.gather` fires all tasks at once
3. **No global coordination:** Thread-based recursion creates isolated rate limiters
4. **No provider awareness:** Hardcoded limits don't match actual provider limits
5. **Retry strategy inadequate:** Exponential backoff happens AFTER rate limits are hit

**The Result:**
- System attempts to make requests faster than the API allows
- Rate limits are hit repeatedly
- Each failed request retries with exponential backoff
- Total execution time increases dramatically (48+ seconds for 2 failed iterations)
- System becomes effectively unusable with Mistral's strict rate limits

**To Fix:**
Rate limiting must be rearchitected to:
1. Acquire token **BEFORE** semaphore
2. Stagger batch execution (not fire all at once)
3. Use a global rate limiter across all threads
4. Support per-provider rate limit configuration
5. Implement pre-emptive rate limiting (not reactive retries)
