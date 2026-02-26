"""
RLM Code Executor using ADK's BaseCodeExecutor.

This module provides a custom code executor that wraps LocalREPL
and provides llm_query() and FINAL() functions for the RLM pattern.
"""

import asyncio
import concurrent.futures
import logging
from queue import Empty
from queue import Queue
import threading
import time
from typing import Any
from typing import AsyncGenerator
from typing import TYPE_CHECKING
import uuid

logger = logging.getLogger(__name__)
from adk_rlm.events import RLMEventData
from adk_rlm.events import RLMEventType
from adk_rlm.litellm_client import LiteLLMClient
from adk_rlm.repl.local_repl import LocalREPL
from adk_rlm.usage import UsageTracker
from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.events.event import Event
from pydantic import PrivateAttr

if TYPE_CHECKING:
  from adk_rlm.logging.rlm_logger import RLMLogger


class RLMCodeExecutor(BaseCodeExecutor):
  """
  Code executor that provides llm_query() and FINAL() functions.
  """

  stateful: bool = True  # Persist namespace across code blocks

  # Use ```repl delimiter instead of ```python
  code_block_delimiters: list[tuple[str, str]] = [
      ("```repl\n", "\n```"),
  ]

  # Private attributes (not part of the Pydantic schema)
  _sub_model: str = PrivateAttr(default="gemini/gemini-1.5-flash")
  _current_depth: int = PrivateAttr(default=0)
  _max_depth: int = PrivateAttr(default=5)
  _max_iterations: int = PrivateAttr(default=30)
  _repl: LocalREPL | None = PrivateAttr(default=None)
  _final_answer: str | None = PrivateAttr(default=None)
  _usage_tracker: UsageTracker = PrivateAttr(default_factory=UsageTracker)
  _logger: "RLMLogger | None" = PrivateAttr(default=None)
  _parent_agent: str | None = PrivateAttr(default=None)
  _current_iteration: int = PrivateAttr(default=0)
  _current_block_index: int = PrivateAttr(default=0)

  # Real-time event streaming via thread-safe queue
  _event_queue: Queue = PrivateAttr(default_factory=Queue)
  _execution_complete: threading.Event = PrivateAttr(
      default_factory=threading.Event
  )

  # Ancestry tracking for nested agents
  _ancestry: list[dict] = PrivateAttr(default_factory=list)

  # Counter for unique child agent names
  _child_agent_counter: int = PrivateAttr(default=0)

  def __init__(
      self,
      sub_model: str = "gemini/gemini-1.5-flash",
      current_depth: int = 0,
      max_depth: int = 5,
      max_iterations: int = 30,
      usage_tracker: UsageTracker | None = None,
      logger: "RLMLogger | None" = None,
      parent_agent: str | None = None,
      ancestry: list[dict] | None = None,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self._sub_model = sub_model
    self._current_depth = current_depth
    self._max_depth = max_depth
    self._max_iterations = max_iterations
    self._repl = None
    self._final_answer = None
    self._usage_tracker = usage_tracker or UsageTracker()
    self._logger = logger
    self._parent_agent = parent_agent
    self._ancestry = ancestry.copy() if ancestry else []

    # Initialize queue and threading event
    self._event_queue = Queue()
    self._execution_complete = threading.Event()

  def _create_llm_query_fn(self):
    def llm_query(
        prompt: str,
        context: Any = None,
        model: str | None = None,
        recursive: bool = True,
    ) -> str:
      target_model = model or self._sub_model
      can_recurse = recursive and (self._current_depth < self._max_depth)

      if can_recurse:
        return self._run_recursive_rlm(
            prompt, target_model, context_obj=context
        )
      else:
        return self._simple_llm_call(prompt, target_model)

    return llm_query

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

        response = client.completion(prompt=prompt)
        response_text = response.choices[0].message.content

        # Track usage
        if hasattr(response, "usage") and response.usage:
            self._usage_tracker.add(
                model,
                input_tokens=getattr(response.usage, "prompt_tokens", 0),
                output_tokens=getattr(response.usage, "completion_tokens", 0)
            )

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

  def _get_current_ancestry_entry(self) -> dict:
    return {
        "agent": self._parent_agent,
        "depth": self._current_depth,
        "iteration": self._current_iteration,
        "block_index": self._current_block_index,
    }

  def _emit_sub_llm_event(
      self,
      event_type: RLMEventType,
      model: str,
      prompt: str | None = None,
      response: str | None = None,
      error: str | None = None,
      execution_time_ms: float | None = None,
      batch_index: int | None = None,
      batch_size: int | None = None,
  ) -> None:
    event_data = RLMEventData(
        event_type=event_type,
        model=model,
        prompt_preview=prompt[:200] if prompt else None,
        response_preview=response[:500] if response else None,
        response_full=response,
        error=error,
        execution_time_ms=execution_time_ms,
        iteration=self._current_iteration,
        block_index=self._current_block_index,
        batch_index=batch_index,
        batch_size=batch_size,
        metadata={"recursive": False},
    )

    metadata = event_data.to_dict()
    metadata["agent_name"] = self._parent_agent
    metadata["agent_depth"] = self._current_depth
    metadata["ancestry"] = self._ancestry + [self._get_current_ancestry_entry()]

    event = Event(
        invocation_id=str(uuid.uuid4()),
        author=self._parent_agent or "code_executor",
        custom_metadata=metadata,
    )

    self._event_queue.put(event)

  def _log_simple_llm_call(
      self,
      prompt: str,
      response: str,
      model: str,
      execution_time_ms: float,
      batch_index: int | None = None,
      batch_size: int | None = None,
      error: str | None = None,
  ) -> None:
    if self._logger is None:
      return

    self._logger.log_simple_llm_call(
        prompt=prompt,
        response=response,
        model=model,
        execution_time_ms=execution_time_ms,
        depth=self._current_depth,
        agent_name=self._parent_agent,
        parent_iteration=self._current_iteration,
        parent_block_index=self._current_block_index,
        batch_index=batch_index,
        batch_size=batch_size,
        error=error,
    )

  def _run_recursive_rlm(
      self,
      prompt: str,
      model: str,
      context_obj: Any = None,
      parallel_batch_id: str | None = None,
      batch_index: int | None = None,
      batch_size: int | None = None,
  ) -> str:
    next_depth = self._current_depth + 1
    child_index = self._child_agent_counter
    self._child_agent_counter += 1
    nested_agent_name = f"rlm_agent_depth_{next_depth}_{child_index}"
    child_ancestry = self._ancestry + [self._get_current_ancestry_entry()]
    event_queue = self._event_queue
    child_context = context_obj

    async def run_nested_async():
      import uuid
      from adk_rlm.agents.rlm_agent import RLMAgent
      from google.adk.agents.invocation_context import InvocationContext
      from google.adk.sessions import InMemorySessionService
      from google.adk.sessions import Session

      nested_agent = RLMAgent(
          name=nested_agent_name,
          model=model,
          sub_model=self._sub_model,
          max_iterations=self._max_iterations,
          max_depth=self._max_depth,
          current_depth=next_depth,
          logger=self._logger,
          parent_agent=self._parent_agent,
          ancestry=child_ancestry,
          verbose=False,
      )

      rlm_context = (
          child_context if child_context is not None else {"query": prompt}
      )

      mock_session = Session(
          id=str(uuid.uuid4()),
          app_name="adk_rlm",
          user_id="default_user",
          state={
              "rlm_context": rlm_context,
              "rlm_prompt": prompt,
          },
      )
      mock_session_service = InMemorySessionService()
      mock_ctx = InvocationContext(
          invocation_id=str(uuid.uuid4()),
          session=mock_session,
          session_service=mock_session_service,
          agent=nested_agent,
      )

      final_answer = None

      try:
        async for event in nested_agent._run_async_impl(mock_ctx):
          if event.custom_metadata and "ancestry" not in event.custom_metadata:
            event.custom_metadata["ancestry"] = child_ancestry
            event.custom_metadata["agent_name"] = nested_agent_name
            event.custom_metadata["agent_depth"] = next_depth
            event.custom_metadata["parent_agent"] = self._parent_agent
            event.custom_metadata["parent_iteration"] = self._current_iteration
            event.custom_metadata["parent_block_index"] = (
                self._current_block_index
            )
            if parallel_batch_id is not None:
              event.custom_metadata["parallel_batch_id"] = parallel_batch_id
              event.custom_metadata["batch_index"] = batch_index
              event.custom_metadata["batch_size"] = batch_size

          event_queue.put(event)

          if event.custom_metadata:
            from adk_rlm.events import RLMEventType
            event_type = event.custom_metadata.get("event_type")
            if event_type == RLMEventType.FINAL_ANSWER.value:
              final_answer = event.custom_metadata.get("answer")

        self._usage_tracker.merge(nested_agent._usage_tracker)
      finally:
        pass

      return final_answer

    try:
      try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
          future = pool.submit(asyncio.run, run_nested_async())
          final_answer = future.result()
      except RuntimeError:
        final_answer = asyncio.run(run_nested_async())

      if final_answer is None:
        return "[Recursive RLM returned no result]"
      return final_answer

    except Exception as e:
      return (
          f"[Recursive RLM at depth {next_depth} failed: {e}]\n"
          + self._simple_llm_call(prompt, model)
      )

  def _create_llm_query_batched_fn(self):
    def llm_query_batched(
        prompts: list[str],
        contexts: list[Any] | None = None,
        model: str | None = None,
        recursive: bool = False,
    ) -> list[str]:
      if contexts is not None and len(contexts) != len(prompts):
        raise ValueError(
            f"contexts length ({len(contexts)}) must match prompts length"
            f" ({len(prompts)})"
        )

      target_model = model or self._sub_model

      if recursive and self._current_depth < self._max_depth:
        return self._run_parallel_recursive(prompts, contexts, target_model)

      # Restore event emission and logging for batched calls
      batch_size = len(prompts)

      async def query_single_async(prompt: str, idx: int) -> str:
          # Use _simple_llm_call but wrap it in a thread if called from async
          # Actually, _simple_llm_call is sync. We should make an async version
          # or just call it in a thread.
          # But we want to use the stagger.

          # Add stagger
          stagger = 0.2 # 5 requests per second
          await asyncio.sleep(idx * stagger)

          return await asyncio.to_thread(
              self._simple_llm_call,
              prompt,
              target_model,
              batch_index=idx,
              batch_size=batch_size
          )

      async def run_all():
          tasks = [query_single_async(p, i) for i, p in enumerate(prompts)]
          return await asyncio.gather(*tasks)

      try:
          asyncio.get_running_loop()
          with concurrent.futures.ThreadPoolExecutor() as pool:
              future = pool.submit(asyncio.run, run_all())
              return future.result()
      except RuntimeError:
          return asyncio.run(run_all())

    return llm_query_batched

  def _run_parallel_recursive(
      self,
      prompts: list[str],
      contexts: list[Any] | None,
      model: str,
  ) -> list[str]:
    contexts = contexts or [None] * len(prompts)
    batch_id = str(uuid.uuid4())
    batch_size = len(prompts)

    def run_one(idx: int) -> tuple[int, str]:
      prompt = prompts[idx]
      context = contexts[idx]
      # Add small stagger for recursive calls too
      time.sleep(idx * 0.5)
      try:
        result = self._run_recursive_rlm(
            prompt,
            model,
            context_obj=context,
            parallel_batch_id=batch_id,
            batch_index=idx,
            batch_size=batch_size,
        )
        return (idx, result)
      except Exception as e:
        return (idx, f"[Error in batch item {idx}: {e}]")

    results = [None] * len(prompts)

    with concurrent.futures.ThreadPoolExecutor() as pool:
      futures = [pool.submit(run_one, i) for i in range(len(prompts))]

      for future in concurrent.futures.as_completed(futures):
        try:
          idx, result = future.result()
          results[idx] = result
        except Exception:
          pass

    for i, result in enumerate(results):
      if result is None:
        results[i] = f"[Error: batch item {i} returned no result]"

    return results

  def _ensure_repl(self) -> LocalREPL:
    if self._repl is None:
      self._repl = LocalREPL(
          llm_query_fn=self._create_llm_query_fn(),
          llm_query_batched_fn=self._create_llm_query_batched_fn(),
      )
    return self._repl

  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    repl = self._ensure_repl()
    result = repl.execute_code(code_execution_input.code)
    if "FINAL_ANSWER" in repl.locals:
      self._final_answer = str(repl.locals["FINAL_ANSWER"])

    return CodeExecutionResult(
        stdout=result.stdout,
        stderr=result.stderr,
        output_files=[],
    )

  def reset_event_state(self) -> None:
    self._event_queue = Queue()
    self._execution_complete.clear()

  async def execute_code_async(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    result = await asyncio.to_thread(
        self._execute_code_with_completion,
        invocation_context,
        code_execution_input,
    )
    return result

  def _execute_code_with_completion(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    try:
      return self.execute_code(invocation_context, code_execution_input)
    finally:
      self._execution_complete.set()

  async def poll_child_events(self) -> AsyncGenerator[Event, None]:
    while (
        not self._execution_complete.is_set() or not self._event_queue.empty()
    ):
      try:
        event = self._event_queue.get_nowait()
        yield event
      except Empty:
        await asyncio.sleep(0.01)

  def load_context(self, context_payload: dict | list | str) -> None:
    repl = self._ensure_repl()
    repl.load_context(context_payload)

  def add_context(self, context_payload: dict | list | str) -> int:
    repl = self._ensure_repl()
    return repl.add_context(context_payload)

  def get_context_count(self) -> int:
    if self._repl is None:
      return 0
    return self._repl.get_context_count()

  def get_history_count(self) -> int:
    if self._repl is None:
      return 0
    return self._repl.get_history_count()

  def add_history(self, message_history: list[dict[str, Any]]) -> int:
    repl = self._ensure_repl()
    return repl.add_history(message_history)

  @property
  def final_answer(self) -> str | None:
    return self._final_answer

  def reset_final_answer(self) -> None:
    self._final_answer = None

  @property
  def locals(self) -> dict[str, Any]:
    if self._repl is None:
      return {}
    return self._repl.locals

  @property
  def usage_tracker(self) -> UsageTracker:
    return self._usage_tracker

  def set_iteration_context(self, iteration: int, block_index: int) -> None:
    self._current_iteration = iteration
    self._current_block_index = block_index

  def pop_child_events(self) -> list:
    events = []
    while not self._event_queue.empty():
      try:
        events.append(self._event_queue.get_nowait())
      except Empty:
        break
    return events

  def cleanup(self) -> None:
    if self._repl:
      self._repl.cleanup()
      self._repl = None
    self._final_answer = None
    while not self._event_queue.empty():
      try:
        self._event_queue.get_nowait()
      except Empty:
        break
    self._execution_complete.clear()
