"""
Tests for non-recursive (simple) LLM call events and logging.

These tests verify that when llm_query() or llm_query_batched() is called with
recursive=False, proper events are emitted and calls are logged.
"""

import json
from unittest.mock import MagicMock
from unittest.mock import patch

from adk_rlm.code_executor import RLMCodeExecutor
from adk_rlm.events import RLMEventType
from adk_rlm.logging.rlm_logger import RLMLogger
import pytest


class TestLoggerSimpleLLMCall:
  """Tests for RLMLogger.log_simple_llm_call method."""

  def test_log_simple_llm_call_success(self, temp_log_dir):
    """Log a successful simple LLM call."""
    logger = RLMLogger(temp_log_dir)
    logger.log_simple_llm_call(
        prompt="What is 2+2?",
        response="The answer is 4.",
        model="gemini/gemini-1.5-flash",
        execution_time_ms=150.5,
        depth=0,
        agent_name="rlm_agent",
    )

    with open(logger.get_log_path()) as f:
      entry = json.loads(f.readline())

    assert entry["type"] == "simple_llm_call"
    assert entry["prompt"] == "What is 2+2?"
    assert entry["response"] == "The answer is 4."
    assert entry["model"] == "gemini/gemini-1.5-flash"
    assert entry["execution_time_ms"] == 150.5
    assert entry["depth"] == 0
    assert entry["agent_name"] == "rlm_agent"
    assert entry["recursive"] is False
    assert entry["success"] is True
    assert "error" not in entry

  def test_log_simple_llm_call_failure(self, temp_log_dir):
    """Log a failed simple LLM call."""
    logger = RLMLogger(temp_log_dir)
    logger.log_simple_llm_call(
        prompt="What is 2+2?",
        response="Error: LLM query failed - Connection timeout",
        model="gemini/gemini-1.5-flash",
        execution_time_ms=5000.0,
        error="Connection timeout",
    )

    with open(logger.get_log_path()) as f:
      entry = json.loads(f.readline())

    assert entry["type"] == "simple_llm_call"
    assert entry["success"] is False
    assert entry["error"] == "Connection timeout"
    assert "Error:" in entry["response"]


class TestCodeExecutorEmitSubLLMEvent:
  """Tests for RLMCodeExecutor._emit_sub_llm_event method."""

  def test_emit_sub_llm_start_event(self):
    """Emit SUB_LLM_START event."""
    executor = RLMCodeExecutor(
        sub_model="gemini/gemini-1.5-flash",
        current_depth=0,
        max_depth=5,
        parent_agent="rlm_agent",
    )
    executor._current_iteration = 2
    executor._current_block_index = 1

    executor._emit_sub_llm_event(
        RLMEventType.SUB_LLM_START,
        model="gemini/gemini-1.5-flash",
        prompt="Test prompt",
    )

    # Check event was queued
    assert not executor._event_queue.empty()
    event = executor._event_queue.get()

    metadata = event.custom_metadata
    assert metadata["event_type"] == RLMEventType.SUB_LLM_START.value
    assert metadata["model"] == "gemini/gemini-1.5-flash"
    assert metadata["prompt_preview"] == "Test prompt"
    assert metadata["iteration"] == 2
    assert metadata["block_index"] == 1
    assert metadata["agent_name"] == "rlm_agent"
    assert metadata["agent_depth"] == 0
    assert metadata["metadata"]["recursive"] is False


class TestCodeExecutorSimpleLLMCall:
  """Tests for RLMCodeExecutor._simple_llm_call method."""

  def test_simple_llm_call_emits_events(self):
    """Simple LLM call emits START and END events."""
    executor = RLMCodeExecutor(sub_model="test-model")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked response"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    mock_client = MagicMock()
    mock_client.completion.return_value = mock_response

    with patch("adk_rlm.code_executor.LiteLLMClient", return_value=mock_client):
      result = executor._simple_llm_call("Test prompt", "test-model")

    assert result == "Mocked response"

    # Should have 2 events: START and END
    events = []
    while not executor._event_queue.empty():
      events.append(executor._event_queue.get())

    assert len(events) == 2
    assert events[0].custom_metadata["event_type"] == RLMEventType.SUB_LLM_START.value
    assert events[1].custom_metadata["event_type"] == RLMEventType.SUB_LLM_END.value
    assert events[1].custom_metadata["response_full"] == "Mocked response"

  def test_simple_llm_call_logs_to_jsonl(self, temp_log_dir):
    """Simple LLM call logs to JSONL logger."""
    logger = RLMLogger(temp_log_dir)
    executor = RLMCodeExecutor(
        sub_model="test-model",
        logger=logger,
        parent_agent="rlm_agent",
    )
    executor._current_iteration = 1
    executor._current_block_index = 0

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Logged response"
    mock_response.usage = MagicMock()

    mock_client = MagicMock()
    mock_client.completion.return_value = mock_response

    with patch("adk_rlm.code_executor.LiteLLMClient", return_value=mock_client):
      executor._simple_llm_call("Logged prompt", "test-model")

    with open(logger.get_log_path()) as f:
      entry = json.loads(f.readline())

    assert entry["type"] == "simple_llm_call"
    assert entry["prompt_full"] == "Logged prompt"
    assert entry["response_full"] == "Logged response"


class TestCodeExecutorBatchedNonRecursive:
  """Tests for llm_query_batched with recursive=False."""

  def test_batched_non_recursive_emits_events(self):
    """Batched non-recursive calls emit events for each query."""
    executor = RLMCodeExecutor(sub_model="test-model")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Batch response"
    mock_response.usage = MagicMock()

    mock_client = MagicMock()
    mock_client.completion.return_value = mock_response

    llm_query_batched = executor._create_llm_query_batched_fn()

    with patch("adk_rlm.code_executor.LiteLLMClient", return_value=mock_client):
      results = llm_query_batched(
          ["Query 1", "Query 2"],
          recursive=False,
      )

    assert len(results) == 2

    events = []
    while not executor._event_queue.empty():
      events.append(executor._event_queue.get())

    # 2 queries * 2 events each = 4 events
    assert len(events) == 4

    start_events = [e for e in events if e.custom_metadata["event_type"] == RLMEventType.SUB_LLM_START.value]
    assert len(start_events) == 2

  def test_batched_non_recursive_logs_all_calls(self, temp_log_dir):
    """Batched non-recursive calls log all queries."""
    logger = RLMLogger(temp_log_dir)
    executor = RLMCodeExecutor(sub_model="test-model", logger=logger)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Batch response"
    mock_response.usage = MagicMock()

    mock_client = MagicMock()
    mock_client.completion.return_value = mock_response

    llm_query_batched = executor._create_llm_query_batched_fn()

    with patch("adk_rlm.code_executor.LiteLLMClient", return_value=mock_client):
      llm_query_batched(["Q1", "Q2"], recursive=False)

    with open(logger.get_log_path()) as f:
      entries = [json.loads(line) for line in f]

    assert len(entries) == 2
    assert all(e["type"] == "simple_llm_call" for e in entries)
