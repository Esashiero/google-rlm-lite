"""
Main entry point and convenience wrapper for ADK-RLM.

This module provides the RLM class which is the primary interface
for using Recursive Language Models with ADK framework integration.
"""

from pathlib import Path
from typing import Any, AsyncGenerator, Optional, TYPE_CHECKING

from adk_rlm.agents.rlm_agent import RLMAgent
from adk_rlm.config import get_config
from adk_rlm.events import RLMEventType
from adk_rlm.files import FileLoader, FileParser, FileSource
from adk_rlm.logging.rlm_logger import RLMLogger
from adk_rlm.types import RLMChatCompletion
from google.adk import Runner
from google.adk.events.event import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types

if TYPE_CHECKING:
  from adk_rlm.files import LazyFileCollection


class RLM:
  """
  Recursive Language Model - main user-facing class.
  """

  def __init__(
      self,
      model: Optional[str] = None,
      sub_model: Optional[str] = None,
      max_iterations: Optional[int] = None,
      max_depth: int = 5,
      custom_system_prompt: Optional[str] = None,
      log_dir: Optional[str] = None,
      verbose: bool = False,
      persistent: bool = False,
      # File handling
      file_sources: Optional[dict[str, FileSource]] = None,
      file_parsers: Optional[list[FileParser]] = None,
      base_path: Optional[str | Path] = None,
      # Legacy kwargs for compatibility
      backend: Optional[str] = None,
      backend_kwargs: Optional[dict[str, Any]] = None,
      **kwargs,
  ):
    config = get_config()

    model = model or config.default_model
    max_iterations = max_iterations if max_iterations is not None else config.max_iterations

    # Handle legacy backend_kwargs
    if (
        backend_kwargs
        and "model_name" in backend_kwargs
        and model == config.default_model
    ):
      model = backend_kwargs["model_name"]

    # Create logger if log_dir specified
    logger = RLMLogger(log_dir) if log_dir else None

    # Create the underlying agent
    self._agent = RLMAgent(
        name="rlm_agent",
        model=model,
        sub_model=sub_model,
        max_iterations=max_iterations,
        max_depth=max_depth,
        custom_system_prompt=custom_system_prompt,
        logger=logger,
        verbose=verbose,
        persistent=persistent,
    )

    # Create session service for ADK Runner
    self._session_service = InMemorySessionService()

    # Create ADK Runner
    self._runner = Runner(
        app_name="adk_rlm",
        agent=self._agent,
        session_service=self._session_service,
    )

    # Create file loader for file handling
    self._file_loader = FileLoader(
        sources=file_sources,
        parsers=file_parsers,
        base_path=base_path,
    )

    # Store config for reference
    self.model = model
    self.sub_model = sub_model or model
    self.max_iterations = max_iterations
    self.max_depth = max_depth
    self.persistent = persistent
    self.verbose = verbose
    self._logger = logger

  async def run_streaming(
      self,
      context: str | dict | list,
      prompt: Optional[str] = None,
      conversation_history: Optional[list[dict[str, str]]] = None,
  ) -> AsyncGenerator[Event, None]:
    # Create session with context in state
    session = await self._session_service.create_session(
        app_name="adk_rlm",
        user_id="default_user",
        state={
            "rlm_context": context,
            "rlm_prompt": prompt,
            "rlm_conversation_history": conversation_history,
        },
    )

    # Build user message (the agent reads from session state)
    message = types.Content(
        role="user", parts=[types.Part(text=prompt or "Analyze the context.")]
    )

    # Run agent and yield events
    async for event in self._runner.run_async(
        user_id="default_user",
        session_id=session.id,
        new_message=message,
    ):
      yield event

  def close(self) -> None:
    """Clean up resources (call when done with persistent mode)."""
    self._agent.close()

  def __enter__(self) -> "RLM":
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
    self.close()
    return False

  @property
  def log_path(self) -> Optional[str]:
    """Return the path to the log file if logging is enabled."""
    return self._logger.get_log_path() if self._logger else None

  @property
  def file_loader(self) -> FileLoader:
    """Access the file loader for direct file operations."""
    return self._file_loader

  @property
  def agent(self) -> RLMAgent:
    """Access the underlying RLM agent."""
    return self._agent

  @property
  def runner(self) -> Runner:
    """Access the ADK Runner for advanced usage."""
    return self._runner

  def load_files(
      self, files: list[str], lazy: bool = True
  ) -> "LazyFileCollection | list":
    if lazy:
      return self._file_loader.create_lazy_files(files)
    else:
      return self._file_loader.load_files(files)


def completion(
    context: Optional[str | dict | list] = None,
    prompt: Optional[str] = None,
    *,
    files: Optional[list[str]] = None,
    model: Optional[str] = None,
    sub_model: Optional[str] = None,
    max_iterations: Optional[int] = None,
    max_depth: int = 5,
    log_dir: Optional[str] = None,
    verbose: bool = False,
) -> RLMChatCompletion:
  import asyncio
  import time

  time_start = time.perf_counter()

  # Create RLM instance
  rlm = RLM(
      model=model,
      sub_model=sub_model,
      max_iterations=max_iterations,
      max_depth=max_depth,
      log_dir=log_dir,
      verbose=verbose,
  )

  # Build context from files if provided
  if files:
    file_context = rlm.file_loader.build_context(files, lazy=True)
    if context is not None:
      ctx = _merge_context(context, file_context)
    else:
      ctx = file_context
  else:
    if context is None:
      raise ValueError("Either 'context' or 'files' must be provided")
    ctx = context

  # Run streaming and collect final answer
  async def _run():
    final_answer = None
    async for event in rlm.run_streaming(ctx, prompt):
      if event.custom_metadata:
        event_type = event.custom_metadata.get("event_type")
        if event_type == RLMEventType.FINAL_ANSWER.value:
          final_answer = event.custom_metadata.get("answer")
    return final_answer

  try:
    final_answer = asyncio.run(_run())
  finally:
    rlm.close()

  time_end = time.perf_counter()

  return RLMChatCompletion(
      root_model=rlm.model,
      prompt=str(context) if context else str(files),
      response=final_answer or "",
      usage_summary=None,
      execution_time=time_end - time_start,
  )


def _merge_context(
    context: str | dict | list,
    file_context: dict,
) -> dict:
  """Merge direct context with file context."""
  if isinstance(context, str):
    return {"user_context": context, **file_context}
  elif isinstance(context, dict):
    return {**context, **file_context}
  else:
    return {"user_context": context, **file_context}
