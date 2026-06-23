import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Any


@dataclass
class BackgroundTaskDef:
    """Definition for a task that should be detached and run in the background.

    When a tool returns this object instead of a string, the ToolExecutor
    intercepts it and hands the `coro` to the TaskManager.
    """

    # The asynchronous task to execute (or already executing) in the background
    task: Awaitable[Any]

    # A unique identifier for the task, generated automatically
    task_id: str = field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")

    # Context or metadata about the task, e.g. the original tool name
    context: str = "shell_tool"
