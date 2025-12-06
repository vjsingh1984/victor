# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Status bar widget for Victor TUI."""

from typing import Any

from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Status bar showing model info and metrics."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $primary-darken-3;
        color: $text-muted;
        padding: 0 1;
    }
    """

    provider: reactive[str] = reactive("ollama")
    model: reactive[str] = reactive("unknown")
    tokens: reactive[int] = reactive(0)
    tool_calls: reactive[int] = reactive(0)
    tool_budget: reactive[int] = reactive(15)
    status: reactive[str] = reactive("Ready")

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "unknown",
        tool_budget: int = 15,
        **kwargs: Any,
    ) -> None:
        """Initialize status bar.

        Args:
            provider: The LLM provider name.
            model: The model name.
            tool_budget: Maximum tool calls allowed.
            **kwargs: Additional arguments passed to parent Static widget.
        """
        super().__init__(**kwargs)
        self.provider = provider
        self.model = model
        self.tool_budget = tool_budget

    def render(self) -> str:
        """Render the status bar content."""
        return (
            f" {self.provider}/{self.model} | "
            f"Tokens: ~{self.tokens:,} | "
            f"Tools: {self.tool_calls}/{self.tool_budget} | "
            f"{self.status}"
        )

    def update_metrics(
        self,
        tokens: int | None = None,
        tool_calls: int | None = None,
        status: str | None = None,
    ) -> None:
        """Update status bar metrics.

        Args:
            tokens: Approximate token count.
            tool_calls: Number of tool calls used.
            status: Current status message.
        """
        if tokens is not None:
            self.tokens = tokens
        if tool_calls is not None:
            self.tool_calls = tool_calls
        if status is not None:
            self.status = status

    def set_streaming(self) -> None:
        """Set status to streaming."""
        self.status = "Streaming..."

    def set_ready(self) -> None:
        """Set status to ready."""
        self.status = "Ready"

    def set_thinking(self) -> None:
        """Set status to thinking."""
        self.status = "Thinking..."

    def set_tool_running(self, tool_name: str) -> None:
        """Set status to show tool is running.

        Args:
            tool_name: Name of the running tool.
        """
        self.status = f"Running {tool_name}..."
