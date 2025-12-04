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

"""Tool execution indicator widget for Victor TUI."""

from textual.reactive import reactive
from textual.widgets import Static


class ToolIndicator(Static):
    """Simple indicator for tool execution status."""

    DEFAULT_CSS = """
    ToolIndicator {
        margin: 0 2;
        padding: 0 1;
        height: 1;
        background: $surface;
    }

    ToolIndicator.running {
        color: $warning;
    }

    ToolIndicator.success {
        color: $success;
    }

    ToolIndicator.error {
        color: $error;
    }
    """

    tool_name: reactive[str] = reactive("")
    status: reactive[str] = reactive("running")
    elapsed: reactive[float] = reactive(0.0)

    def __init__(
        self,
        tool_name: str,
        status: str = "running",
    ) -> None:
        """Initialize tool indicator.

        Args:
            tool_name: Name of the tool being executed.
            status: Current status (running, success, error).
        """
        # Build initial content
        content = f"⟳ [bold]{tool_name}[/] running..."
        super().__init__(content)
        self.tool_name = tool_name
        self.status = status

    def on_mount(self) -> None:
        """Set initial class based on status."""
        self.add_class(self.status)

    def _update_display(self) -> None:
        """Update the display based on current status."""
        if self.status == "running":
            self.update(f"⟳ [bold]{self.tool_name}[/] running...")
        elif self.status == "success":
            elapsed_text = f" ({self.elapsed:.1f}s)" if self.elapsed > 0 else ""
            self.update(f"✓ [bold]{self.tool_name}[/] OK{elapsed_text}")
        elif self.status == "error":
            elapsed_text = f" ({self.elapsed:.1f}s)" if self.elapsed > 0 else ""
            self.update(f"✗ [bold]{self.tool_name}[/] FAILED{elapsed_text}")

    def set_success(self, elapsed: float = 0.0, preview: str = "") -> None:
        """Mark tool as successfully completed.

        Args:
            elapsed: Time elapsed in seconds.
            preview: Preview of the result (unused).
        """
        self.remove_class("running")
        self.add_class("success")
        self.status = "success"
        self.elapsed = elapsed
        self._update_display()

    def set_error(self, elapsed: float = 0.0, error_msg: str = "") -> None:
        """Mark tool as failed.

        Args:
            elapsed: Time elapsed in seconds.
            error_msg: Error message (unused).
        """
        self.remove_class("running")
        self.add_class("error")
        self.status = "error"
        self.elapsed = elapsed
        self._update_display()
