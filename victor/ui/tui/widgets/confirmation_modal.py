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

"""Confirmation modal widget for Victor TUI."""

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Static, Label

from victor.agent.safety import ConfirmationRequest, RiskLevel


class ConfirmationModal(ModalScreen[bool]):
    """Modal dialog for dangerous operation confirmation."""

    DEFAULT_CSS = """
    ConfirmationModal {
        align: center middle;
    }

    ConfirmationModal > Vertical {
        background: $surface;
        padding: 1 2;
        border: thick $warning;
        width: 70;
        max-width: 90%;
        height: auto;
        max-height: 80%;
    }

    ConfirmationModal .modal-title {
        text-align: center;
        text-style: bold;
        padding: 1 0;
        width: 100%;
    }

    ConfirmationModal .modal-title.safe {
        color: $success;
    }

    ConfirmationModal .modal-title.low {
        color: $primary;
    }

    ConfirmationModal .modal-title.medium {
        color: $warning;
    }

    ConfirmationModal .modal-title.high {
        color: $error;
    }

    ConfirmationModal .modal-title.critical {
        color: $error;
        text-style: bold;
    }

    ConfirmationModal .modal-content {
        padding: 1 2;
    }

    ConfirmationModal .field-label {
        color: $text-muted;
        margin-right: 1;
    }

    ConfirmationModal .field-value {
        color: $text;
    }

    ConfirmationModal .details-section {
        margin-top: 1;
        padding: 1;
        background: $background;
        border: round $primary-darken-2;
    }

    ConfirmationModal .detail-item {
        color: $text;
        margin-left: 2;
    }

    ConfirmationModal Horizontal {
        height: auto;
        width: 100%;
        align: center middle;
        padding: 1 0;
    }

    ConfirmationModal Button {
        margin: 0 2;
        min-width: 12;
    }

    ConfirmationModal #confirm-btn {
        background: $error;
    }

    ConfirmationModal #cancel-btn {
        background: $primary;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "confirm", "Confirm"),
    ]

    def __init__(self, request: ConfirmationRequest) -> None:
        """Initialize confirmation modal.

        Args:
            request: The confirmation request with risk details.
        """
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        """Compose the modal content."""
        risk_icons = {
            RiskLevel.SAFE: "OK",
            RiskLevel.LOW: "!",
            RiskLevel.MEDIUM: "!!",
            RiskLevel.HIGH: "!!!",
            RiskLevel.CRITICAL: "!!!",
        }
        icon = risk_icons.get(self.request.risk_level, "?")

        with Vertical():
            # Title with risk level
            title = Static(
                f"[bold]{icon} {self.request.risk_level.value.upper()} RISK OPERATION {icon}[/]",
                classes=f"modal-title {self.request.risk_level.value}",
            )
            yield title

            # Content
            with Vertical(classes="modal-content"):
                yield Static(f"[bold]Tool:[/] [cyan]{self.request.tool_name}[/]")
                yield Static(f"[bold]Action:[/] {self.request.description}")

                # Details section
                if self.request.details:
                    yield Static("")
                    yield Static("[bold]Details:[/]")
                    with Vertical(classes="details-section"):
                        for detail in self.request.details:
                            yield Static(f"[dim]-[/] {detail}", classes="detail-item")

            # Buttons
            with Horizontal():
                yield Button("Cancel", id="cancel-btn", variant="primary")
                yield Button("Confirm", id="confirm-btn", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press.

        Args:
            event: The button pressed event.
        """
        if event.button.id == "confirm-btn":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss(False)

    def action_confirm(self) -> None:
        """Handle enter key."""
        self.dismiss(True)
