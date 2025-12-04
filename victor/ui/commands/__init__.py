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

"""Modular slash commands for Victor CLI.

This package organizes slash commands into logical groups:
- base: Command registry and base classes
- session: Session management (save, load, resume)
- model: Model/provider management
- context: Context and initialization
- history: Undo/redo/snapshots
- tools: Tool management
- analysis: Review, metrics, search

Backward Compatibility:
- SlashCommandHandler is re-exported from slash_commands.py for existing code
"""

# Re-export legacy SlashCommandHandler for backward compatibility
from victor.ui.slash_commands import SlashCommand as LegacySlashCommand
from victor.ui.slash_commands import SlashCommandHandler

# New modular command system
from victor.ui.commands.base import (
    SlashCommand,
    CommandGroup,
    CommandRegistry,
    CommandContext,
    get_command_registry,
    set_command_registry,
    reset_command_registry,
)

__all__ = [
    # Backward compatibility
    "SlashCommandHandler",
    "LegacySlashCommand",
    # New modular system
    "SlashCommand",
    "CommandGroup",
    "CommandRegistry",
    "CommandContext",
    "get_command_registry",
    "set_command_registry",
    "reset_command_registry",
]
