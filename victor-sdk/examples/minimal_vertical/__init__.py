"""Example minimal vertical using only victor-sdk.

This demonstrates how to create a zero-runtime-dependency vertical
that only depends on victor-sdk (protocols only).
"""

from victor_sdk import PluginContext, VictorPlugin
from victor_sdk.verticals.protocols.base import VerticalBase
from victor_sdk.verticals.protocols import ToolProvider, SafetyProvider
from victor_sdk.core.types import Tier


class MinimalVertical(VerticalBase):
    """A minimal vertical example with zero runtime dependencies.

    This vertical only imports from victor-sdk and can be installed
    without pulling in the entire victor-ai framework.
    """

    @classmethod
    def get_name(cls) -> str:
        """Return vertical identifier."""
        return "minimal"

    @classmethod
    def get_description(cls) -> str:
        """Return human-readable description."""
        return "Minimal example vertical"

    @classmethod
    def get_tools(cls) -> list[str]:
        """Return list of tool names."""
        return [
            "read",  # Read file contents
            "write",  # Write file contents
            "search",  # Search codebase
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Return system prompt."""
        return """You are a helpful assistant with access to basic file tools.

You can:
- Read files to understand their contents
- Write files to create or modify code
- Search the codebase to find relevant information

Always be helpful and accurate in your responses."""


class MinimalPlugin(VictorPlugin):
    """Minimal VictorPlugin wrapper for the SDK-only example."""

    @property
    def name(self) -> str:
        return "minimal"

    def register(self, context: PluginContext) -> None:
        context.register_vertical(MinimalVertical)

    def get_cli_app(self):
        return None

    def on_activate(self) -> None:
        return None

    def on_deactivate(self) -> None:
        return None

    async def on_activate_async(self) -> None:
        return None

    async def on_deactivate_async(self) -> None:
        return None

    def health_check(self) -> dict[str, object]:
        return {"healthy": True, "vertical": "minimal"}


plugin = MinimalPlugin()


__all__ = ["MinimalVertical", "MinimalPlugin", "plugin"]
