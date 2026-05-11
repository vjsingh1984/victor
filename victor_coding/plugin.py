"""Victor plugin entry point for the coding vertical."""

from __future__ import annotations

from typing import Any, Dict, Optional

from victor_sdk import PluginContext, VictorPlugin


class CodingPlugin(VictorPlugin):
    """VictorPlugin adapter for the coding vertical package."""

    @property
    def name(self) -> str:
        return "coding"

    def register(self, context: PluginContext) -> None:
        from victor_coding.assistant import CodingAssistant

        context.register_vertical(CodingAssistant)

    def get_cli_app(self) -> Optional[Any]:
        return None

    def on_activate(self) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    async def on_activate_async(self) -> None:
        pass

    async def on_deactivate_async(self) -> None:
        pass

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "vertical": "coding",
            "vertical_class": "CodingAssistant",
        }


plugin = CodingPlugin()


__all__ = ["CodingPlugin", "plugin"]
