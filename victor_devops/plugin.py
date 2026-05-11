"""Victor plugin entry point for the devops vertical."""

from __future__ import annotations

from typing import Any, Dict, Optional

from victor_sdk import PluginContext, VictorPlugin


class DevOpsPlugin(VictorPlugin):
    """VictorPlugin adapter for the devops vertical package."""

    @property
    def name(self) -> str:
        return "devops"

    def register(self, context: PluginContext) -> None:
        from victor_devops.assistant import DevOpsAssistant

        context.register_vertical(DevOpsAssistant)

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
            "vertical": "devops",
            "vertical_class": "DevOpsAssistant",
        }


plugin = DevOpsPlugin()


__all__ = ["DevOpsPlugin", "plugin"]
