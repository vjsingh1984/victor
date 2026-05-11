"""Victor plugin entry point for the research vertical."""

from __future__ import annotations

from typing import Any, Dict, Optional

from victor_sdk import PluginContext, VictorPlugin


class ResearchPlugin(VictorPlugin):
    """VictorPlugin adapter for the research vertical package."""

    @property
    def name(self) -> str:
        return "research"

    def register(self, context: PluginContext) -> None:
        from victor_research.assistant import ResearchAssistant

        context.register_vertical(ResearchAssistant)

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
            "vertical": "research",
            "vertical_class": "ResearchAssistant",
        }


plugin = ResearchPlugin()


__all__ = ["ResearchPlugin", "plugin"]
