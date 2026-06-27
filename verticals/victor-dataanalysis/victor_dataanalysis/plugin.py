"""Victor plugin entry point for the data analysis vertical."""

from __future__ import annotations

from typing import Any, Dict, Optional

from victor_contracts import PluginContext, VictorPlugin


class DataAnalysisPlugin(VictorPlugin):
    """VictorPlugin adapter for the data analysis vertical package."""

    @property
    def name(self) -> str:
        return "dataanalysis"

    def register(self, context: PluginContext) -> None:
        from victor_dataanalysis.assistant import DataAnalysisAssistant

        context.register_vertical(DataAnalysisAssistant)

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
            "vertical": "dataanalysis",
            "vertical_class": "DataAnalysisAssistant",
        }


plugin = DataAnalysisPlugin()


__all__ = ["DataAnalysisPlugin", "plugin"]
