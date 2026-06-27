"""Victor plugin entry point for the RAG vertical."""

from __future__ import annotations

from typing import Any, Dict, Optional

from victor_contracts import PluginContext, VictorPlugin


class RAGPlugin(VictorPlugin):
    """VictorPlugin adapter for the RAG vertical package."""

    @property
    def name(self) -> str:
        return "rag"

    def register(self, context: PluginContext) -> None:
        from victor_rag.assistant import RAGAssistant

        context.register_vertical(RAGAssistant)

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
            "vertical": "rag",
            "vertical_class": "RAGAssistant",
        }


plugin = RAGPlugin()


__all__ = ["RAGPlugin", "plugin"]
