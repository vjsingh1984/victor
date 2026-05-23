"""Victor plugin entry point for the coding vertical."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from victor_contracts import PluginContext, VictorPlugin

logger = logging.getLogger(__name__)


class CodingPlugin(VictorPlugin):
    """VictorPlugin adapter for the coding vertical package."""

    @property
    def name(self) -> str:
        return "coding"

    def register(self, context: PluginContext) -> None:
        from victor_coding.assistant import CodingAssistant

        context.register_vertical(CodingAssistant)
        self._register_ccg_builder(context)
        self._register_tree_sitter_analysis(context)

    def _register_ccg_builder(self, context: PluginContext) -> None:
        """Register the language-plugin-backed CCG builder when the host supports it."""
        register_ccg_builder = getattr(context, "register_ccg_builder", None)
        if not callable(register_ccg_builder):
            return
        from victor_coding.codebase.ccg_builder import PluginBackedCCGBuilder

        register_ccg_builder("all", PluginBackedCCGBuilder())
        logger.debug("Registered victor-coding plugin-backed CCG builder")

    def _register_tree_sitter_analysis(self, context: PluginContext) -> None:
        """Register the analysis-level tree-sitter provider with the host registry.

        Probes the host for ``register_capability`` and the root
        ``TreeSitterAnalysisProtocol``; if either is missing we silently skip
        so older hosts continue to work with the root null stub.
        """
        register_capability = getattr(context, "register_capability", None)
        if not callable(register_capability):
            return
        try:
            from victor.framework.vertical_protocols import TreeSitterAnalysisProtocol
        except Exception:
            logger.debug("Root TreeSitterAnalysisProtocol unavailable; skipping registration")
            return
        from victor_coding.codebase.tree_sitter_analysis import (
            TreeSitterAnalysisProvider,
        )

        try:
            register_capability(TreeSitterAnalysisProtocol, TreeSitterAnalysisProvider())
            logger.debug("Registered victor-coding tree-sitter analysis provider")
        except Exception as exc:
            logger.warning("Failed to register tree-sitter analysis provider: %s", exc)

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
            "ccg_builder": self._ccg_builder_health(),
        }

    def _ccg_builder_health(self) -> Dict[str, Any]:
        try:
            from victor_coding.codebase.ccg_builder import PluginBackedCCGBuilder

            info = PluginBackedCCGBuilder().get_builder_info()
            return {"registered": True, **info}
        except Exception as exc:
            return {
                "registered": False,
                "error": str(exc),
            }


plugin = CodingPlugin()


__all__ = ["CodingPlugin", "plugin"]
