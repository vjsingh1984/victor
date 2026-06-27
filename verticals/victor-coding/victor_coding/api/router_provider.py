"""FastAPI router provider for coding vertical endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LSPRequest(BaseModel):
    """LSP request payload."""

    file: str
    line: int = 0
    character: int = 0


def create_coding_lsp_router() -> APIRouter:
    """Create `/lsp/*` routes for coding integrations."""
    router = APIRouter(tags=["LSP"])

    @router.post("/lsp/completions")
    async def lsp_completions(request: LSPRequest) -> dict[str, Any]:
        """LSP completions."""
        try:
            manager = _get_lsp_manager()
            completions = await manager.get_completions(
                request.file,
                request.line,
                request.character,
            )
            return {
                "completions": [
                    {
                        "label": item.label,
                        "kind": item.kind,
                        "detail": item.detail,
                        "insert_text": item.insert_text,
                    }
                    for item in completions
                ]
            }
        except Exception as exc:
            logger.exception("LSP completions error")
            return {"completions": [], "error": str(exc)}

    @router.post("/lsp/hover")
    async def lsp_hover(request: LSPRequest) -> dict[str, Any]:
        """LSP hover."""
        try:
            manager = _get_lsp_manager()
            hover = await manager.get_hover(request.file, request.line, request.character)
            return {"contents": hover.contents if hover else None}
        except Exception as exc:
            logger.exception("LSP hover error")
            return {"contents": None, "error": str(exc)}

    @router.post("/lsp/definition")
    async def lsp_definition(request: LSPRequest) -> dict[str, Any]:
        """LSP definition."""
        try:
            manager = _get_lsp_manager()
            locations = await manager.get_definition(request.file, request.line, request.character)
            return {"locations": locations}
        except Exception as exc:
            logger.exception("LSP definition error")
            return {"locations": [], "error": str(exc)}

    @router.post("/lsp/references")
    async def lsp_references(request: LSPRequest) -> dict[str, Any]:
        """LSP references."""
        try:
            manager = _get_lsp_manager()
            locations = await manager.get_references(request.file, request.line, request.character)
            return {"locations": locations}
        except Exception as exc:
            logger.exception("LSP references error")
            return {"locations": [], "error": str(exc)}

    @router.post("/lsp/diagnostics")
    async def lsp_diagnostics(request: LSPRequest) -> dict[str, Any]:
        """LSP diagnostics."""
        try:
            manager = _get_lsp_manager()
            diagnostics = manager.get_diagnostics(request.file)
            return {"diagnostics": diagnostics}
        except Exception as exc:
            logger.exception("LSP diagnostics error")
            return {"diagnostics": [], "error": str(exc)}

    return router


def get_fastapi_router_provider(*, workspace_root: str) -> APIRouter:
    """Entry-point callable for ``victor.api_routers``."""
    _ = workspace_root
    return create_coding_lsp_router()


def _get_lsp_manager() -> Any:
    from victor_coding.lsp.manager import get_lsp_manager

    return get_lsp_manager()


__all__ = [
    "create_coding_lsp_router",
    "get_fastapi_router_provider",
]
