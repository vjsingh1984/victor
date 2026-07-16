# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Framework LSP adapter — activates FEP-0019 in victor-coding.

Bridges victor-coding's :class:`LSPConnectionPool` (which returns
``victor.framework.lsp`` types and display-oriented dicts) to the framework's
``LSPServiceProtocol`` ``LSP*`` types, so the FEP-0019 chain activates:

* Phase 1 — ``LSPVerifier`` (post-COMPLETE diagnostics gate)
* Phase 2 — ``LSPDiagnosticMiddleware`` (same-turn post-edit diagnostics)
* Phase 3 — ``LSPContextProvider`` (proactive document-symbol context)

Only the methods the framework runtime actually calls are implemented; the
position-based ones (hover/completions/definition/references) gracefully
degrade (``LSPCapability`` returns ``None``/``[]`` via its ``hasattr`` check)
until a cursor feature needs them.

Exposed via :meth:`CodingAssistant.get_lsp`, which the framework's
``FrameworkStepHandler.apply_lsp`` routes to ``orchestrator.set_lsp``.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from victor.framework.lsp import DocumentSymbol
from victor.framework.lsp_protocols import (
    LSPDiagnostic,
    LSPPosition,
    LSPRange,
    LSPSymbol,
)

logger = logging.getLogger(__name__)

# The pool renders severity as a display name; map back to LSP severity ints.
# https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#diagnosticSeverity
_SEVERITY_NAME_TO_INT = {
    "error": 1,
    "warning": 2,
    "info": 3,
    "hint": 4,
}


class FrameworkLSPAdapter:
    """Adapt ``LSPConnectionPool`` to the framework's ``LSPServiceProtocol``.

    Implements the methods the FEP-0019 runtime consumes:
    ``open_document``, ``update_document``, ``close_document``,
    ``get_diagnostics``, ``get_document_symbols``.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    # --- document lifecycle -------------------------------------------------

    async def open_document(self, file_path: str, content: Optional[str] = None) -> bool:
        try:
            return bool(await self._pool.open_document(file_path, content))
        except Exception:
            logger.debug("LSP open_document failed for %s", file_path, exc_info=True)
            return False

    async def update_document(self, file_path: str, content: str) -> bool:
        try:
            return bool(await self._pool.update_document(file_path, content))
        except Exception:
            logger.debug("LSP update_document failed for %s", file_path, exc_info=True)
            return False

    def close_document(self, file_path: str) -> None:
        try:
            self._pool.close_document(file_path)
        except Exception:
            logger.debug("LSP close_document failed for %s", file_path, exc_info=True)

    # --- diagnostics (Phase 1 + 2) ------------------------------------------

    def get_diagnostics(self, file_path: str) -> List[LSPDiagnostic]:
        """Convert the pool's display dicts to ``LSPDiagnostic``.

        The pool returns ``[{line (1-indexed), character, message, severity
        (name), source, code}]``; the framework wants 0-indexed ranges + int
        severity.
        """
        try:
            raw = self._pool.get_diagnostics(file_path) or []
        except Exception:
            logger.debug("LSP get_diagnostics failed for %s", file_path, exc_info=True)
            return []
        out: List[LSPDiagnostic] = []
        for d in raw:
            line = max(0, int(d.get("line") or 1) - 1)  # pool is 1-indexed
            character = int(d.get("character") or 0)
            rng = LSPRange(
                start=LSPPosition(line=line, character=character),
                end=LSPPosition(line=line, character=character),
            )
            code = d.get("code")
            out.append(
                LSPDiagnostic(
                    range=rng,
                    message=str(d.get("message") or ""),
                    severity=_SEVERITY_NAME_TO_INT.get(str(d.get("severity")).lower(), 1),
                    source=d.get("source"),
                    code=None if code is None else str(code),
                )
            )
        return out

    # --- document symbols (Phase 3) ----------------------------------------

    async def get_document_symbols(self, file_path: str) -> List[LSPSymbol]:
        try:
            symbols = await self._pool.get_document_symbols(file_path)
        except Exception:
            logger.debug("LSP get_document_symbols failed for %s", file_path, exc_info=True)
            return []
        return [_convert_symbol(s) for s in (symbols or [])]


def _convert_range(rng: Any) -> LSPRange:
    start = getattr(rng, "start", None)
    end = getattr(rng, "end", None) or start
    return LSPRange(
        start=LSPPosition(
            line=int(getattr(start, "line", 0)),
            character=int(getattr(start, "character", 0)),
        ),
        end=LSPPosition(
            line=int(getattr(end, "line", 0)),
            character=int(getattr(end, "character", 0)),
        ),
    )


def _convert_symbol(symbol: Any) -> LSPSymbol:
    children = getattr(symbol, "children", None) or []
    return LSPSymbol(
        name=getattr(symbol, "name", ""),
        kind=int(getattr(symbol, "kind", 0)),
        range=_convert_range(getattr(symbol, "range", None)),
        selection_range=_convert_range(getattr(symbol, "selection_range", None)),
        detail=getattr(symbol, "detail", None),
        children=[_convert_symbol(c) for c in children] or None,
        deprecated=bool(getattr(symbol, "deprecated", False)),
    )
