# Symbol resolution utilities for graph linking.
from __future__ import annotations

from typing import Dict, Iterable, List, Optional


class SymbolResolver:
    """Resolve symbol names to node ids with basic heuristics."""

    def __init__(self) -> None:
        # name -> list of node_ids
        self._index: Dict[str, List[str]] = {}

    def ingest(self, node_ids: Iterable[str]) -> None:
        """Ingest symbol node ids (format: symbol:<file>:<name>)."""
        for node_id in node_ids:
            if not node_id.startswith("symbol:"):
                continue
            try:
                _, sym_path = node_id.split("symbol:", 1)
                file_part, sym_name = sym_path.split(":", 1)
            except ValueError:
                continue
            self._index.setdefault(sym_name, []).append(node_id)
            # Also index by short basename if present (e.g., Class.method -> method)
            if "." in sym_name:
                short = sym_name.split(".")[-1]
                self._index.setdefault(short, []).append(node_id)

    def resolve(self, symbol_name: str, preferred_file: Optional[str] = None) -> Optional[str]:
        """Resolve a symbol to a node id, preferring matches outside preferred_file."""
        matches = self._index.get(symbol_name)
        if not matches:
            return None
        if preferred_file:
            for m in matches:
                # node id format: symbol:<file>:<name>
                if not m.startswith(f"symbol:{preferred_file}:"):
                    return m
        return matches[0]

    def clear(self) -> None:
        self._index.clear()
