"""Backward/forward compatibility layer for LanceDB API changes.

LanceDB >=0.25 replaced ``list_tables()`` (returning an object with a
``.tables`` attribute) with ``table_names()`` (returning a plain list).
This helper transparently supports both APIs so call-sites don't break
when the installed version changes.
"""

from __future__ import annotations

from typing import Any, List


def get_table_names(db: Any) -> List[str]:
    """Get table names with backward/forward LanceDB API compatibility."""
    if hasattr(db, "table_names"):
        result = db.table_names()
        return list(result) if result else []
    if hasattr(db, "list_tables"):
        response = db.list_tables()
        if hasattr(response, "tables"):
            return list(response.tables)
        return list(response) if response else []
    return []
