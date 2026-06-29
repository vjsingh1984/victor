"""Backward/forward compatibility layer for LanceDB API changes.

LanceDB's table-listing API has shifted across versions:

- ``list_tables()`` is the canonical method on current versions (returns a
  plain list of names). It is tried first so call-sites never trigger the
  ``table_names() is deprecated`` warning emitted by recent releases.
- ``table_names()`` is the legacy listing method (also returns a plain list
  on newer versions but is deprecated there); kept as a fallback for older
  installs that predate ``list_tables()``.
- On the oldest versions ``list_tables()`` returned an object with a
  ``.tables`` attribute rather than a plain list; that shape is normalized.

This helper transparently supports all three so call-sites don't break when
the installed version changes.
"""

from __future__ import annotations

import warnings
from typing import Any, List


def get_table_names(db: Any) -> List[str]:
    """Get table names with backward/forward LanceDB API compatibility."""
    if hasattr(db, "list_tables"):
        response = db.list_tables()
        if hasattr(response, "tables"):
            return list(response.tables)
        return list(response) if response else []
    if hasattr(db, "table_names"):
        # Legacy LanceDB: list_tables() did not exist yet. table_names() is
        # deprecated on current versions but is the only option here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = db.table_names()
        return list(result) if result else []
    return []
