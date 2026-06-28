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
    """Get table names with backward/forward LanceDB API compatibility.

    Prefers ``list_tables()`` (canonical on current versions; avoids the
    ``table_names()`` deprecation warning) but falls back to ``table_names()``
    when ``list_tables()`` is absent OR returns empty — some versions report an
    empty list from ``list_tables()`` while ``table_names()`` still lists the
    tables.
    """
    if hasattr(db, "list_tables"):
        response = db.list_tables()
        if hasattr(response, "tables"):
            names = list(response.tables)
        else:
            names = list(response) if response else []
        if names:
            return names
    if hasattr(db, "table_names"):
        # Legacy LanceDB (no list_tables) OR list_tables() returned empty on a
        # version where table_names() still lists the tables. table_names() is
        # deprecated on current versions, so silence the warning here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = db.table_names()
        return list(result) if result else []
    return []
