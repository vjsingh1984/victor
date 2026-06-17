"""Token-optimized serialization strategies."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class SerializationSettings(BaseModel):
    """Token-optimized serialization strategies."""

    serialization_enabled: bool = True
    serialization_default_format: Optional[str] = None
    serialization_min_savings_threshold: float = 0.15
    serialization_include_format_hint: bool = True
    serialization_min_rows_for_tabular: int = 3
    serialization_debug_mode: bool = False
