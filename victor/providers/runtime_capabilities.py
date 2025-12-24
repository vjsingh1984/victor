"""Runtime capability discovery structures for providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ProviderRuntimeCapabilities:
    """Runtime capabilities and limits for a provider/model pair."""

    provider: str
    model: str
    context_window: int
    supports_tools: bool
    supports_streaming: bool
    source: str = "config"  # e.g., "discovered", "config"
    raw: Optional[Dict[str, Any]] = None
