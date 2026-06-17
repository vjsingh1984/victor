"""Backward-compatible analytics exports.

This namespace preserves older ``victor.analytics`` imports while routing
callers to the canonical observability analytics package.
"""

from victor.observability.analytics import *  # noqa: F403
