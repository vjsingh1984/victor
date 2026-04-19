"""Framework version helpers for vertical compatibility checks."""

from __future__ import annotations


def get_framework_version() -> str:
    """Return the running Victor framework version.

    Preference order:
    1. Installed package metadata for ``victor-ai``
    2. The package ``victor.__version__`` fallback
    3. A conservative ``0.0.0`` fallback if neither is available
    """

    try:
        from importlib.metadata import version as get_version

        return get_version("victor-ai")
    except Exception:
        try:
            from victor import __version__

            return __version__
        except Exception:
            return "0.0.0"
