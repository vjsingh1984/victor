"""Framework-wide constants.

Keep this module dependency-free so any victor subpackage can import from it
without risking circular imports or pulling in heavy modules.
"""

import os

# The fallback vertical used when no explicit vertical is specified.
# Override via VICTOR_DEFAULT_VERTICAL env var.
# User-configurable default also lives in victor.config.settings.default_vertical.
DEFAULT_VERTICAL: str = os.environ.get("VICTOR_DEFAULT_VERTICAL", "")
