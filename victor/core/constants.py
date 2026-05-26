"""Framework-wide constants.

Keep this module dependency-free so any victor subpackage can import from it
without risking circular imports or pulling in heavy modules.
"""

import os

# Default vertical name used when no --vertical flag is passed.
# Empty string means "no vertical / base framework mode".
# Override via VICTOR_DEFAULT_VERTICAL env var; also user-configurable via
# victor.config.settings.default_vertical.
DEFAULT_VERTICAL: str = os.environ.get("VICTOR_DEFAULT_VERTICAL", "")
