"""Framework-wide constants.

Keep this module dependency-free so any victor subpackage can import from it
without risking circular imports or pulling in heavy modules.
"""

# The fallback vertical used when no explicit vertical is specified.
# User-configurable default lives in victor.config.settings.default_vertical;
# this constant is for function-signature defaults and dataclass fields only.
DEFAULT_VERTICAL: str = "coding"
