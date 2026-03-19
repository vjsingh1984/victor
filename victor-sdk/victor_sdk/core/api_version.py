"""API version constants for the Victor SDK extension contract.

The api_version field in ExtensionManifest tracks breaking changes to the
vertical ↔ framework integration surface. Bump CURRENT_API_VERSION when the
manifest schema or negotiation protocol changes in a non-backward-compatible way.
"""

CURRENT_API_VERSION: int = 2
MIN_SUPPORTED_API_VERSION: int = 1


def is_compatible(version: int) -> bool:
    """Return True if *version* falls within the supported range."""
    return MIN_SUPPORTED_API_VERSION <= version <= CURRENT_API_VERSION
