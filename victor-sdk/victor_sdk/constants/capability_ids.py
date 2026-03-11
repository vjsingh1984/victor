"""Stable SDK-owned capability identifiers."""

from typing import Set


class CapabilityIds:
    """Registry of canonical host/runtime capability identifiers."""

    FILE_OPS = "file_ops"
    FILE_OPERATIONS = FILE_OPS
    GIT = "git"
    LSP = "lsp"
    WEB_ACCESS = "web_access"

    PROMPT_CONTRIBUTIONS = "prompt_contributions"
    PRIVACY = "privacy"
    SECRETS_MASKING = "secrets_masking"
    AUDIT_LOGGING = "audit_logging"
    STAGES = "stages"
    GROUNDING_RULES = "grounding_rules"
    VALIDATION = "validation"
    SAFETY_RULES = "safety_rules"
    TASK_HINTS = "task_hints"
    SOURCE_VERIFICATION = "source_verification"


def get_all_capability_ids() -> Set[str]:
    """Return all canonical capability identifiers."""

    return {
        getattr(CapabilityIds, attr)
        for attr in dir(CapabilityIds)
        if not attr.startswith("_") and isinstance(getattr(CapabilityIds, attr), str)
    }


def is_known_capability_id(capability_id: str) -> bool:
    """Check whether a capability identifier is part of the SDK registry."""

    return capability_id in get_all_capability_ids()


__all__ = [
    "CapabilityIds",
    "get_all_capability_ids",
    "is_known_capability_id",
]
