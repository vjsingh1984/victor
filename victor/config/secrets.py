from __future__ import annotations

"""Helpers for working with secret-bearing configuration values."""

from typing import Any

from pydantic import SecretStr


def reveal_secret(value: SecretStr | str | None) -> str | None:
    """Return the raw secret value for adapter/runtime edges only."""
    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


def unwrap_secrets(value: Any) -> Any:
    """Recursively convert SecretStr values into plain strings."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, dict):
        return {key: unwrap_secrets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unwrap_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(unwrap_secrets(item) for item in value)
    return value


__all__ = ["reveal_secret", "unwrap_secrets"]
