"""Shared UI helpers for delegate follow-up contract handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DelegateFollowUpContractError(ValueError):
    """Raised when a delegate follow-up contract file is malformed."""


def load_delegate_follow_up_contract_file(contract_path: Path) -> dict[str, Any]:
    """Load a delegate follow-up contract JSON object from disk."""
    if not contract_path.exists():
        raise FileNotFoundError(str(contract_path))

    try:
        with contract_path.open(encoding="utf-8") as f:
            contract = json.load(f)
    except json.JSONDecodeError as e:
        raise DelegateFollowUpContractError(
            f"Invalid JSON in delegate follow-up contract: {e}"
        ) from e

    if not isinstance(contract, dict):
        raise DelegateFollowUpContractError("Delegate follow-up contract must be a JSON object")

    return contract
