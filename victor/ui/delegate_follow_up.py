"""Shared UI helpers for delegate follow-up contract handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


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


def build_delegate_follow_up_suggestions(
    *,
    workflow_path: str | Path,
    contract_path: str | Path,
    contract: Mapping[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Build selectable UI suggestions from a delegate follow-up contract."""
    steps = _extract_delegate_follow_up_steps(contract)
    if not steps:
        raise DelegateFollowUpContractError(
            "Delegate follow-up contract does not contain selectable next_steps"
        )

    workflow_arg = str(workflow_path)
    contract_arg = str(contract_path)
    suggestions: list[dict[str, Any]] = []
    for step in steps[:limit]:
        step_id = _coerce_nonempty_text(step.get("step_id") or step.get("step"))
        if step_id is None:
            continue
        description = _build_step_description(step, step_id)
        suggestions.append(
            {
                "command": f"/delegate-follow-up {workflow_arg} {contract_arg} {step_id}",
                "description": description,
                "step_id": step_id,
                "requires_approval": step.get("requires_approval"),
            }
        )

    if not suggestions:
        raise DelegateFollowUpContractError(
            "Delegate follow-up contract does not contain selectable step IDs"
        )
    return suggestions


def _extract_delegate_follow_up_steps(
    contract: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    raw_steps = contract.get("next_steps")
    if not isinstance(raw_steps, list):
        approval_contract = contract.get("approval_contract")
        if isinstance(approval_contract, Mapping):
            raw_steps = approval_contract.get("next_steps")
    if not isinstance(raw_steps, list):
        return []
    return [step for step in raw_steps if isinstance(step, Mapping)]


def _build_step_description(step: Mapping[str, Any], step_id: str) -> str:
    summary = (
        _coerce_nonempty_text(step.get("instruction"))
        or _coerce_nonempty_text(step.get("description"))
        or _coerce_nonempty_text(step.get("step"))
        or step_id
    )
    description = f"{step_id}: {summary}"
    if len(description) > 120:
        return description[:117] + "..."
    return description


def _coerce_nonempty_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
