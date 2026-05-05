# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Guardrails for PromptRuntimeSupport compatibility-only usage."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path("victor")
ALLOWED_FILES = {
    Path("victor/agent/coordinators/__init__.py"),
    Path("victor/agent/coordinators/coordinator_factory.py"),
    Path("victor/agent/coordinators/factory_support.py"),
    Path("victor/agent/factory/coordination_builders.py"),
    Path("victor/agent/services/__init__.py"),
    Path("victor/agent/services/prompt_runtime_support.py"),
    Path("victor/agent/services/system_prompt_runtime.py"),
}


def test_internal_runtime_code_does_not_import_prompt_runtime_support() -> None:
    violations: list[str] = []

    for path in sorted(ROOT.rglob("*.py")):
        if path in ALLOWED_FILES:
            continue

        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue

            if node.module == "victor.agent.services.prompt_runtime_support":
                violations.append(f"{path}:{node.lineno} imports PromptRuntimeSupport directly")

            if node.module == "victor.agent.services.system_prompt_runtime" and any(
                alias.name == "SystemPromptCoordinator" for alias in node.names
            ):
                violations.append(f"{path}:{node.lineno} imports SystemPromptCoordinator directly")

            if node.module in {
                "victor.agent.coordinators.factory_support",
                "victor.agent.factory.coordination_builders",
            }:
                disallowed = {
                    alias.name
                    for alias in node.names
                    if alias.name in {"create_prompt_runtime_support", "create_system_prompt_coordinator"}
                }
                for name in sorted(disallowed):
                    violations.append(f"{path}:{node.lineno} imports {name}")

            if node.module == "victor.agent.coordinators" and any(
                alias.name == "SystemPromptCoordinator" for alias in node.names
            ):
                violations.append(
                    f"{path}:{node.lineno} imports SystemPromptCoordinator from coordinators"
                )

    assert not violations, "\n".join(violations)
