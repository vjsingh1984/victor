# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Architecture guardrails for the canonical chat runtime path."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text()


def _method_source(relative_path: str, class_name: str, method_name: str) -> str:
    source = _read(relative_path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == method_name
                ):
                    segment = ast.get_source_segment(source, child)
                    if segment is None:
                        raise AssertionError(
                            f"Could not recover source for {class_name}.{method_name} in {relative_path}"
                        )
                    return segment
    raise AssertionError(f"Method {class_name}.{method_name} not found in {relative_path}")


def _function_source(relative_path: str, function_name: str) -> str:
    source = _read(relative_path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise AssertionError(
                    f"Could not recover source for function {function_name} in {relative_path}"
                )
            return segment
    raise AssertionError(f"Function {function_name} not found in {relative_path}")


def test_orchestrator_public_chat_entrypoints_delegate_to_chat_service() -> None:
    source = _read("victor/agent/orchestrator.py")
    assert "async def _chat_via_agentic_loop" not in source

    chat_source = _method_source("victor/agent/orchestrator.py", "AgentOrchestrator", "chat")
    assert "_chat_service.chat(" in chat_source
    assert "AgenticLoop" not in chat_source

    stream_source = _method_source(
        "victor/agent/orchestrator.py", "AgentOrchestrator", "stream_chat"
    )
    assert "_chat_service.stream_chat(" in stream_source
    assert "AgenticLoop" not in stream_source
    assert "_get_service_streaming_runtime" not in stream_source


def test_chat_service_no_longer_owns_loop_execution() -> None:
    chat_source = _method_source("victor/agent/services/chat_service.py", "ChatService", "chat")
    stream_source = _method_source(
        "victor/agent/services/chat_service.py", "ChatService", "stream_chat"
    )
    loop_guard_source = _method_source(
        "victor/agent/services/chat_service.py", "ChatService", "_run_agentic_loop"
    )

    assert "_run_agentic_loop(" not in chat_source
    assert "_run_agentic_loop(" not in stream_source
    assert "raise RuntimeError" in loop_guard_source
    assert "while " not in loop_guard_source
    assert "_get_completion(" not in loop_guard_source


def test_deprecated_chat_shims_do_not_materialize_local_chat_loops() -> None:
    chat_source = _method_source("victor/agent/services/chat_compat.py", "ChatCoordinator", "chat")
    chat_planning_source = _method_source(
        "victor/agent/services/chat_compat.py", "ChatCoordinator", "_chat_with_planning"
    )
    chat_stream_source = _method_source(
        "victor/agent/services/chat_compat.py", "ChatCoordinator", "stream_chat"
    )
    turn_executor_source = _method_source(
        "victor/agent/services/chat_compat.py", "ChatCoordinator", "turn_executor"
    )
    sync_source = _method_source(
        "victor/agent/services/sync_chat_compat.py", "SyncChatCoordinator", "chat"
    )
    sync_planning_source = _method_source(
        "victor/agent/services/sync_chat_compat.py", "SyncChatCoordinator", "_chat_with_planning"
    )
    streaming_source = _method_source(
        "victor/agent/services/streaming_chat_compat.py",
        "StreamingChatCoordinator",
        "stream_chat",
    )
    unified_chat_source = _method_source(
        "victor/agent/services/unified_chat_compat.py", "UnifiedChatCoordinator", "chat"
    )
    unified_stream_source = _method_source(
        "victor/agent/services/unified_chat_compat.py", "UnifiedChatCoordinator", "stream_chat"
    )

    assert "execute_agentic_loop(" not in chat_source
    assert "PlanningCoordinator(" not in chat_planning_source
    assert "_get_service_streaming_runtime" not in chat_stream_source
    assert "_stream_chat_runtime" not in chat_stream_source
    assert "_get_orchestrator_runtime_property(" not in turn_executor_source
    assert "materializing a legacy local" not in turn_executor_source
    assert "execute_agentic_loop(" not in sync_source
    assert "execute_agentic_loop(" not in sync_planning_source
    assert "_stream_from_provider(" not in streaming_source
    assert "raise RuntimeError" in streaming_source
    assert "_sync.chat(" not in unified_chat_source
    assert "_streaming.stream_chat(" not in unified_chat_source
    assert "_streaming.stream_chat(" not in unified_stream_source
    assert "raise RuntimeError" in unified_chat_source
    assert "raise RuntimeError" in unified_stream_source


def test_deprecated_sync_chat_coordinator_is_not_wired_to_turn_executor() -> None:
    source = _function_source(
        "victor/agent/orchestrator_properties.py",
        "_ensure_sync_chat_coordinator",
    )
    assert "turn_executor=self.turn_executor" not in source


def test_deprecated_unified_chat_coordinator_is_not_wired_to_nested_chat_shims() -> None:
    source = _function_source(
        "victor/agent/orchestrator_properties.py",
        "_ensure_unified_chat_coordinator",
    )
    assert "_ensure_sync_chat_coordinator(self)" not in source
    assert "_ensure_streaming_chat_coordinator(self)" not in source
