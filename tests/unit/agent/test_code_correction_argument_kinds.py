# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""FEP-0024 Phase 3: per-argument kind trait gates correction precisely.

A tool can declare ``argument_kinds`` on its ``ToolContract`` to mark which argument
is executable code (eligible for auto-fix) vs file content (never mutated), overriding
the tool-level ``access_mode`` proxy from #622. This lets a mixed tool (e.g. a notebook
runner) expose ``snippet`` as executable while keeping ``content`` as a document.
"""

import pytest

from victor.agent.code_correction_middleware import CodeCorrectionMiddleware
from victor_contracts.tools import AccessMode, ArgumentKind, ToolContract


class _Tool:
    """Minimal tool stub exposing contract traits like a decorated tool."""

    def __init__(self, access_mode="write", argument_kinds=()):
        self.access_mode = access_mode
        self.argument_kinds = argument_kinds


@pytest.fixture
def middleware():
    from victor.evaluation.correction import CodeValidatorRegistry

    CodeValidatorRegistry.reset_singleton()
    return CodeCorrectionMiddleware()


def test_argument_kind_enum_members():
    assert ArgumentKind.EXECUTABLE_CODE.value == "executable_code"
    assert ArgumentKind.FILE_CONTENT.value == "file_content"
    assert ArgumentKind.DATA.value == "data"


def test_tool_contract_argument_kinds_frozen_hashable():
    contract = ToolContract(
        access_mode=AccessMode.EXECUTE,
        argument_kinds=(
            ("code", ArgumentKind.EXECUTABLE_CODE),
            ("content", ArgumentKind.FILE_CONTENT),
        ),
    )
    assert dict(contract.argument_kinds)["code"] is ArgumentKind.EXECUTABLE_CODE
    hash(contract)  # frozen dataclass stays hashable (tuple-of-pairs, not dict)


def test_should_validate_eligible_via_executable_code_arg(middleware):
    """access_mode is WRITE (not EXECUTE) but an executable-code arg is declared."""
    tool = _Tool(
        access_mode="write",
        argument_kinds=(("snippet", ArgumentKind.EXECUTABLE_CODE),),
    )
    assert middleware.should_validate("notebook_run", tool=tool) is True


def test_should_validate_not_eligible_when_only_file_content(middleware):
    tool = _Tool(
        access_mode="write",
        argument_kinds=(("content", ArgumentKind.FILE_CONTENT),),
    )
    assert middleware.should_validate("notebook_run", tool=tool) is False


def test_find_code_argument_uses_executable_code_kind(middleware):
    tool = _Tool(
        argument_kinds=(
            ("snippet", ArgumentKind.EXECUTABLE_CODE),
            ("content", ArgumentKind.FILE_CONTENT),
        )
    )
    args = {"snippet": "print('hi')", "content": "# doc", "path": "x.ipynb"}
    name, value = middleware.find_code_argument(args, tool=tool)
    assert name == "snippet"
    assert value == "print('hi')"


def test_find_code_argument_skips_file_content_only(middleware):
    tool = _Tool(argument_kinds=(("content", ArgumentKind.FILE_CONTENT),))
    args = {"content": "# doc"}
    assert middleware.find_code_argument(args, tool=tool) is None


def test_process_corrects_executable_arg_leaves_file_content(middleware):
    tool = _Tool(
        access_mode="write",
        argument_kinds=(
            ("snippet", ArgumentKind.EXECUTABLE_CODE),
            ("content", ArgumentKind.FILE_CONTENT),
        ),
    )
    fenced = "```python\nprint(1)\n```"
    doc = "# a markdown doc\n\n```text\ncmd\n```\n"
    args = {"snippet": fenced, "content": doc}

    out_args, result = middleware.process("notebook_run", args, tool=tool)

    assert result is not None and result.was_corrected is True
    assert "```" not in out_args["snippet"]  # executable arg unwrapped/corrected
    assert out_args["content"] == doc  # file content untouched


def test_undeclared_tool_falls_back_to_access_mode(middleware):
    """No argument_kinds -> #622 behavior: access_mode gate + name allowlist."""
    assert middleware.should_validate("code_executor", tool=_Tool(access_mode="execute")) is True
    assert middleware.should_validate("write", tool=_Tool(access_mode="write")) is False
