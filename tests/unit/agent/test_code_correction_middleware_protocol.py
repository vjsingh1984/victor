# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""FEP-0024 Phase 2: CodeCorrectionMiddleware is a MiddlewareProtocol member.

The framework's default correction runs in the ToolExecutor (the choke point). This
makes the middleware a first-class ``MiddlewareProtocol`` citizen so a vertical can
register it on its own ``MiddlewareChain`` for domain interception, with an optional
``tool_resolver`` for contract-trait gating on the chain path.
"""

import pytest

from victor.agent.code_correction_middleware import CodeCorrectionMiddleware
from victor.core.verticals.protocols.middleware import MiddlewareProtocol
from victor.core.vertical_types import MiddlewareResult


@pytest.fixture
def middleware():
    from victor.evaluation.correction import CodeValidatorRegistry

    CodeValidatorRegistry.reset_singleton()
    return CodeCorrectionMiddleware()


def test_satisfies_middleware_protocol(middleware):
    """CodeCorrectionMiddleware is usable as a MiddlewareChain member."""
    assert isinstance(middleware, MiddlewareProtocol)


async def test_before_tool_call_corrects_executable_code(middleware):
    """before_tool_call unwraps/corrects an executable-code arg via process()."""
    args = {"code": "```python\nprint(1)\n```"}
    result = await middleware.before_tool_call("code_executor", args)

    assert isinstance(result, MiddlewareResult)
    assert result.modified_arguments is not None
    assert "```" not in result.modified_arguments["code"]
    assert result.metadata.get("code_corrected") is True


async def test_before_tool_call_leaves_file_content(middleware):
    """File-authoring tools (write) are not eligible; args pass through unchanged."""
    doc = "# title\n\n```text\ncmd\n```\n"
    args = {"path": "x.md", "content": doc, "force": True}
    result = await middleware.before_tool_call("write", args)

    assert result.modified_arguments is None
    assert result.metadata.get("code_corrected") in (None, False)


async def test_tool_resolver_enables_custom_executable_tool():
    """A tool_resolver lets the chain path gate a non-allowlist executable tool."""
    from victor.evaluation.correction import CodeValidatorRegistry

    CodeValidatorRegistry.reset_singleton()

    class _ExecTool:
        access_mode = "execute"

    mw = CodeCorrectionMiddleware(
        tool_resolver=lambda name: _ExecTool() if name == "my_runner" else None
    )
    args = {"code": "```python\nprint(2)\n```"}
    result = await mw.before_tool_call("my_runner", args)

    assert result.modified_arguments is not None
    assert "```" not in result.modified_arguments["code"]
