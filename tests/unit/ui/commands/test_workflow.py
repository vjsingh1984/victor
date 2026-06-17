from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import typer

from victor.ui.commands.workflow import (
    _load_registered_handlers,
    run_workflow,
    validate_workflow,
)
from victor.workflows.definition import TransformNode, WorkflowDefinition


def test_load_registered_handlers_supports_host_injected_registrar(monkeypatch) -> None:
    fake_module = ModuleType("victor.fake.handlers")
    fake_module.HANDLERS = {"alpha": object()}
    captured = []

    def register_handlers(registrar):
        registrar("alpha", fake_module.HANDLERS["alpha"])
        captured.append("registered")

    fake_module.register_handlers = register_handlers

    import importlib

    real_import_module = importlib.import_module

    def fake_import_module(name: str, *args, **kwargs):
        # Only stub the vertical handler module; delegate everything else to the
        # real importer. pytest's own monkeypatch.setattr(<dotted string>) calls
        # importlib.import_module internally to resolve the target, so a fake
        # that hard-asserts the name breaks unrelated resolution (e.g. it
        # resolves "victor" first) — non-deterministically, depending on what is
        # already cached in sys.modules.
        if name == "victor.fake.handlers":
            return fake_module
        return real_import_module(name, *args, **kwargs)

    # Patch the registry targets BEFORE swapping import_module so their dotted
    # resolution uses the real importer, then install the fake importer.
    monkeypatch.setattr(
        "victor.workflows.compute_registry.register_compute_handler",
        lambda name, handler: captured.append((name, handler)),
    )
    monkeypatch.setattr(
        "victor.workflows.compute_registry.list_compute_handlers",
        lambda: ["alpha"],
    )
    monkeypatch.setattr("importlib.import_module", fake_import_module)

    handlers = _load_registered_handlers("fake")

    assert "alpha" in handlers
    assert captured[0] == ("alpha", fake_module.HANDLERS["alpha"])
    assert captured[1] == "registered"


def _build_workflow(name: str = "sample") -> WorkflowDefinition:
    return WorkflowDefinition(
        name=name,
        nodes={
            "start": TransformNode(
                id="start",
                name="Start",
                transform=lambda ctx: ctx,
            )
        },
        start_node="start",
    )


def test_validate_workflow_uses_canonical_compile_helper(tmp_path) -> None:
    workflow = _build_workflow()
    compiler = MagicMock(name="compiler")

    with (
        patch(
            "victor.ui.commands.workflow._load_workflow_file",
            return_value={workflow.name: workflow},
        ),
        patch(
            "victor.ui.commands.workflow._create_compile_only_compiler",
            return_value=compiler,
        ) as create_compiler,
        patch("victor.ui.commands.workflow.console.print"),
    ):
        validate_workflow(
            str(tmp_path / "sample.yaml"),
            verbose=False,
            check_handlers=False,
            check_escape_hatches=True,
            vertical=None,
        )

    create_compiler.assert_called_once_with()
    compiler.compile_definition.assert_called_once_with(workflow)


def test_run_workflow_dry_run_uses_canonical_compile_helper(tmp_path) -> None:
    workflow = _build_workflow()
    compiler = MagicMock(name="compiler")

    with (
        patch(
            "victor.ui.commands.workflow._load_workflow_file",
            return_value={workflow.name: workflow},
        ),
        patch(
            "victor.ui.commands.workflow._create_compile_only_compiler",
            return_value=compiler,
        ) as create_compiler,
        patch("victor.ui.commands.workflow.console.print"),
        patch("victor.ui.commands.workflow.run_sync") as run_sync,
    ):
        run_workflow(
            str(tmp_path / "sample.yaml"),
            context=None,
            context_file=None,
            delegate_follow_up_contract=None,
            delegate_next_step_id=None,
            workflow_name=None,
            profile=None,
            dry_run=True,
            log_level=None,
        )

    create_compiler.assert_called_once_with()
    compiler.compile_definition.assert_called_once_with(workflow)
    run_sync.assert_not_called()


def test_validate_workflow_exits_on_canonical_compile_failure(tmp_path) -> None:
    workflow = _build_workflow()
    compiler = MagicMock(name="compiler")
    compiler.compile_definition.side_effect = ValueError("boom")

    with (
        patch(
            "victor.ui.commands.workflow._load_workflow_file",
            return_value={workflow.name: workflow},
        ),
        patch(
            "victor.ui.commands.workflow._create_compile_only_compiler",
            return_value=compiler,
        ),
        patch("victor.ui.commands.workflow.console.print"),
        pytest.raises(typer.Exit) as exc_info,
    ):
        validate_workflow(
            str(tmp_path / "sample.yaml"),
            verbose=False,
            check_handlers=False,
            check_escape_hatches=True,
            vertical=None,
        )

    assert exc_info.value.exit_code == 1
