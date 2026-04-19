from __future__ import annotations

from types import ModuleType

from victor.ui.commands.workflow import _load_registered_handlers


def test_load_registered_handlers_supports_host_injected_registrar(monkeypatch) -> None:
    fake_module = ModuleType("victor.fake.handlers")
    fake_module.HANDLERS = {"alpha": object()}
    captured = []

    def register_handlers(registrar):
        registrar("alpha", fake_module.HANDLERS["alpha"])
        captured.append("registered")

    fake_module.register_handlers = register_handlers

    def fake_import_module(name: str):
        assert name == "victor.fake.handlers"
        return fake_module

    monkeypatch.setattr("importlib.import_module", fake_import_module)
    monkeypatch.setattr(
        "victor.workflows.executor.register_compute_handler",
        lambda name, handler: captured.append((name, handler)),
    )
    monkeypatch.setattr(
        "victor.workflows.executor.list_compute_handlers",
        lambda: ["alpha"],
    )

    handlers = _load_registered_handlers("fake")

    assert "alpha" in handlers
    assert captured[0] == ("alpha", fake_module.HANDLERS["alpha"])
    assert captured[1] == "registered"
