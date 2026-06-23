from types import SimpleNamespace
from victor.agent.tool_catalog_loader import ToolCatalogLoader
from victor.agent.tool_registrar import ToolRegistrar, ToolRegistrarConfig
from victor.tools.base import BaseTool, ToolResult
from victor.tools.batch_registration import BatchRegistrar
from victor.tools.decorators import tool
from victor.tools.registry import ToolRegistry


class DummyTool(BaseTool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "dummy tool"

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, _exec_ctx, **_kwargs):
        return ToolResult(success=True, output={"ok": True})


class FakeSharedRegistry:
    def __init__(self):
        self.bootstrap_calls = 0
        self.full_calls = 0
        self.ensure_calls = []

    def get_bootstrap_tools_for_registration(self, airgapped_mode=False):
        self.bootstrap_calls += 1
        return [DummyTool("read"), DummyTool("shell")]

    def get_all_tools_for_registration(self, airgapped_mode=False):
        self.full_calls += 1
        return [DummyTool("read"), DummyTool("shell"), DummyTool("graph")]

    def get_tools_for_names(self, tool_names, airgapped_mode=False):
        names = list(tool_names)
        self.ensure_calls.append(names)
        return [DummyTool(name) for name in names]

    def infer_demand_tools(self, text: str):
        return ["graph"] if "graph" in text.lower() else []


def test_catalog_loader_bootstrap_does_not_full_discover(monkeypatch):
    fake_shared = FakeSharedRegistry()
    monkeypatch.setattr(
        "victor.agent.shared_tool_registry.SharedToolRegistry.get_instance",
        lambda: fake_shared,
    )

    loader = ToolCatalogLoader(
        registry=ToolRegistry(),
        settings=SimpleNamespace(load_tool_config=lambda: {}),
    )
    result = loader.load_bootstrap()

    assert result.tools_loaded == 2
    assert result.full_catalog_loaded is False
    assert fake_shared.bootstrap_calls == 1
    assert fake_shared.full_calls == 0


def test_batch_registration_accepts_decorated_tool_functions():
    registry = ToolRegistry()

    @tool(name="decorated_batch_test")
    async def decorated_batch_test(value: str = "ok"):
        """Decorated test tool."""
        return value

    result = BatchRegistrar(registry).register_batch([decorated_batch_test])

    assert result.success_count == 1
    assert result.failed == []
    assert registry.get("decorated_batch_test") is not None


def test_registrar_hydrates_graph_on_query_demand(monkeypatch):
    fake_shared = FakeSharedRegistry()
    monkeypatch.setattr(
        "victor.agent.shared_tool_registry.SharedToolRegistry.get_instance",
        lambda: fake_shared,
    )
    registry = ToolRegistry()
    registrar = ToolRegistrar(
        tools=registry,
        settings=SimpleNamespace(load_tool_config=lambda: {}, use_mcp_tools=False),
        config=ToolRegistrarConfig(enable_plugins=False, lazy_startup=True),
    )
    registrar.register_default_tools()

    assert registry.get("graph") is None
    loaded = registrar.ensure_tools_for_query("inspect the dependency graph")

    assert loaded == 1
    assert registry.get("graph") is not None
    assert fake_shared.ensure_calls == [["graph"]]


def test_chat_file_watchers_are_demand_driven_by_default(monkeypatch):
    from victor.ui.commands.chat import _should_start_file_watchers_on_startup

    monkeypatch.delenv("VICTOR_CHAT_FILE_WATCHERS", raising=False)
    assert _should_start_file_watchers_on_startup(SimpleNamespace()) is False

    settings = SimpleNamespace(chat_file_watchers_on_startup=True)
    assert _should_start_file_watchers_on_startup(settings) is True

    monkeypatch.setenv("VICTOR_CHAT_FILE_WATCHERS", "0")
    assert _should_start_file_watchers_on_startup(settings) is False

    monkeypatch.setenv("VICTOR_CHAT_FILE_WATCHERS", "1")
    assert _should_start_file_watchers_on_startup(SimpleNamespace()) is True


def test_shared_registry_bootstrap_exposes_grouped_command_tools():
    from victor.agent.shared_tool_registry import SharedToolRegistry

    SharedToolRegistry.reset_instance()
    try:
        registry = SharedToolRegistry.get_instance()
        tools = registry.get_bootstrap_tools_for_registration()
        names = {getattr(tool, "name", "") for tool in tools}
    finally:
        SharedToolRegistry.reset_instance()

    assert {"fs", "search", "code", "web", "shell"}.issubset(names)
    assert "git" not in names
    assert "pr" not in names
