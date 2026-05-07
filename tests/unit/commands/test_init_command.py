# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the init command."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

import victor.ui.commands.init as init_module
from victor.ui.commands.init_content import (
    count_architecture_patterns,
    ensure_architecture_evidence_section,
    ensure_architecture_patterns_section,
)


def test_init_ccg_reports_project_graph_stats_without_name_error(tmp_path, monkeypatch):
    """CCG init should report graph stats from the project graph store."""
    output = StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    monkeypatch.setattr(init_module, "console", test_console)

    project_paths = SimpleNamespace(
        global_victor_dir=tmp_path / "global-victor",
        project_context_file=tmp_path / "init.md",
        project_victor_dir=tmp_path / ".victor",
    )
    monkeypatch.setattr(init_module, "get_project_paths", lambda: project_paths)
    monkeypatch.setattr(
        init_module,
        "get_database",
        lambda: SimpleNamespace(db_path=project_paths.global_victor_dir / "victor.db"),
    )
    monkeypatch.setattr(
        init_module,
        "get_project_database",
        lambda: SimpleNamespace(db_path=project_paths.project_victor_dir / "project.db"),
    )
    monkeypatch.setattr(init_module, "latest_mtime", lambda _root: 0.0)
    monkeypatch.setattr(
        init_module,
        "ensure_project_graph_enriched",
        lambda *_args, **_kwargs: SimpleNamespace(
            total_edges=0,
            implements_edges=0,
            decorates_edges=0,
            registers_edges=0,
        ),
    )
    monkeypatch.setattr(init_module, "_generate_init_content", lambda **_kwargs: "# init\n")
    monkeypatch.setattr(init_module, "_gather_graph_context", AsyncMock(return_value=None))

    fake_graph_store = SimpleNamespace(stats=AsyncMock(return_value={"nodes": 12, "edges": 34}))
    create_graph_store = MagicMock(return_value=fake_graph_store)

    fake_graph_rag = ModuleType("victor.core.graph_rag")
    fake_graph_rag.GraphIndexConfig = lambda **kwargs: SimpleNamespace(**kwargs)

    class FakePipeline:
        def __init__(self, graph_store, config) -> None:
            self.graph_store = graph_store
            self.config = config

        async def index_repository(self):
            return SimpleNamespace(
                files_processed=1,
                files_deleted=0,
                files_unchanged=2,
                nodes_created=3,
                edges_created=4,
            )

    fake_graph_rag.GraphIndexingPipeline = FakePipeline

    with patch.dict(sys.modules, {"victor.core.graph_rag": fake_graph_rag}):
        with patch("victor.storage.graph.create_graph_store", create_graph_store):
            with patch("victor.ui.commands.utils.setup_logging"):
                init_module.init(
                    ctx=SimpleNamespace(invoked_subcommand=None),
                    update=False,
                    force=False,
                    learn=False,
                    index=False,
                    deep=False,
                    quick=False,
                    ccg=True,
                    symlinks=False,
                    config_only=False,
                    interactive=False,
                    local=False,
                    airgapped=False,
                    wizard=False,
                    provider=None,
                    model=None,
                    log_level=None,
                )

    create_graph_store.assert_called_once_with("sqlite", project_path=Path.cwd())
    fake_graph_store.stats.assert_awaited_once()

    rendered = output.getvalue()
    assert "CCG index updated" in rendered
    assert "12 total nodes, 34 total edges in database" in rendered
    assert "CCG indexing skipped" not in rendered


async def test_generate_init_content_async_closes_temporary_agent(monkeypatch):
    """Enhanced init should close the temporary synthesis agent."""
    mock_agent = SimpleNamespace(close=AsyncMock())
    mock_generator = AsyncMock(return_value="# generated\n")

    monkeypatch.setattr(init_module, "_create_init_agent", AsyncMock(return_value=mock_agent))
    monkeypatch.setattr(
        init_module,
        "load_codebase_analyzer_attr",
        lambda name: mock_generator if name == "generate_enhanced_init_md" else None,
    )

    result = await init_module._generate_init_content_async(
        mode="enhanced",
        use_llm=True,
        include_conversations=False,
        provider="zai-coding",
        model=None,
    )

    assert result == "# generated\n"
    mock_agent.close.assert_awaited_once()
    assert mock_generator.await_count == 1


async def test_create_init_agent_uses_lightweight_profile_provider(monkeypatch):
    """CLI init should not build a full orchestrator just to synthesize init.md."""
    fake_provider = SimpleNamespace(close=AsyncMock(), name="zai")
    mock_settings = SimpleNamespace(
        load_profiles=lambda: {"zai-coding": SimpleNamespace(provider="zai", model="glm-5.1")}
    )

    monkeypatch.setattr(init_module, "load_settings", lambda: mock_settings, raising=False)

    with patch("victor.config.settings.load_settings", return_value=mock_settings):
        with patch("victor.providers.registry.ProviderRegistry.create", return_value=fake_provider):
            agent = await init_module._create_init_agent("zai-coding")

    assert agent.provider is fake_provider
    assert agent.provider_name == "zai"
    assert agent.model == "glm-5.1"


def test_count_architecture_patterns_reads_markdown_section() -> None:
    content = """# init.md

## Architecture Patterns

- **Facade**: One public entry point
- **Registry**: Pluggable extensions

## Development Commands
"""

    assert count_architecture_patterns(content) == 2


def test_ensure_architecture_patterns_section_adds_graph_fallback() -> None:
    content = """# init.md

## Development Commands

```bash
make test
```
"""
    graph_context = {
        "has_ccg": True,
        "stats": {"ccg_edges": 42},
        "patterns": {
            "registry": 3,
            "protocol": 5,
            "decorator": 0,
            "inheritance": 7,
        },
    }

    enhanced = ensure_architecture_patterns_section(content, graph_context)

    assert "## Architecture Patterns" in enhanced
    assert "Registry/plugin extensibility" in enhanced
    assert "Protocol/interface contracts" in enhanced
    assert count_architecture_patterns(enhanced) >= 3


def test_ensure_architecture_evidence_section_adds_graph_backed_summary() -> None:
    content = """# init.md

## Architecture Patterns

- **Facade**: One entry point
"""
    graph_context = {
        "has_ccg": True,
        "stats": {
            "total_nodes": 1200,
            "total_edges": 3400,
            "ccg_edges": 240,
        },
        "patterns": {
            "registry": 4,
            "protocol": 6,
            "decorator": 2,
            "inheritance": 9,
        },
        "complexity": {
            "avg_branching": 2.83,
        },
    }

    enhanced = ensure_architecture_evidence_section(content, graph_context)

    assert "## Architecture Evidence" in enhanced
    assert "Graph scale" in enhanced
    assert "Registry/plugin evidence" in enhanced
    assert "Protocol/interface evidence" in enhanced
    assert "Statement-level flow evidence" in enhanced


def test_init_reports_architecture_patterns_from_markdown_section(tmp_path, monkeypatch):
    output = StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    monkeypatch.setattr(init_module, "console", test_console)

    project_paths = SimpleNamespace(
        global_victor_dir=tmp_path / "global-victor",
        project_context_file=tmp_path / "init.md",
        project_victor_dir=tmp_path / ".victor",
    )
    monkeypatch.setattr(init_module, "get_project_paths", lambda: project_paths)
    monkeypatch.setattr(
        init_module,
        "get_database",
        lambda: SimpleNamespace(db_path=project_paths.global_victor_dir / "victor.db"),
    )
    monkeypatch.setattr(
        init_module,
        "get_project_database",
        lambda: SimpleNamespace(db_path=project_paths.project_victor_dir / "project.db"),
    )
    monkeypatch.setattr(init_module, "latest_mtime", lambda _root: 0.0)

    generated = """# init.md

## Architecture Patterns

- **Facade**: One entry point
- **Registry**: Shared extension lookup
"""
    monkeypatch.setattr(init_module, "_generate_init_content", lambda **_kwargs: generated)

    with patch("victor.ui.commands.utils.setup_logging"):
        init_module.init(
            ctx=SimpleNamespace(invoked_subcommand=None),
            update=False,
            force=False,
            learn=False,
            index=False,
            deep=False,
            quick=True,
            ccg=True,
            symlinks=False,
            config_only=False,
            interactive=False,
            local=False,
            airgapped=False,
            wizard=False,
            provider=None,
            model=None,
            log_level=None,
        )

    rendered = output.getvalue()
    assert "Found 2 architecture patterns" in rendered


def test_init_enriches_content_with_architecture_evidence(tmp_path, monkeypatch):
    output = StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    monkeypatch.setattr(init_module, "console", test_console)

    project_paths = SimpleNamespace(
        global_victor_dir=tmp_path / "global-victor",
        project_context_file=tmp_path / "init.md",
        project_victor_dir=tmp_path / ".victor",
    )
    monkeypatch.setattr(init_module, "get_project_paths", lambda: project_paths)
    monkeypatch.setattr(
        init_module,
        "get_database",
        lambda: SimpleNamespace(db_path=project_paths.global_victor_dir / "victor.db"),
    )
    monkeypatch.setattr(
        init_module,
        "get_project_database",
        lambda: SimpleNamespace(db_path=project_paths.project_victor_dir / "project.db"),
    )
    monkeypatch.setattr(init_module, "latest_mtime", lambda _root: 0.0)
    monkeypatch.setattr(
        init_module,
        "ensure_project_graph_enriched",
        lambda *_args, **_kwargs: SimpleNamespace(
            total_edges=0,
            implements_edges=0,
            decorates_edges=0,
            registers_edges=0,
        ),
    )
    monkeypatch.setattr(
        init_module,
        "_generate_init_content",
        lambda **_kwargs: "# init.md\n\n## Architecture Patterns\n\n- **Facade**: One entry point\n",
    )
    monkeypatch.setattr(
        init_module,
        "_gather_graph_context",
        AsyncMock(
            return_value={
                "has_ccg": True,
                "stats": {
                    "total_nodes": 100,
                    "total_edges": 250,
                    "ccg_edges": 50,
                },
                "patterns": {
                    "registry": 2,
                    "protocol": 3,
                    "decorator": 0,
                    "inheritance": 4,
                },
                "complexity": {"avg_branching": 2.5},
            }
        ),
    )

    with patch("victor.ui.commands.utils.setup_logging"):
        init_module.init(
            ctx=SimpleNamespace(invoked_subcommand=None),
            update=False,
            force=False,
            learn=False,
            index=False,
            deep=True,
            quick=False,
            ccg=False,
            symlinks=False,
            config_only=False,
            interactive=False,
            local=False,
            airgapped=False,
            wizard=False,
            provider=None,
            model=None,
            log_level=None,
        )

    written = project_paths.project_context_file.read_text(encoding="utf-8")
    assert "## Architecture Evidence" in written
    assert "Registry/plugin evidence" in written
