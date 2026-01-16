# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Command-line interface for Victor - Open-source AI coding assistant."""

import warnings

# Suppress pydantic warnings from third-party libraries (lancedb, etc.)
# These warnings are not under our control and don't affect functionality
warnings.filterwarnings(
    "ignore",
    message='Field "model_" has conflict with protected namespace "model_"',
    category=UserWarning,
)

import typer
from rich.console import Console
from typing import Optional, Any, Callable
import importlib

from victor import __version__


class LazyTyper:
    """Lazy loading wrapper for Typer subcommands.

    This class defers importing command modules until they are actually needed,
    significantly improving CLI startup time.
    """

    def __init__(self, import_path: str) -> None:
        """Initialize lazy loader.

        Args:
            import_path: Module path in format "module.submodule:attribute"
        """
        self.import_path = import_path
        self._module = None
        self._typer_instance = None

    def _load(self) -> typer.Typer:
        """Load the actual Typer instance on first access."""
        if self._typer_instance is None:
            module_path, attr_name = self.import_path.split(":")
            module = importlib.import_module(module_path)
            self._typer_instance = getattr(module, attr_name)
        return self._typer_instance

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the wrapped Typer instance."""
        return getattr(self._load(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy callable to the wrapped Typer instance."""
        return self._load()(*args, **kwargs)


app = typer.Typer(
    name="victor",
    help="Victor - Open-source AI coding assistant with multi-provider support.",
    add_completion=False,
)

# Lazy load all commands to improve startup time
# Commands are only imported when actually invoked
# Format: LazyTyper("module.path:typer_instance")
app.add_typer(LazyTyper("victor.ui.commands.benchmark:benchmark_app"))
app.add_typer(LazyTyper("victor.ui.commands.capabilities:capabilities_app"))
app.add_typer(LazyTyper("victor.ui.commands.chat:chat_app"))
app.add_typer(LazyTyper("victor.ui.commands.checkpoint:checkpoint_app"))
app.add_typer(LazyTyper("victor.ui.commands.config:config_app"))
app.add_typer(LazyTyper("victor.ui.commands.dashboard:dashboard_app"))
app.add_typer(LazyTyper("victor.ui.commands.docs:docs_app"))
app.add_typer(LazyTyper("victor.ui.commands.embeddings:embeddings_app"))
app.add_typer(LazyTyper("victor.ui.commands.errors:errors_app"))
app.add_typer(LazyTyper("victor.ui.commands.examples:examples_app"))
app.add_typer(LazyTyper("victor.ui.commands.experiments:experiment_app"))
app.add_typer(LazyTyper("victor.ui.commands.fep:fep_app"))
app.add_typer(LazyTyper("victor.ui.commands.index:index_app"))
app.add_typer(LazyTyper("victor.ui.commands.init:init_app"))
app.add_typer(LazyTyper("victor.ui.commands.keys:keys_app"))
app.add_typer(LazyTyper("victor.ui.commands.mcp:mcp_app"))
app.add_typer(LazyTyper("victor.ui.commands.memory:memory_app"))
app.add_typer(LazyTyper("victor.ui.commands.models:models_app"))
app.add_typer(LazyTyper("victor.ui.commands.profiles:profiles_app"))
app.add_typer(LazyTyper("victor.ui.commands.providers:providers_app"))
app.add_typer(LazyTyper("victor.ui.commands.rag:rag_app"))
app.add_typer(LazyTyper("victor.ui.commands.security:security_app"))
app.add_typer(LazyTyper("victor.ui.commands.serve:serve_app"))
app.add_typer(LazyTyper("victor.ui.commands.test_provider:test_provider_app"))
app.add_typer(LazyTyper("victor.ui.commands.tools:tools_app"))
app.add_typer(LazyTyper("victor.ui.commands.scheduler:scheduler_app"))
app.add_typer(LazyTyper("victor.ui.commands.sessions:sessions_app"))
vertical_lazy = LazyTyper("victor.ui.commands.vertical:vertical_app")
app.add_typer(vertical_lazy)
app.add_typer(LazyTyper("victor.ui.commands.workflow:workflow_app"))

# Add scaffold as subcommand of vertical
# We need to import it and add it to the vertical app
# This is deferred until vertical is actually loaded
def _ensure_scaffold_registered():
    """Ensure scaffold is registered as a subcommand of vertical.

    This is called lazily when vertical command is first accessed.
    """
    from victor.ui.commands.vertical import vertical_app
    from victor.ui.commands.scaffold import scaffold_app

    # Only add if not already added
    if not any(cmd.name == "scaffold" for cmd in vertical_app.registered_commands):
        vertical_app.add_typer(scaffold_app, name="scaffold")


# Monkey-patch LazyTyper to register scaffold on first load
original_load = vertical_lazy._load


def _load_with_scaffold(self):
    """Load vertical app and ensure scaffold is registered."""
    result = original_load()
    _ensure_scaffold_registered()
    return result


vertical_lazy._load = lambda: _load_with_scaffold(vertical_lazy)


def _get_default_interactive():
    """Lazy load the default interactive function."""
    from victor.ui.commands.chat import _run_default_interactive
    return _run_default_interactive


console = Console()


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"Victor v{__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """Victor - Open-source AI coding assistant with multi-provider support."""
    if ctx.invoked_subcommand is None:
        _get_default_interactive()()


if __name__ == "__main__":
    app()
