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

"""Command-line interface for Victor - Open-source agentic AI framework."""

import typer
from rich.console import Console
from typing import Optional

from victor import __version__
from victor.core.utils.coding_support import load_coding_analyze_app
from victor.ui.commands.benchmark import benchmark_app
from victor.ui.commands.capabilities import capabilities_app
from victor.ui.commands.chat import chat_app, _run_default_interactive
from victor.ui.commands.config import config_app
from victor.ui.commands.doctor import run_doctor
from victor.ui.commands.dashboard import dashboard_app
from victor.ui.commands.docs import docs_app
from victor.ui.commands.observability import app as observability_app
from victor.ui.commands.embeddings import embeddings_app
from victor.ui.commands.examples import examples_app
from victor.ui.commands.experiments import experiment_app
from victor.ui.commands.fep import fep_app
from victor.ui.commands.index import index_app
from victor.ui.commands.init import init_app
from victor.ui.commands.keys import keys_app
from victor.ui.commands.mcp import mcp_app
from victor.ui.commands.models import models_app
from victor.ui.commands.profiles import profiles_app
from victor.ui.commands.providers import providers_app
from victor.ui.commands.security import security_app
from victor.ui.commands.serve import serve_app
from victor.ui.commands.test_provider import test_provider_app
from victor.ui.commands.tools import tools_app
from victor.ui.commands.scaffold import scaffold_app
from victor.ui.commands.scheduler import scheduler_app
from victor.ui.commands.sessions import sessions_app
from victor.ui.commands.db import db_app
from victor.ui.commands.vertical import vertical_app
from victor.ui.commands.plugin import plugin_app
from victor.ui.commands.skills import skills_app
from victor.ui.commands.workflow import workflow_app

# Import new unified auth command
from victor.ui.commands.auth import auth_app

app = typer.Typer(
    name="victor",
    help="Victor - Open-source agentic AI framework with multi-provider support.",
    add_completion=False,
)

console = Console()


# Define doctor_command BEFORE registering it
@app.command("doctor")
def doctor_command(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed diagnostic output"),
    fix: bool = typer.Option(False, "--fix", "-f", help="Automatically fix common issues"),
) -> None:
    """Run system diagnostics and troubleshooting.

    Performs comprehensive system checks:
    - Python version and environment
    - Dependencies and packages
    - Provider connectivity and configuration
    - Tool availability
    - File permissions
    - Performance optimizations
    """
    import sys

    exit_code = run_doctor(verbose=verbose, fix=fix)
    raise typer.Exit(exit_code)


# Register all the subcommands
# Register unified auth command (comprehensive account management)
app.add_typer(auth_app, name="auth", help="Manage authentication and provider accounts.")
app.add_typer(benchmark_app)
app.add_typer(capabilities_app)
app.add_typer(chat_app)
app.add_typer(config_app)
app.add_typer(dashboard_app)
app.add_typer(docs_app)
app.add_typer(observability_app, name="observability")
app.add_typer(embeddings_app)
app.add_typer(examples_app)
app.add_typer(experiment_app)
app.add_typer(fep_app)
app.add_typer(index_app)
app.add_typer(init_app)
app.add_typer(keys_app)
app.add_typer(mcp_app)
app.add_typer(models_app)
app.add_typer(profiles_app)
app.add_typer(providers_app)
app.add_typer(security_app)
app.add_typer(serve_app)
app.add_typer(test_provider_app)
app.add_typer(tools_app)
# Register scaffold_app as "vertical scaffold" subcommand under vertical_app
vertical_app.add_typer(scaffold_app, name="scaffold")
app.add_typer(sessions_app)
app.add_typer(db_app)
app.add_typer(vertical_app)
app.add_typer(plugin_app)
app.add_typer(skills_app)
# Register workflow_app
app.add_typer(workflow_app)

# Create coding sub-command for plugin registration
coding_app = typer.Typer(name="coding", help="Coding vertical specialized commands.")
app.add_typer(coding_app)


def _register_plugin_commands():
    """Discover and register CLI commands from plugins."""
    try:
        from victor.core.plugins.registry import PluginRegistry
        from victor.core.bootstrap import ensure_bootstrapped

        # Ensure the framework is bootstrapped, which triggers plugin registration
        ensure_bootstrapped()

        registry = PluginRegistry.get_instance()

        # 1. Register commands from PluginContext (New method)
        analyze_registered = False
        if registry.context:
            for name, sub_app in registry.context.commands.items():
                try:
                    if name == "analyze":
                        analyze_registered = True
                        # Register under both coding subcommand AND top level
                        coding_app.add_typer(sub_app, name=name)
                        app.add_typer(sub_app, name=name)
                    else:
                        # Register command under its specified name at top level
                        app.add_typer(sub_app, name=name)
                except Exception as e:
                    console.print(f"[yellow]Warning:[/] Failed to load CLI command '{name}': {e}")

        # 2. Register apps from get_cli_app (Legacy method)
        for plugin in registry.discover():
            if plugin.name == "coding":
                # Skip coding as it's already registered via the new method
                continue

            try:
                sub_app = plugin.get_cli_app()
                if sub_app:
                    # Legacy hook often doesn't specify name, uses app name or plugin name
                    app.add_typer(sub_app)
            except Exception as e:
                console.print(
                    f"[yellow]Warning:[/] Failed to load legacy CLI app from plugin '{plugin.name}': {e}"
                )

        # 3. Fallback: register analyze from contrib if plugin context didn't provide it
        if not analyze_registered:
            try:
                _analyze_fallback = load_coding_analyze_app()

                coding_app.add_typer(_analyze_fallback, name="analyze")
                app.add_typer(_analyze_fallback, name="analyze")
            except (ImportError, Exception):
                pass  # Neither plugin nor contrib available
    except Exception:
        pass


# Register plugin commands after builtin ones
_register_plugin_commands()


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
    """Victor - Open-source agentic AI framework with multi-provider support."""
    if ctx.invoked_subcommand is None:
        _run_default_interactive()


if __name__ == "__main__":
    app()
