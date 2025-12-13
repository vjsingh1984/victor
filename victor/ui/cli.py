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

"""Command-line interface for Victor - Enterprise-Ready AI Coding Assistant."""

import typer
from rich.console import Console
from typing import Optional

from victor import __version__
from victor.ui.commands.chat import chat_app, _run_default_interactive
from victor.ui.commands.config import config_app
from victor.ui.commands.embeddings import embeddings_app
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

app = typer.Typer(
    name="victor",
    help="Victor - Enterprise-Ready AI Coding Assistant.",
    add_completion=False,
)

# Register all the subcommands
app.add_typer(chat_app)
app.add_typer(config_app)
app.add_typer(embeddings_app)
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
    """Victor - Enterprise-Ready AI Coding Assistant."""
    if ctx.invoked_subcommand is None:
        _run_default_interactive()


if __name__ == "__main__":
    app()
