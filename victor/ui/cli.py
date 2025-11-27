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

"""Command-line interface for Victor - Universal AI Coding Assistant."""

import asyncio
import logging
import os
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from victor import __version__
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import load_settings

# Configure logging
log_level = os.getenv("VICTOR_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="victor",
    help="Victor - Code to Victory with Any AI. Universal terminal-based coding assistant supporting multiple LLM providers.",
    add_completion=False,
)

console = Console()


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"Victor v{__version__}")
        raise typer.Exit()


@app.command()
def main(
    message: Optional[str] = typer.Argument(
        None,
        help="Message to send to the agent (starts interactive mode if not provided)",
    ),
    profile: str = typer.Option(
        "default",
        "--profile",
        "-p",
        help="Profile to use from profiles.yaml",
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Stream responses",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """Victor - Universal AI coding assistant. Code to Victory with Any AI.

    Examples:
        # Interactive mode
        victor

        # One-shot command
        victor "Write a Python function to calculate Fibonacci numbers"

        # Use specific profile
        victor --profile claude "Explain how async/await works"
    """
    # Load settings
    settings = load_settings()

    if message:
        # One-shot mode
        asyncio.run(run_oneshot(message, settings, profile, stream))
    else:
        # Interactive REPL mode
        asyncio.run(run_interactive(settings, profile, stream))


async def run_oneshot(
    message: str,
    settings: any,
    profile: str,
    stream: bool,
) -> None:
    """Run a single message and exit.

    Args:
        message: Message to send
        settings: Application settings
        profile: Profile name
        stream: Whether to stream response
    """
    try:
        agent = await AgentOrchestrator.from_settings(settings, profile)

        # Start background embedding preload to avoid blocking first query
        agent.start_embedding_preload()

        if stream and agent.provider.supports_streaming():
            async for chunk in agent.stream_chat(message):
                if chunk.content:
                    console.print(chunk.content, end="")
            console.print()  # New line at end
        else:
            response = await agent.chat(message)
            console.print(Markdown(response.content))

        await agent.provider.close()

    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(1)


async def run_interactive(
    settings: any,
    profile: str,
    stream: bool,
) -> None:
    """Run interactive REPL mode.

    Args:
        settings: Application settings
        profile: Profile name
        stream: Whether to stream responses
    """
    try:
        # Load profile info
        profiles = settings.load_profiles()
        profile_config = profiles.get(profile)

        if not profile_config:
            console.print(f"[bold red]Error:[/] Profile '{profile}' not found")
            raise typer.Exit(1)

        # Create agent
        agent = await AgentOrchestrator.from_settings(settings, profile)

        # Start background embedding preload to avoid blocking first query
        agent.start_embedding_preload()

        # Welcome message
        console.print(
            Panel(
                f"[bold]CodingAgent v{__version__}[/]\n\n"
                f"Provider: [cyan]{profile_config.provider}[/]\n"
                f"Model: [cyan]{profile_config.model}[/]\n\n"
                f"Type your message and press Enter to chat.\n"
                f"Type [bold]exit[/] or [bold]quit[/] to leave.\n"
                f"Type [bold]clear[/] to reset conversation.",
                title="Welcome",
                border_style="blue",
            )
        )

        # REPL loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold green]You[/]")

                # Handle special commands
                if user_input.lower() in ["exit", "quit"]:
                    console.print("[dim]Goodbye![/]")
                    break

                if user_input.lower() == "clear":
                    agent.reset_conversation()
                    console.print("[dim]Conversation cleared[/]")
                    continue

                if not user_input.strip():
                    continue

                # Send message and get response
                console.print("\n[bold blue]Assistant[/]")

                if stream and agent.provider.supports_streaming():
                    async for chunk in agent.stream_chat(user_input):
                        if chunk.content:
                            console.print(chunk.content, end="")
                    console.print()  # New line at end
                else:
                    response = await agent.chat(user_input)
                    console.print(Markdown(response.content))

            except KeyboardInterrupt:
                console.print("\n[dim]Use 'exit' or 'quit' to leave[/]")
                continue

            except EOFError:
                break

        await agent.provider.close()

    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def init() -> None:
    """Initialize configuration files."""
    from pathlib import Path
    import shutil

    config_dir = Path.home() / ".victor"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Copy example profiles if they don't exist
    profiles_file = config_dir / "profiles.yaml"
    if not profiles_file.exists():
        console.print(f"Creating default configuration at {profiles_file}")

        # Create a basic default profile
        default_config = """profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7
    max_tokens: 4096

providers:
  ollama:
    base_url: http://localhost:11434
"""
        profiles_file.write_text(default_config)
        console.print("[green]✓[/] Configuration created successfully!")
    else:
        console.print("[yellow]Configuration already exists[/]")

    console.print(f"\nConfiguration directory: {config_dir}")
    console.print(f"Edit {profiles_file} to customize profiles")


@app.command()
def providers() -> None:
    """List all available providers."""
    from victor.providers.registry import ProviderRegistry
    from rich.table import Table

    available_providers = ProviderRegistry.list_providers()

    table = Table(title="Available Providers", show_header=True)
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Features")

    provider_info = {
        "ollama": ("✅ Ready", "Local models, Free, Tool calling"),
        "anthropic": ("✅ Ready", "Claude, Tool calling, Streaming"),
        "openai": ("✅ Ready", "GPT-4/3.5, Function calling, Vision"),
        "google": ("✅ Ready", "Gemini, 1M context, Multimodal"),
        "xai": ("✅ Ready", "Grok, Real-time info, Vision"),
        "grok": ("✅ Ready", "Alias for xai"),
    }

    for provider in sorted(available_providers):
        status, features = provider_info.get(provider, ("❓ Unknown", ""))
        table.add_row(provider, status, features)

    console.print(table)
    console.print("\n[dim]Use 'victor profiles' to see configured profiles[/]")


@app.command()
def profiles_cmd() -> None:
    """List configured profiles."""
    from rich.table import Table

    settings = load_settings()
    profiles = settings.load_profiles()

    if not profiles:
        console.print("[yellow]No profiles configured[/]")
        console.print("Run [bold]victor init[/] to create default configuration")
        return

    table = Table(title="Configured Profiles", show_header=True)
    table.add_column("Profile", style="cyan", no_wrap=True)
    table.add_column("Provider", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Temperature")
    table.add_column("Max Tokens")

    for name, profile in profiles.items():
        table.add_row(
            name,
            profile.provider,
            profile.model,
            f"{profile.temperature}",
            f"{profile.max_tokens}",
        )

    console.print(table)
    console.print(
        f"\n[dim]Config file: {settings.get_config_dir() / 'profiles.yaml'}[/]"
    )


@app.command()
def models(
    provider: str = typer.Option(
        "ollama",
        "--provider",
        "-p",
        help="Provider to list models from",
    ),
) -> None:
    """List available models for a provider."""
    asyncio.run(list_models_async(provider))


async def list_models_async(provider: str) -> None:
    """Async function to list models."""
    from rich.table import Table

    settings = load_settings()

    try:
        # Special handling for Ollama
        if provider == "ollama":
            from victor.providers.ollama import OllamaProvider

            provider_settings = settings.get_provider_settings(provider)
            ollama = OllamaProvider(**provider_settings)

            try:
                models_list = await ollama.list_models()

                if not models_list:
                    console.print(f"[yellow]No models found for {provider}[/]")
                    console.print("\nPull a model with: [bold]ollama pull qwen2.5-coder:7b[/]")
                    return

                table = Table(title=f"Available Models ({provider})", show_header=True)
                table.add_column("Model", style="cyan", no_wrap=True)
                table.add_column("Size", style="yellow")
                table.add_column("Modified", style="dim")

                for model in models_list:
                    name = model.get("name", "unknown")
                    size = model.get("size", 0)
                    size_gb = size / (1024**3) if size else 0

                    modified = model.get("modified_at", "")
                    if modified:
                        # Format timestamp
                        from datetime import datetime
                        try:
                            dt = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                            modified = dt.strftime("%Y-%m-%d")
                        except:
                            pass

                    table.add_row(
                        name,
                        f"{size_gb:.1f} GB" if size_gb > 0 else "unknown",
                        modified,
                    )

                console.print(table)
                console.print(
                    f"\n[dim]Use a model with: [bold]victor --profile <profile>[/dim]"
                )

                await ollama.close()

            except Exception as e:
                console.print(f"[red]Error listing models:[/] {e}")
                console.print("\nMake sure Ollama is running: [bold]ollama serve[/]")

        else:
            console.print(f"[yellow]Model listing not yet implemented for {provider}[/]")
            console.print(
                "Currently only Ollama supports model listing via CLI"
            )

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")


@app.command()
def test_provider(
    provider: str = typer.Argument(..., help="Provider name to test"),
) -> None:
    """Test if a provider is working correctly."""
    console.print(f"Testing provider: [cyan]{provider}[/]")

    asyncio.run(test_provider_async(provider))


async def test_provider_async(provider: str) -> None:
    """Async function to test provider."""
    from victor.providers.registry import ProviderRegistry, ProviderNotFoundError

    settings = load_settings()

    try:
        # Check if provider is registered
        if not ProviderRegistry.is_registered(provider):
            console.print(f"[red]✗[/] Provider '{provider}' not found")
            console.print(f"\nAvailable providers: {', '.join(ProviderRegistry.list_providers())}")
            return

        console.print(f"[green]✓[/] Provider registered")

        # Get provider settings
        provider_settings = settings.get_provider_settings(provider)

        # Check API key for cloud providers
        if provider in ["anthropic", "openai", "google", "xai", "grok"]:
            api_key = provider_settings.get("api_key")
            if not api_key:
                console.print(f"[red]✗[/] No API key configured for {provider}")
                console.print(f"\nSet environment variable: [bold]{provider.upper()}_API_KEY[/]")
                return
            console.print(f"[green]✓[/] API key configured")

        # For Ollama, test connection
        if provider == "ollama":
            from victor.providers.ollama import OllamaProvider

            ollama = OllamaProvider(**provider_settings)
            try:
                models = await ollama.list_models()
                if models:
                    console.print(f"[green]✓[/] Ollama is running with {len(models)} models")
                else:
                    console.print("[yellow]⚠[/] Ollama is running but no models installed")
                    console.print("\nPull a model: [bold]ollama pull qwen2.5-coder:7b[/]")

                await ollama.close()

            except Exception as e:
                console.print(f"[red]✗[/] Cannot connect to Ollama: {e}")
                console.print("\nMake sure Ollama is running: [bold]ollama serve[/]")
                return

        console.print(f"\n[green]✓[/] Provider {provider} is ready to use!")

    except Exception as e:
        console.print(f"[red]✗[/] Error: {e}")


if __name__ == "__main__":
    app()
