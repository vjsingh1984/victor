import typer
import asyncio
from datetime import datetime
from rich.console import Console
from rich.table import Table
import logging
from typing import Optional

from victor.config.settings import load_settings
from victor.providers.config_factory_registry import ProviderConfigRegistry

models_app = typer.Typer(name="models", help="List available models for a provider.")
console = Console()
logger = logging.getLogger(__name__)


def get_supported_providers() -> list[str]:
    """Get list of supported providers from registry."""
    return ProviderConfigRegistry.list_providers()


@models_app.command("list")
def list_models(
    provider: str = typer.Option(
        "ollama",
        "--provider",
        "-p",
        help="Provider to list models from",
    ),
    endpoint: str = typer.Option(
        None,
        "--endpoint",
        "-e",
        help="Endpoint URL for LMStudio/Ollama (e.g., http://192.168.1.20:1234)",
    ),
) -> None:
    """List available models for a provider.

    Supports all registered providers (ollama, lmstudio, anthropic, openai, google, etc.)

    Examples:
        victor models list -p ollama
        victor models list -p lmstudio
        victor models list -p lmstudio -e http://192.168.1.20:1234
        victor models list -p anthropic
        victor models list -p openai
        victor models list -p google
    """
    asyncio.run(list_models_async(provider, endpoint))


async def list_models_async(provider: str, endpoint: Optional[str] = None) -> None:
    """Async function to list models using the provider registry.

    This implementation follows the Open/Closed Principle (OCP) by using
    a registry pattern instead of hardcoded if-elif chains for each provider.

    Args:
        provider: Provider name
        endpoint: Optional endpoint URL override
    """
    settings = load_settings()
    provider = provider.lower()

    # Check if provider is registered
    if not ProviderConfigRegistry.is_registered(provider):
        console.print(f"[red]Unsupported provider: {provider}[/]")
        console.print(f"Supported providers: {', '.join(get_supported_providers())}")
        return

    try:
        # Use registry to list models (OCP compliant - no hardcoded if-elif)
        models_list = await ProviderConfigRegistry.list_models(provider, settings, endpoint)

        if not models_list:
            console.print(f"[yellow]No models found for {provider}[/]")
            if provider == "ollama":
                console.print("\nPull a model with: [bold]ollama pull qwen2.5-coder:7b[/]")
            return

        # Display models in a table
        table = Table(title=f"Available Models ({provider})", show_header=True)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Size/Info", style="yellow")

        for model in models_list:
            model_id = model.get("id", model.get("name", "unknown"))
            name = model.get("name", model_id)
            size_info = model.get("size", model.get("modified_at", ""))

            # Format size if present
            if isinstance(size_info, (int, float)):
                size_gb = size_info / (1024**3)
                size_info = f"{size_gb:.1f} GB"
            elif isinstance(size_info, str) and size_info:
                try:
                    dt = datetime.fromisoformat(size_info.replace("Z", "+00:00"))
                    size_info = dt.strftime("%Y-%m-%d")
                except (ValueError, AttributeError):
                    pass

            table.add_row(model_id, name, str(size_info) if size_info else "-")

        console.print(table)
        console.print("\n[dim]Use a model with: [bold]victor --profile <profile>[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        logger.exception(f"Failed to list models for {provider}")
