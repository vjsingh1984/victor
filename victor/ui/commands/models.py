import typer
import asyncio
from rich.console import Console
from rich.table import Table
import logging

from victor.config.settings import load_settings

models_app = typer.Typer(name="models", help="List available models for a provider.")
console = Console()
logger = logging.getLogger(__name__)

@models_app.command("list")
def list_models(
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
    settings = load_settings()

    try:
        # Special handling for Ollama
        if provider == "ollama":
            from victor.providers.ollama_provider import OllamaProvider

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
                        except (ValueError, AttributeError) as e:
                            logger.debug(f"Failed to parse timestamp '{modified}': {e}")
                            pass

                    table.add_row(
                        name,
                        f"{size_gb:.1f} GB" if size_gb > 0 else "unknown",
                        modified,
                    )

                console.print(table)
                console.print("\n[dim]Use a model with: [bold]victor --profile <profile>[/dim]")

                await ollama.close()

            except Exception as e:
                console.print(f"[red]Error listing models:[/] {e}")
                console.print("\nMake sure Ollama is running: [bold]ollama serve[/]")

        else:
            console.print(f"[yellow]Model listing not yet implemented for {provider}[/]")
            console.print("Currently only Ollama supports model listing via CLI")

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
