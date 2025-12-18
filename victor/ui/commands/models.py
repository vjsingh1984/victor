import typer
import asyncio
from datetime import datetime
from rich.console import Console
from rich.table import Table
import logging

from victor.config.settings import load_settings

models_app = typer.Typer(name="models", help="List available models for a provider.")
console = Console()
logger = logging.getLogger(__name__)

# Supported providers for model listing
SUPPORTED_PROVIDERS = ["ollama", "anthropic", "openai", "google"]


@models_app.command("list")
def list_models(
    provider: str = typer.Option(
        "ollama",
        "--provider",
        "-p",
        help=f"Provider to list models from ({', '.join(SUPPORTED_PROVIDERS)})",
    ),
) -> None:
    """List available models for a provider.

    Supports: ollama, anthropic, openai, google

    Examples:
        victor models list -p ollama
        victor models list -p anthropic
        victor models list -p openai
        victor models list -p google
    """
    asyncio.run(list_models_async(provider))


async def list_models_async(provider: str) -> None:
    """Async function to list models."""
    settings = load_settings()
    provider = provider.lower()

    if provider not in SUPPORTED_PROVIDERS:
        console.print(f"[red]Unsupported provider: {provider}[/]")
        console.print(f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}")
        return

    try:
        if provider == "ollama":
            await _list_ollama_models(settings)
        elif provider == "anthropic":
            await _list_anthropic_models(settings)
        elif provider == "openai":
            await _list_openai_models(settings)
        elif provider == "google":
            await _list_google_models(settings)

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")


async def _list_ollama_models(settings) -> None:
    """List Ollama models."""
    from victor.providers.ollama_provider import OllamaProvider

    provider_settings = settings.get_provider_settings("ollama")
    ollama = OllamaProvider(**provider_settings)

    try:
        models_list = await ollama.list_models()

        if not models_list:
            console.print("[yellow]No models found for ollama[/]")
            console.print("\nPull a model with: [bold]ollama pull qwen2.5-coder:7b[/]")
            return

        table = Table(title="Available Models (ollama)", show_header=True)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Size", style="yellow")
        table.add_column("Modified", style="dim")

        for model in models_list:
            name = model.get("name", "unknown")
            size = model.get("size", 0)
            size_gb = size / (1024**3) if size else 0

            modified = model.get("modified_at", "")
            if modified:
                try:
                    dt = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                    modified = dt.strftime("%Y-%m-%d")
                except (ValueError, AttributeError) as e:
                    logger.debug(f"Failed to parse timestamp '{modified}': {e}")

            table.add_row(
                name,
                f"{size_gb:.1f} GB" if size_gb > 0 else "unknown",
                modified,
            )

        console.print(table)
        console.print("\n[dim]Use a model with: [bold]victor --profile <profile>[/dim]")

    except Exception as e:
        console.print(f"[red]Error listing models:[/] {e}")
        console.print("\nMake sure Ollama is running: [bold]ollama serve[/]")
    finally:
        await ollama.close()


async def _list_anthropic_models(settings) -> None:
    """List Anthropic Claude models."""
    from victor.providers.anthropic_provider import AnthropicProvider

    provider_settings = settings.get_provider_settings("anthropic")
    api_key = provider_settings.get("api_key")

    if not api_key:
        console.print("[red]Anthropic API key not configured[/]")
        console.print("Set ANTHROPIC_API_KEY environment variable or configure in profiles.yaml")
        return

    anthropic = AnthropicProvider(**provider_settings)

    try:
        models_list = await anthropic.list_models()

        table = Table(title="Available Models (anthropic)", show_header=True)
        table.add_column("Model ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="yellow")
        table.add_column("Context", style="green")
        table.add_column("Description", style="dim")

        for model in models_list:
            context = model.get("context_window", "")
            context_str = f"{context:,}" if context else "-"
            table.add_row(
                model.get("id", "unknown"),
                model.get("name", ""),
                context_str,
                model.get("description", "")[:50],
            )

        console.print(table)
        console.print("\n[dim]Use a model with: [bold]victor --profile anthropic[/dim]")

    except Exception as e:
        console.print(f"[red]Error listing models:[/] {e}")
    finally:
        await anthropic.close()


async def _list_openai_models(settings) -> None:
    """List OpenAI models."""
    from victor.providers.openai_provider import OpenAIProvider

    provider_settings = settings.get_provider_settings("openai")
    api_key = provider_settings.get("api_key")

    if not api_key:
        console.print("[red]OpenAI API key not configured[/]")
        console.print("Set OPENAI_API_KEY environment variable or configure in profiles.yaml")
        return

    openai = OpenAIProvider(**provider_settings)

    try:
        models_list = await openai.list_models()

        if not models_list:
            console.print("[yellow]No chat models found for openai[/]")
            return

        table = Table(title="Available Models (openai)", show_header=True)
        table.add_column("Model ID", style="cyan", no_wrap=True)
        table.add_column("Owner", style="yellow")
        table.add_column("Created", style="dim")

        for model in models_list:
            created = model.get("created", 0)
            created_str = ""
            if created:
                try:
                    dt = datetime.fromtimestamp(created)
                    created_str = dt.strftime("%Y-%m-%d")
                except (ValueError, OSError):
                    pass

            table.add_row(
                model.get("id", "unknown"),
                model.get("owned_by", "-"),
                created_str,
            )

        console.print(table)
        console.print(f"\n[dim]Found {len(models_list)} chat-capable models[/]")
        console.print("[dim]Use a model with: [bold]victor --profile openai[/dim]")

    except Exception as e:
        console.print(f"[red]Error listing models:[/] {e}")
    finally:
        await openai.close()


async def _list_google_models(settings) -> None:
    """List Google Gemini models."""
    from victor.providers.google_provider import GoogleProvider

    provider_settings = settings.get_provider_settings("google")
    api_key = provider_settings.get("api_key")

    if not api_key:
        console.print("[red]Google API key not configured[/]")
        console.print("Set GOOGLE_API_KEY environment variable or configure in profiles.yaml")
        return

    try:
        google = GoogleProvider(**provider_settings)
    except ImportError:
        console.print("[red]Google provider not installed[/]")
        console.print("Install with: [bold]pip install victor[google][/bold]")
        return

    try:
        models_list = await google.list_models()

        if not models_list:
            console.print("[yellow]No generative models found for google[/]")
            return

        table = Table(title="Available Models (google)", show_header=True)
        table.add_column("Model ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="yellow")
        table.add_column("Input Tokens", style="green")
        table.add_column("Output Tokens", style="green")

        for model in models_list:
            input_limit = model.get("input_token_limit", 0)
            output_limit = model.get("output_token_limit", 0)
            table.add_row(
                model.get("id", "unknown"),
                model.get("name", "-"),
                f"{input_limit:,}" if input_limit else "-",
                f"{output_limit:,}" if output_limit else "-",
            )

        console.print(table)
        console.print(f"\n[dim]Found {len(models_list)} generative models[/]")
        console.print("[dim]Use a model with: [bold]victor --profile google[/dim]")

    except Exception as e:
        console.print(f"[red]Error listing models:[/] {e}")
    finally:
        await google.close()
