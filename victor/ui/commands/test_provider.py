import typer
import asyncio
from rich.console import Console

from victor.config.settings import load_settings
from victor.providers.registry import ProviderRegistry

test_provider_app = typer.Typer(name="test-provider", help="Test if a provider is working correctly.")
console = Console()

@test_provider_app.callback(invoke_without_command=True)
def test_provider(
    ctx: typer.Context,
    provider: str = typer.Argument(..., help="Provider name to test"),
):
    """Test if a provider is working correctly."""
    if ctx.invoked_subcommand is None:
        console.print(f"Testing provider: [cyan]{provider}[/]")
        asyncio.run(test_provider_async(provider))


async def test_provider_async(provider: str) -> None:
    """Async function to test provider."""
    settings = load_settings()

    try:
        # Check if provider is registered
        if not ProviderRegistry.is_registered(provider):
            console.print(f"[red]✗[/] Provider '{provider}' not found")
            console.print(f"\nAvailable providers: {', '.join(ProviderRegistry.list_providers())}")
            return

        console.print("[green]✓[/] Provider registered")

        # Get provider settings
        provider_settings = settings.get_provider_settings(provider)

        # Check API key for cloud providers
        if provider in ["anthropic", "openai", "google", "xai", "grok"]:
            api_key = provider_settings.get("api_key")
            if not api_key:
                console.print(f"[red]✗[/] No API key configured for {provider}")
                console.print(f"\nSet environment variable: [bold]{provider.upper()}_API_KEY[/]")
                return
            console.print("[green]✓[/] API key configured")

        # For Ollama, test connection
        if provider == "ollama":
            from victor.providers.ollama_provider import OllamaProvider

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
