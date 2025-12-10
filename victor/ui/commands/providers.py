import typer
from rich.console import Console
from rich.table import Table

from victor.providers.registry import ProviderRegistry

providers_app = typer.Typer(name="providers", help="List all available providers.")
console = Console()

@providers_app.command("list")
def list_providers() -> None:
    """List all available providers."""
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
        "lmstudio": ("✅ Ready", "Local models via LMStudio"),
        "vllm": ("✅ Ready", "High-throughput local inference"),
        "moonshot": ("✅ Ready", "Kimi K2, 256K context, Reasoning"),
        "kimi": ("✅ Ready", "Alias for moonshot"),
        "deepseek": ("✅ Ready", "DeepSeek-V3, 128K, Cheap"),
        "groqcloud": ("✅ Ready", "Ultra-fast LPU, Free tier, Tool calling"),
    }

    for provider in sorted(available_providers):
        status, features = provider_info.get(provider, ("❓ Unknown", ""))
        table.add_row(provider, status, features)

    console.print(table)
    console.print("\n[dim]Use 'victor profiles' to see configured profiles[/]")
