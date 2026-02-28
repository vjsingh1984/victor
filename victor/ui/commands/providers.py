import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from victor.providers.registry import ProviderRegistry
from victor.providers.health import ProviderHealthChecker

providers_app = typer.Typer(name="providers", help="List all available providers.")
console = Console()

# Aliases map to their primary provider (for consolidation)
PROVIDER_ALIASES = {
    "grok": "xai",
    "kimi": "moonshot",
    "llama-cpp": "llamacpp",
    "llama.cpp": "llamacpp",
    "vertexai": "vertex",
    "azure-openai": "azure",
    "aws": "bedrock",
    "hf": "huggingface",
}


@providers_app.command("list")
def list_providers() -> None:
    """List all available providers."""
    available_providers = ProviderRegistry.list_providers()

    # Provider info: (status, features, aliases)
    provider_info = {
        # Tested and working providers
        "ollama": ("✅ Ready", "Local models, Free, Tool calling", None),
        "anthropic": ("✅ Ready", "Claude 3.5/4, Tool calling, Streaming", None),
        "openai": ("✅ Ready", "GPT-4.1/4o/o1/o3, Function calling, Vision", None),
        "google": ("✅ Ready", "Gemini 2.5, 1M context, Multimodal", None),
        "xai": ("✅ Ready", "Grok, Real-time info, Vision", "grok"),
        "lmstudio": ("✅ Ready", "Local models via LMStudio", None),
        "vllm": ("✅ Ready", "High-throughput local inference", None),
        "moonshot": ("✅ Ready", "Kimi K2, 256K context, Reasoning", "kimi"),
        "deepseek": ("✅ Ready", "DeepSeek-V3, 128K, Cheap", None),
        "groqcloud": ("✅ Ready", "Ultra-fast LPU, Free tier, Llama/GPT-OSS", None),
        "cerebras": ("✅ Ready", "Ultra-fast inference, Free tier, Qwen/Llama", None),
        # Local inference
        "llamacpp": ("✅ Ready", "Local GGUF models, CPU/GPU", "llama-cpp, llama.cpp"),
        # Tested providers
        "mistral": ("✅ Ready", "Mistral Large/Codestral, 500K tokens/min free", None),
        # Known but untested providers
        "together": ("⚠️ Untested", "Together AI, $25 free credits", None),
        "openrouter": ("✅ Ready", "Unified gateway, 350+ models, Free tier", None),
        "fireworks": ("✅ Ready", "Fast inference, $1 free credits, Tool calling", None),
        "huggingface": ("⚠️ Untested", "HuggingFace Inference API", "hf"),
        "replicate": ("⚠️ Untested", "Replicate, Open models", None),
        # Enterprise providers (require setup)
        "vertex": ("🏢 Enterprise", "Google Cloud Vertex AI", "vertexai"),
        "azure": ("🏢 Enterprise", "Azure OpenAI Service", "azure-openai"),
        "bedrock": ("🏢 Enterprise", "AWS Bedrock (Claude, Llama, Titan)", "aws"),
    }

    table = Table(title="Available Providers", show_header=True)
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Features")
    table.add_column("Aliases", style="dim")

    # Filter out aliases, show only primary providers
    primary_providers = [p for p in available_providers if p not in PROVIDER_ALIASES]

    for provider in sorted(primary_providers):
        info = provider_info.get(provider)
        if info:
            status, features, aliases = info
            table.add_row(provider, status, features, aliases or "")
        else:
            table.add_row(provider, "❓ Unknown", "", "")

    console.print(table)
    console.print(
        f"\n[dim]Total: {len(primary_providers)} providers ({len(available_providers) - len(primary_providers)} aliases hidden)[/]"
    )
    console.print("[dim]Use 'victor profiles' to see configured profiles[/]")


@providers_app.command("check")
def check_provider(
    provider: str = typer.Argument(..., help="Provider name (e.g., deepseek, anthropic, ollama)"),
    model: str = typer.Option("deepseek-chat", help="Model to check"),
    connectivity: bool = typer.Option(False, "--connectivity", "-c", help="Perform connectivity test (slower)"),
    timeout: float = typer.Option(5.0, help="Timeout for connectivity check (seconds)"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Check if a provider is properly configured.

    Example:
        victor providers check deepseek
        victor providers check anthropic --connectivity
        victor providers check ollama
    """
    async def run_check():
        checker = ProviderHealthChecker()
        result = await checker.check_provider(
            provider=provider,
            model=model,
            check_connectivity=connectivity,
            timeout=timeout,
        )

        if json_output:
            import json
            console.print(json.dumps(result.to_dict(), indent=2))
            return

        # Display results in a nice format
        if result.healthy:
            status_panel = Panel(
                f"[bold green]✓ HEALTHY[/bold green]",
                title=f"[bold]{provider.upper()}[/bold]",
                subtitle=f"Model: {model}",
            )
            console.print(status_panel)

            # Show info
            if result.info:
                info_table = Table(title="Configuration", show_header=False)
                info_table.add_column("Key", style="cyan")
                info_table.add_column("Value", style="green")

                for key, value in result.info.items():
                    info_table.add_row(key, str(value))

                console.print(info_table)

            # Show warnings if any
            if result.warnings:
                console.print("\n[bold yellow]Warnings:[/bold yellow]")
                for warning in result.warnings:
                    console.print(f"  ⚠️  {warning}")
        else:
            status_panel = Panel(
                f"[bold red]✗ UNHEALTHY[/bold red]",
                title=f"[bold]{provider.upper()}[/bold]",
                subtitle=f"Model: {model}",
            )
            console.print(status_panel)

            # Show issues
            console.print("\n[bold red]Issues:[/bold red]")
            for i, issue in enumerate(result.issues, 1):
                console.print(f"  {i}. {issue}")

            # Show warnings if any
            if result.warnings:
                console.print("\n[bold yellow]Warnings:[/bold yellow]")
                for warning in result.warnings:
                    console.print(f"  ⚠️  {warning}")

    asyncio.run(run_check())


@providers_app.command("verify")
def verify_provider(
    provider: str = typer.Argument(..., help="Provider name"),
    model: str = typer.Option(..., help="Model to verify"),
    api_key: Optional[str] = typer.Option(None, help="API key (overrides other sources)"),
):
    """Verify provider configuration with detailed diagnostics.

    This is an enhanced health check that provides detailed information
    about where the API key is coming from and whether it's valid.

    Example:
        victor providers verify deepseek --model deepseek-chat
        victor providers verify anthropic --model claude-3-5-haiku --api-key sk-...
    """
    from victor.providers.resolution import UnifiedApiKeyResolver, APIKeyNotFoundError

    async def run_verify():
        console.print(f"[bold]Verifying Provider:[/bold] {provider}")
        console.print(f"[bold]Model:[/bold] {model}\n")

        # Check if provider is registered
        try:
            ProviderRegistry.get(provider)
            console.print("[green]✓[/green] Provider is registered")
        except Exception:
            console.print(f"[red]✗[/red] Provider '{provider}' is not registered")
            available = ProviderRegistry.list_providers()
            console.print(f"\nAvailable providers: {', '.join(available)}")
            raise typer.Exit(1)

        # Check API key resolution
        resolver = UnifiedApiKeyResolver()
        key_result = resolver.get_api_key(provider, explicit_key=api_key)

        console.print(f"\n[bold]API Key Resolution:[/bold]")
        for i, source in enumerate(key_result.sources_attempted, 1):
            status = "[green]✓[/green]" if source.found else "[red]✗[/red]"
            console.print(f"  {i}. {status} {source.description}")

            if source.found and source.value_preview:
                console.print(f"     Preview: {source.value_preview}")

        if key_result.key is None:
            error = APIKeyNotFoundError(
                provider=provider,
                sources_attempted=key_result.sources_attempted,
                non_interactive=key_result.non_interactive,
                model=model,
            )
            console.print(f"\n[red]{error}[/red]")
            raise typer.Exit(1)

        # Run health check
        console.print(f"\n[bold]Health Check:[/bold]")
        checker = ProviderHealthChecker()
        health_result = await checker.check_provider(
            provider=provider,
            model=model,
            check_connectivity=False,
        )

        if health_result.healthy:
            console.print("[green]✓ All checks passed![/green]")
        else:
            console.print("[red]✗ Health check failed:[/red]")
            for issue in health_result.issues:
                console.print(f"  • {issue}")

    asyncio.run(run_verify())
