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
SUPPORTED_PROVIDERS = ["ollama", "lmstudio", "llamacpp", "vllm", "anthropic", "openai", "google"]


@models_app.command("list")
def list_models(
    provider: str = typer.Option(
        "ollama",
        "--provider",
        "-p",
        help=f"Provider to list models from ({', '.join(SUPPORTED_PROVIDERS)})",
    ),
    endpoint: str = typer.Option(
        None,
        "--endpoint",
        "-e",
        help="Endpoint URL for LMStudio/Ollama (e.g., http://192.168.1.20:1234)",
    ),
) -> None:
    """List available models for a provider.

    Supports: ollama, lmstudio, anthropic, openai, google

    Examples:
        victor models list -p ollama
        victor models list -p lmstudio
        victor models list -p lmstudio -e http://192.168.1.20:1234
        victor models list -p anthropic
        victor models list -p openai
        victor models list -p google
    """
    asyncio.run(list_models_async(provider, endpoint))


async def list_models_async(provider: str, endpoint: str = None) -> None:
    """Async function to list models.

    Args:
        provider: Provider name
        endpoint: Optional endpoint URL override
    """
    settings = load_settings()
    provider = provider.lower()

    if provider not in SUPPORTED_PROVIDERS:
        console.print(f"[red]Unsupported provider: {provider}[/]")
        console.print(f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}")
        return

    try:
        if provider == "ollama":
            await _list_ollama_models(settings)
        elif provider == "lmstudio":
            await _list_lmstudio_models(settings, endpoint)
        elif provider == "llamacpp":
            await _list_llamacpp_models(settings, endpoint)
        elif provider == "vllm":
            await _list_vllm_models(settings, endpoint)
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


async def _list_lmstudio_models(settings, endpoint: str = None) -> None:
    """List LMStudio models with capability detection.

    Args:
        settings: Application settings
        endpoint: Optional LMStudio endpoint URL
    """
    from victor.providers.lmstudio_provider import (
        LMStudioProvider,
        _model_supports_tools,
        _model_uses_thinking_tags,
    )

    provider_settings = settings.get_provider_settings("lmstudio")

    # Allow endpoint override
    if endpoint:
        provider_settings["base_url"] = endpoint

    base_url = provider_settings.get("base_url", "http://127.0.0.1:1234")

    console.print(f"[dim]Connecting to LMStudio at {base_url}...[/]")

    try:
        lmstudio = await LMStudioProvider.create(**provider_settings)
    except Exception as e:
        console.print(f"[red]Cannot connect to LMStudio:[/] {e}")
        console.print("\n[yellow]Troubleshooting:[/]")
        console.print("  1. Make sure LMStudio is running")
        console.print("  2. Enable the local server in LMStudio settings")
        console.print("  3. Check the endpoint URL is correct")
        console.print(f"\n[dim]Tried: {base_url}[/]")
        return

    try:
        models_list = await lmstudio.list_models()

        if not models_list:
            console.print("[yellow]No models found on LMStudio server[/]")
            console.print("\n[dim]Load a model in LMStudio to see it here[/]")
            return

        table = Table(title=f"Available Models (LMStudio @ {base_url})", show_header=True)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Tools", style="green", justify="center")
        table.add_column("Thinking", style="yellow", justify="center")
        table.add_column("Recommended", style="magenta")

        # Categorize models
        tool_models = []
        thinking_models = []
        other_models = []

        for model in models_list:
            model_id = model.get("id", "unknown")
            has_tools = _model_supports_tools(model_id)
            has_thinking = _model_uses_thinking_tags(model_id)

            tools_icon = "✅" if has_tools else "❌"
            thinking_icon = "🧠" if has_thinking else "-"

            # Recommend models based on capabilities
            recommended = ""
            if has_tools and "coder" in model_id.lower():
                recommended = "⭐ Best for coding"
                tool_models.append(model_id)
            elif has_tools:
                recommended = "Good for tasks"
                tool_models.append(model_id)
            elif has_thinking:
                recommended = "Good for reasoning"
                thinking_models.append(model_id)
            else:
                other_models.append(model_id)

            table.add_row(model_id, tools_icon, thinking_icon, recommended)

        console.print(table)

        # Summary
        console.print(f"\n[dim]Total models: {len(models_list)}[/]")
        if tool_models:
            console.print(f"[green]Tool-capable models: {len(tool_models)}[/]")
        if thinking_models:
            console.print(f"[yellow]Thinking models: {len(thinking_models)}[/]")

        console.print("\n[dim]Use a model with: [bold]victor chat --provider lmstudio --model <model>[/dim]")

        # Show best recommendations
        if tool_models:
            best = tool_models[0]
            console.print(f"\n[green]Recommended:[/] victor chat --provider lmstudio --model {best}")

    except Exception as e:
        console.print(f"[red]Error listing models:[/] {e}")
    finally:
        await lmstudio.close()


async def _list_llamacpp_models(settings, endpoint: str = None) -> None:
    """List llama.cpp server models with server info.

    Args:
        settings: Application settings
        endpoint: Optional llama.cpp endpoint URL
    """
    from victor.providers.llamacpp_provider import LlamaCppProvider

    provider_settings = settings.get_provider_settings("llamacpp")

    # Allow endpoint override
    if endpoint:
        provider_settings["base_url"] = endpoint

    base_url = provider_settings.get("base_url", "http://localhost:8080")

    console.print(f"[dim]Connecting to llama.cpp at {base_url}...[/]")

    try:
        llamacpp = await LlamaCppProvider.create(**provider_settings)
    except Exception as e:
        console.print(f"[red]Cannot connect to llama.cpp server:[/] {e}")
        console.print("\n[yellow]Troubleshooting:[/]")
        console.print("  1. Make sure llama.cpp server is running")
        console.print("  2. Start with:")
        console.print("     [dim]llama-server -m model.gguf --port 8080[/]")
        console.print("\n  Or using llama-cpp-python:")
        console.print("     [dim]pip install llama-cpp-python[server][/]")
        console.print("     [dim]python -m llama_cpp.server --model model.gguf[/]")
        console.print(f"\n[dim]Tried: {base_url}[/]")
        return

    try:
        models_list = await llamacpp.list_models()
        health = await llamacpp.check_health()
        props = await llamacpp.get_server_props()

        # Server status
        status = health.get("status", "unknown")
        if status == "ok":
            console.print("[green]Server status: OK[/]")
        else:
            console.print(f"[yellow]Server status: {status}[/]")

        if not models_list:
            console.print("[yellow]No models reported by server[/]")
            # llama.cpp may not list models, but still be running
            if props:
                console.print("\n[cyan]Server Properties:[/]")
                if "default_generation_settings" in props:
                    gen = props["default_generation_settings"]
                    if "model" in gen:
                        console.print(f"  Model: {gen.get('model', 'unknown')}")
                    if "n_ctx" in gen:
                        console.print(f"  Context: {gen.get('n_ctx', 'unknown')}")
        else:
            table = Table(title=f"Available Models (llama.cpp @ {base_url})", show_header=True)
            table.add_column("Model", style="cyan", no_wrap=True)
            table.add_column("Type", style="yellow")

            for model in models_list:
                model_id = model.get("id", "unknown")
                model_type = "GGUF" if ".gguf" in model_id.lower() else "Model"
                table.add_row(model_id, model_type)

            console.print(table)

        console.print("\n[dim]Use with: [bold]victor chat --provider llamacpp[/dim]")

    except Exception as e:
        console.print(f"[red]Error listing models:[/] {e}")
    finally:
        await llamacpp.close()


async def _list_vllm_models(settings, endpoint: str = None) -> None:
    """List vLLM server models with capability detection.

    Args:
        settings: Application settings
        endpoint: Optional vLLM endpoint URL
    """
    from victor.providers.vllm_provider import (
        VLLMProvider,
        _model_supports_tools,
        _model_uses_thinking_tags,
    )

    provider_settings = settings.get_provider_settings("vllm")

    # Allow endpoint override
    if endpoint:
        provider_settings["base_url"] = endpoint

    base_url = provider_settings.get("base_url", "http://localhost:8000")

    console.print(f"[dim]Connecting to vLLM at {base_url}...[/]")

    try:
        vllm = await VLLMProvider.create(**provider_settings)
    except Exception as e:
        console.print(f"[red]Cannot connect to vLLM server:[/] {e}")
        console.print("\n[yellow]Troubleshooting:[/]")
        console.print("  1. Make sure vLLM is running with a model loaded")
        console.print("  2. Start vLLM with:")
        console.print("     [dim]python -m vllm.entrypoints.openai.api_server \\[/]")
        console.print("     [dim]  --model Qwen/Qwen2.5-Coder-7B-Instruct \\[/]")
        console.print("     [dim]  --port 8000[/]")
        console.print(f"\n[dim]Tried: {base_url}[/]")
        return

    try:
        models_list = await vllm.list_models()

        if not models_list:
            console.print("[yellow]No models found on vLLM server[/]")
            console.print("\n[dim]Load a model when starting vLLM server[/]")
            return

        table = Table(title=f"Available Models (vLLM @ {base_url})", show_header=True)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Tools", style="green", justify="center")
        table.add_column("Thinking", style="yellow", justify="center")
        table.add_column("Notes", style="dim")

        for model in models_list:
            model_id = model.get("id", "unknown")
            has_tools = _model_supports_tools(model_id)
            has_thinking = _model_uses_thinking_tags(model_id)

            tools_icon = "✅" if has_tools else "❌"
            thinking_icon = "🧠" if has_thinking else "-"

            notes = []
            if "qwen" in model_id.lower() and "coder" in model_id.lower():
                notes.append("⭐ Best for coding")
            elif has_tools:
                notes.append("Good for tasks")

            table.add_row(model_id, tools_icon, thinking_icon, " ".join(notes))

        console.print(table)
        console.print(f"\n[dim]Total models loaded: {len(models_list)}[/]")

        # Health check
        if await vllm.check_health():
            console.print("[green]Server health: OK[/]")
        else:
            console.print("[yellow]Server health: Unknown[/]")

        console.print("\n[dim]Use a model with: [bold]victor chat --provider vllm --model <model>[/dim]")

        # Show recommendation if available
        for model in models_list:
            model_id = model.get("id", "")
            if _model_supports_tools(model_id):
                console.print(f"\n[green]Recommended:[/] victor chat --provider vllm --model {model_id}")
                break

    except Exception as e:
        console.print(f"[red]Error listing models:[/] {e}")
    finally:
        await vllm.close()


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
