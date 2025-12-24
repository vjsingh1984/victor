import typer
from rich.console import Console
from rich.table import Table

from victor.providers.registry import ProviderRegistry

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
    console.print(f"\n[dim]Total: {len(primary_providers)} providers ({len(available_providers) - len(primary_providers)} aliases hidden)[/]")
    console.print("[dim]Use 'victor profiles' to see configured profiles[/]")
