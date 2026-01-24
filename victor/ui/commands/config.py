import typer
from rich.console import Console
from rich.table import Table
import yaml
import asyncio
from typing import Any

from victor.config.settings import load_settings

config_app = typer.Typer(name="config", help="Validate configuration files and profiles.")
console = Console()


@config_app.command("validate")
def config_validate(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed validation results",
    ),
    check_connectivity: bool = typer.Option(
        False,
        "--check-connectivity",
        "-c",
        help="Test provider connectivity (requires network/services)",
    ),
) -> None:
    """Validate configuration files and profiles."""
    errors: list[str] = []
    warnings: list[str] = []
    checks_passed = 0
    checks_total = 0

    settings = load_settings()
    config_dir = settings.get_config_dir()
    profiles_file = config_dir / "profiles.yaml"

    # Check 1: Config directory exists
    checks_total += 1
    if config_dir.exists():
        checks_passed += 1
        if verbose:
            console.print(f"[green]✓[/] Config directory exists: {config_dir}")
    else:
        errors.append(f"Config directory not found: {config_dir}")
        console.print(f"[red]✗[/] Config directory not found: {config_dir}")
        console.print("\nRun [bold]victor init[/] to create configuration")
        raise typer.Exit(1)

    # Check 2: Profiles file exists
    checks_total += 1
    if profiles_file.exists():
        checks_passed += 1
        if verbose:
            console.print(f"[green]✓[/] Profiles file exists: {profiles_file}")
    else:
        errors.append(f"Profiles file not found: {profiles_file}")
        console.print(f"[red]✗[/] Profiles file not found: {profiles_file}")
        console.print("\nRun [bold]victor init[/] to create configuration")
        raise typer.Exit(1)

    # Check 3: Valid YAML syntax
    checks_total += 1
    try:
        with open(profiles_file, "r") as f:
            raw_config = yaml.safe_load(f)
        checks_passed += 1
        if verbose:
            console.print("[green]✓[/] YAML syntax is valid")
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML syntax: {e}")
        console.print(f"[red]✗[/] Invalid YAML syntax: {e}")
        raise typer.Exit(1)

    # Check 4: Profiles section exists
    checks_total += 1
    if raw_config and "profiles" in raw_config:
        checks_passed += 1
        if verbose:
            console.print("[green]✓[/] 'profiles' section found")
    else:
        errors.append("Missing 'profiles' section in configuration")
        console.print("[red]✗[/] Missing 'profiles' section in configuration")
        raise typer.Exit(1)

    # Load and validate profiles
    try:
        profiles = settings.load_profiles()
    except Exception as e:
        errors.append(f"Failed to load profiles: {e}")
        console.print(f"[red]✗[/] Failed to load profiles: {e}")
        raise typer.Exit(1)

    # Check 5: At least one profile exists
    checks_total += 1
    if profiles:
        checks_passed += 1
        if verbose:
            console.print(f"[green]✓[/] Found {len(profiles)} profile(s)")
    else:
        warnings.append("No profiles defined")
        console.print("[yellow]⚠[/] No profiles defined")

    # Validate each profile
    from victor.providers.registry import ProviderRegistry

    available_providers = ProviderRegistry.list_providers()

    if verbose:
        console.print("\n[bold]Profile Validation:[/]")

    for name, profile in profiles.items():
        # Check provider is valid
        checks_total += 1
        if profile.provider in available_providers:
            checks_passed += 1
            if verbose:
                console.print(f"  [green]✓[/] [{name}] Provider '{profile.provider}' is valid")
        else:
            errors.append(f"[{name}] Unknown provider: {profile.provider}")
            console.print(f"  [red]✗[/] [{name}] Unknown provider: {profile.provider}")

        # Check model is specified
        checks_total += 1
        if profile.model:
            checks_passed += 1
            if verbose:
                console.print(f"  [green]✓[/] [{name}] Model specified: {profile.model}")
        else:
            errors.append(f"[{name}] No model specified")
            console.print(f"  [red]✗[/] [{name}] No model specified")

        # Check temperature range
        checks_total += 1
        if 0.0 <= profile.temperature <= 2.0:
            checks_passed += 1
            if verbose:
                console.print(f"  [green]✓[/] [{name}] Temperature {profile.temperature} is valid")
        else:
            errors.append(f"[{name}] Temperature {profile.temperature} out of range [0.0, 2.0]")
            console.print(f"  [red]✗[/] [{name}] Temperature out of range")

        # Check max_tokens
        checks_total += 1
        if profile.max_tokens > 0:
            checks_passed += 1
            if verbose:
                console.print(f"  [green]✓[/] [{name}] Max tokens {profile.max_tokens} is valid")
        else:
            errors.append(f"[{name}] Invalid max_tokens: {profile.max_tokens}")
            console.print(f"  [red]✗[/] [{name}] Invalid max_tokens")

        # Check API keys for cloud providers
        if profile.provider in ["anthropic", "openai", "google", "xai", "grok"]:
            checks_total += 1
            provider_settings = settings.get_provider_settings(profile.provider)
            api_key = provider_settings.get("api_key")
            if api_key:
                checks_passed += 1
                if verbose:
                    console.print(
                        f"  [green]✓[/] [{name}] API key configured for {profile.provider}"
                    )
            else:
                warnings.append(f"[{name}] No API key for {profile.provider}")
                console.print(
                    f"  [yellow]⚠[/] [{name}] No API key configured for {profile.provider}"
                )

    # Optional connectivity checks
    if check_connectivity:
        console.print("\n[bold]Connectivity Checks:[/]")
        asyncio.run(_check_connectivity(settings, profiles, verbose))

    # Summary
    console.print("\n" + "─" * 50)
    if errors:
        console.print(f"[red]Validation failed with {len(errors)} error(s)[/]")
        for err in errors:
            console.print(f"  [red]•[/] {err}")
        raise typer.Exit(1)
    elif warnings:
        console.print(f"[yellow]Validation passed with {len(warnings)} warning(s)[/]")
        for warn in warnings:
            console.print(f"  [yellow]•[/] {warn}")
        console.print(f"\n[green]✓[/] {checks_passed}/{checks_total} checks passed")
    else:
        console.print(
            f"[green]✓ Validation passed![/] {checks_passed}/{checks_total} checks passed"
        )


async def _check_connectivity(settings: Any, profiles: dict[str, Any], verbose: bool) -> None:
    """Check provider connectivity for profiles."""
    checked_providers: set[str] = set()

    for _name, profile in profiles.items():
        provider = profile.provider
        if provider in checked_providers:
            continue
        checked_providers.add(provider)

        if provider == "ollama":
            from victor.providers.ollama_provider import OllamaProvider

            provider_settings = settings.get_provider_settings(provider)
            try:
                ollama = OllamaProvider(**provider_settings)
                models = await ollama.list_models()
                if models:
                    console.print(f"  [green]✓[/] Ollama: Connected ({len(models)} models)")
                else:
                    console.print("  [yellow]⚠[/] Ollama: Connected but no models installed")
                await ollama.close()
            except Exception as e:
                console.print(f"  [red]✗[/] Ollama: Cannot connect - {e}")
        else:
            if verbose:
                console.print(f"  [dim]→[/] {provider}: Connectivity check not implemented")
