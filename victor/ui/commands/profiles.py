import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

import yaml

from victor.config.settings import load_settings, Settings, ProfileConfig

profiles_app = typer.Typer(name="profiles", help="Manage Victor profiles.")
console = Console()


@profiles_app.command("list")
def list_profiles() -> None:
    """List configured profiles."""
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
    table.add_column("Description", style="dim")

    for name, profile in profiles.items():
        table.add_row(
            name,
            profile.provider,
            profile.model,
            f"{profile.temperature}",
            f"{profile.max_tokens}",
            profile.description or "-",
        )

    console.print(table)
    console.print(f"\n[dim]Config file: {settings.get_config_dir() / 'profiles.yaml'}[/]")


@profiles_app.command("create")
def create_profile(
    name: str = typer.Argument(..., help="Profile name"),
    provider: str = typer.Option(..., "--provider", "-p", help="Provider name (ollama, anthropic, openai, google)"),
    model: str = typer.Option(..., "--model", "-m", help="Model identifier"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Temperature (0.0-2.0)"),
    max_tokens: int = typer.Option(4096, "--max-tokens", help="Maximum output tokens"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Profile description"),
) -> None:
    """Create a new profile.

    Examples:
        victor profiles create myprofile -p ollama -m qwen2.5-coder:7b
        victor profiles create cloud -p anthropic -m claude-sonnet-4-20250514 --max-tokens 8192
    """
    profiles_file = Settings.get_config_dir() / "profiles.yaml"

    # Load existing data
    data = _load_profiles_yaml(profiles_file)

    # Check if profile already exists
    if name in data.get("profiles", {}):
        console.print(f"[red]Error:[/] Profile '{name}' already exists")
        console.print(f"Use [bold]victor profiles edit {name}[/] to modify it")
        return

    # Create new profile
    new_profile = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if description:
        new_profile["description"] = description

    # Add to profiles
    if "profiles" not in data:
        data["profiles"] = {}
    data["profiles"][name] = new_profile

    # Save
    _save_profiles_yaml(profiles_file, data)
    console.print(f"[green]Created profile:[/] {name}")
    console.print(f"  Provider: {provider}")
    console.print(f"  Model: {model}")
    console.print(f"  Temperature: {temperature}")
    console.print(f"  Max Tokens: {max_tokens}")
    if description:
        console.print(f"  Description: {description}")
    console.print(f"\n[dim]Use with: [bold]victor --profile {name}[/dim]")


@profiles_app.command("edit")
def edit_profile(
    name: str = typer.Argument(..., help="Profile name to edit"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Provider name"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model identifier"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Temperature (0.0-2.0)"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Maximum output tokens"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Profile description"),
) -> None:
    """Edit an existing profile.

    Examples:
        victor profiles edit myprofile --temperature 0.5
        victor profiles edit cloud --model claude-opus-4-5-20251101
    """
    profiles_file = Settings.get_config_dir() / "profiles.yaml"

    # Load existing data
    data = _load_profiles_yaml(profiles_file)

    # Check if profile exists
    if name not in data.get("profiles", {}):
        console.print(f"[red]Error:[/] Profile '{name}' not found")
        console.print(f"Use [bold]victor profiles create {name}[/] to create it")
        return

    # Update profile
    profile = data["profiles"][name]
    changes = []

    if provider is not None:
        profile["provider"] = provider
        changes.append(f"provider={provider}")
    if model is not None:
        profile["model"] = model
        changes.append(f"model={model}")
    if temperature is not None:
        profile["temperature"] = temperature
        changes.append(f"temperature={temperature}")
    if max_tokens is not None:
        profile["max_tokens"] = max_tokens
        changes.append(f"max_tokens={max_tokens}")
    if description is not None:
        profile["description"] = description
        changes.append(f"description={description}")

    if not changes:
        console.print("[yellow]No changes specified[/]")
        console.print("Use --provider, --model, --temperature, --max-tokens, or --description")
        return

    # Save
    _save_profiles_yaml(profiles_file, data)
    console.print(f"[green]Updated profile:[/] {name}")
    for change in changes:
        console.print(f"  {change}")


@profiles_app.command("delete")
def delete_profile(
    name: str = typer.Argument(..., help="Profile name to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a profile.

    Examples:
        victor profiles delete myprofile
        victor profiles delete myprofile --force
    """
    profiles_file = Settings.get_config_dir() / "profiles.yaml"

    # Load existing data
    data = _load_profiles_yaml(profiles_file)

    # Check if profile exists
    if name not in data.get("profiles", {}):
        console.print(f"[red]Error:[/] Profile '{name}' not found")
        return

    # Confirm deletion
    if not force:
        profile = data["profiles"][name]
        console.print(f"Profile: [cyan]{name}[/]")
        console.print(f"  Provider: {profile.get('provider', '-')}")
        console.print(f"  Model: {profile.get('model', '-')}")

        confirm = typer.confirm("Delete this profile?")
        if not confirm:
            console.print("[yellow]Cancelled[/]")
            return

    # Delete
    del data["profiles"][name]
    _save_profiles_yaml(profiles_file, data)
    console.print(f"[green]Deleted profile:[/] {name}")


@profiles_app.command("show")
def show_profile(
    name: str = typer.Argument(..., help="Profile name to show"),
) -> None:
    """Show details of a specific profile.

    Examples:
        victor profiles show default
    """
    settings = load_settings()
    profiles = settings.load_profiles()

    if name not in profiles:
        console.print(f"[red]Error:[/] Profile '{name}' not found")
        console.print(f"\nAvailable profiles: {', '.join(profiles.keys())}")
        return

    profile = profiles[name]
    console.print(f"[bold]Profile: [cyan]{name}[/bold]")
    console.print(f"  Provider: [green]{profile.provider}[/]")
    console.print(f"  Model: [yellow]{profile.model}[/]")
    console.print(f"  Temperature: {profile.temperature}")
    console.print(f"  Max Tokens: {profile.max_tokens}")
    if profile.description:
        console.print(f"  Description: [dim]{profile.description}[/]")
    if profile.tool_selection:
        console.print(f"  Tool Selection: {profile.tool_selection}")


@profiles_app.command("set-default")
def set_default_profile(
    name: str = typer.Argument(..., help="Profile name to set as default"),
) -> None:
    """Set a profile as the default (rename to 'default').

    This renames the specified profile to 'default', making it the default
    profile used when no --profile is specified.

    Examples:
        victor profiles set-default anthropic
    """
    profiles_file = Settings.get_config_dir() / "profiles.yaml"

    # Load existing data
    data = _load_profiles_yaml(profiles_file)

    # Check if profile exists
    if name not in data.get("profiles", {}):
        console.print(f"[red]Error:[/] Profile '{name}' not found")
        return

    if name == "default":
        console.print(f"[yellow]'{name}' is already the default profile[/]")
        return

    # Swap profiles
    profiles = data.get("profiles", {})

    # Save old default if it exists
    old_default = profiles.get("default")
    new_default = profiles[name]

    # Update profiles
    if old_default:
        profiles[f"old-default"] = old_default
        console.print(f"[dim]Previous default renamed to 'old-default'[/]")

    profiles["default"] = new_default
    del profiles[name]

    # Save
    _save_profiles_yaml(profiles_file, data)
    console.print(f"[green]Set '{name}' as default profile[/]")
    console.print(f"\n[dim]Previous profile renamed to 'old-default'[/]")


def _load_profiles_yaml(profiles_file: Path) -> dict:
    """Load profiles.yaml or return empty dict."""
    if not profiles_file.exists():
        return {"profiles": {}}

    try:
        with open(profiles_file, "r") as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception as e:
        console.print(f"[red]Error loading profiles:[/] {e}")
        return {"profiles": {}}


def _save_profiles_yaml(profiles_file: Path, data: dict) -> None:
    """Save data to profiles.yaml."""
    # Ensure directory exists
    profiles_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(profiles_file, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        console.print(f"[red]Error saving profiles:[/] {e}")
        raise typer.Exit(1)
