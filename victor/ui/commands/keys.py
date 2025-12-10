import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
import platform
import os
import getpass
from typing import Optional

from victor.config.api_keys import (
    APIKeyManager,
    get_configured_providers,
    create_api_keys_template,
    DEFAULT_KEYS_FILE,
    PROVIDER_ENV_VARS,
    is_keyring_available,
    _get_key_from_keyring,
    _set_key_in_keyring,
    _delete_key_from_keyring,
)

keys_app = typer.Typer(name="keys", help="Manage API keys for cloud providers.")
console = Console()

@keys_app.callback(invoke_without_command=True)
def keys(
    ctx: typer.Context,
    setup: bool = typer.Option(False, "--setup", "-s", help="Create API keys template file"),
    list_keys: bool = typer.Option(True, "--list", "-l", help="List configured providers"),
    provider: Optional[str] = typer.Option(None, "--set", help="Set API key for a provider"),
    keyring: bool = typer.Option(
        False, "--keyring", "-k", help="Store key in system keyring (secure)"
    ),
    migrate: bool = typer.Option(False, "--migrate", help="Migrate keys from file to keyring"),
    delete_keyring: Optional[str] = typer.Option(
        None, "--delete-keyring", help="Delete key from keyring"
    ),
):
    """Manage API keys for cloud providers."""
    if ctx.invoked_subcommand is None:
        if setup:
            _setup()
        elif delete_keyring:
            _delete_keyring(delete_keyring)
        elif migrate:
            _migrate()
        elif provider:
            _set_key(provider, keyring)
        else:
            _list_keys()

def _setup():
    # Create template file
    if DEFAULT_KEYS_FILE.exists():
        if not Confirm.ask(f"[yellow]{DEFAULT_KEYS_FILE} already exists. Overwrite?[:]"):
            console.print("[dim]Cancelled[/]")
            return

    DEFAULT_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    template = create_api_keys_template()
    DEFAULT_KEYS_FILE.write_text(template)

    # Set secure permissions
    os.chmod(DEFAULT_KEYS_FILE, 0o600)

    console.print(f"[green]✓[/] Created API keys template at [cyan]{DEFAULT_KEYS_FILE}[/]")
    console.print("\n[yellow]Next steps:[/]")
    console.print(f"  1. Edit [cyan]{DEFAULT_KEYS_FILE}[/]")
    console.print("  2. Replace placeholder values with your actual API keys")
    console.print("  3. File permissions already set to 0600 (owner-only)")
    console.print()
    console.print("[dim]Or use keyring for more secure storage:[/]")
    console.print("  victor keys --set anthropic --keyring")

def _delete_keyring(provider_name: str):
    # Delete key from keyring
    provider_name = provider_name.lower()
    if not is_keyring_available():
        console.print("[red]Keyring not available.[/] Install with: pip install keyring")
        raise typer.Exit(1)

    if _delete_key_from_keyring(provider_name):
        console.print(f"[green]✓[/] Deleted [cyan]{provider_name}[/] from keyring")
    else:
        console.print(f"[yellow]Key for {provider_name} not found in keyring")

def _migrate():
    # Migrate keys from file to keyring
    if not is_keyring_available():
        console.print("[red]Keyring not available.[/] Install with: pip install keyring")
        console.print("\n[dim]Platform requirements:[/]")
        console.print("  macOS: Built-in Keychain support")
        console.print("  Windows: Built-in Credential Manager")
        console.print("  Linux: Install gnome-keyring or kwallet")
        raise typer.Exit(1)

    backend_name, status = _get_keyring_info()
    console.print(f"[cyan]Keyring backend:[/] {backend_name} {status}")
    console.print()

    if not DEFAULT_KEYS_FILE.exists():
        console.print(f"[yellow]No keys file found at {DEFAULT_KEYS_FILE}[/]")
        raise typer.Exit(1)

    # Load keys from file
    import yaml

    with open(DEFAULT_KEYS_FILE) as f:
        data = yaml.safe_load(f) or {}

    api_keys = data.get("api_keys", data)
    migrated = 0
    failed = 0

    for prov, key in api_keys.items():
        if key and not key.startswith("your_"):
            if _set_key_in_keyring(prov, key):
                console.print(f"  [green]✓[/] Migrated [cyan]{prov}[/]")
                migrated += 1
            else:
                console.print(f"  [red]✗[/] Failed: [cyan]{prov}[/]")
                failed += 1

    console.print()
    console.print(f"[green]Migrated {migrated} keys to keyring")
    if failed:
        console.print(f"[red]Failed to migrate {failed} keys")

    if migrated > 0:
        console.print()
        console.print("[yellow]Security recommendation:[/]")
        console.print(f"  Delete the file: rm {DEFAULT_KEYS_FILE}")
        console.print("  Keys are now stored securely in system keyring")

def _set_key(provider: str, keyring: bool):
    # Set API key for a provider
    provider = provider.lower()
    if provider not in PROVIDER_ENV_VARS:
        console.print(f"[red]Unknown provider:[/] {provider}")
        console.print(f"Valid providers: {', '.join(sorted(PROVIDER_ENV_VARS.keys()))}")
        raise typer.Exit(1)

    # Check keyring availability if requested
    if keyring and not is_keyring_available():
        console.print("[red]Keyring not available.[/] Install with: pip install keyring")
        console.print("[dim]Falling back to file storage...[/]")
        keyring = False

    storage_type = "keyring" if keyring else "file"
    console.print(f"[cyan]Setting API key for {provider}[/] (storage: {storage_type})")
    console.print("[dim]Paste your API key (input hidden):[/]")

    key = getpass.getpass("")

    if not key.strip():
        console.print("[red]No key provided. Cancelled.[/]")
        raise typer.Exit(1)
    
    manager = APIKeyManager()
    if manager.set_key(provider, key.strip(), use_keyring=keyring):
        location = "system keyring" if keyring else str(DEFAULT_KEYS_FILE)
        console.print(f"[green]✓[/] API key for [cyan]{provider}[/] saved to {location}")
    else:
        console.print("[red]Failed to save API key")
        raise typer.Exit(1)

def _list_keys():
    # Default: list configured providers with source info
    configured = get_configured_providers()

    table = Table(title="API Keys Status", show_header=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Source")
    table.add_column("Env Var")

    for prov, env_var in sorted(PROVIDER_ENV_VARS.items()):
        if prov in ("kimi",):  # Skip aliases
            continue

        # Determine source
        source = "[dim]--[/]"
        if os.environ.get(env_var):
            source = "[green]env[/]"
        elif is_keyring_available() and _get_key_from_keyring(prov):
            source = "[blue]keyring[/]"
        elif prov in configured:
            source = "[yellow]file[/]"

        status = "[green]✓ Configured[/]" if prov in configured else "[dim]Not set[/]"
        table.add_row(prov, status, source, env_var)

    console.print(table)

    # Show keyring status
    console.print()
    backend_name, status = _get_keyring_info()
    keyring_status = (
        "[green]✓ Available[/]" if is_keyring_available() else "[yellow]Not installed[/]"
    )
    console.print(f"[dim]Keyring:[/] {backend_name} {keyring_status}")
    console.print(f"[dim]Keys file:[/] {DEFAULT_KEYS_FILE}")

    if not configured:
        console.print()
        console.print("[yellow]No API keys configured.[/]")
        console.print("  [cyan]victor keys --setup[/]          Create template file")
        console.print("  [cyan]victor keys --set anthropic --keyring[/]  Store in keyring (secure)")

def _get_keyring_info() -> tuple[str, str]:
    """Get keyring backend and platform info."""
    system = platform.system()
    if system == "Darwin":
        return "macOS Keychain", "[green]Supported[/]"
    elif system == "Windows":
        return "Windows Credential Manager", "[green]Supported[/]"
    elif system == "Linux":
        return "Secret Service (GNOME Keyring/KWallet)", "[green]Supported[/]"
    else:
        return "Unknown", "[yellow]May not be supported[/]"
