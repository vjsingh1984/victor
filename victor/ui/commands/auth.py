# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive authentication and account management for Victor.

This module provides a comprehensive auth command with subcommands for:
- setup: Interactive wizard for first-time setup
- add: Quick add a provider account
- list: Show configured accounts
- remove: Remove an account
- migrate: Migrate from old configuration format
- test: Test a provider connection

The new unified configuration format replaces the old profiles.yaml + api_keys.yaml
with a single config.yaml file.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt


class AuthStatus(str, Enum):
    """OAuth authentication status for a provider."""

    AUTHENTICATED = "authenticated"  # Valid token exists and is not expired
    EXPIRED = "expired"  # Token exists but has expired
    PENDING = "pending"  # No valid token (not authenticated or token missing)


from rich.table import Table
from rich.text import Text

from victor.config.accounts import (
    AccountManager,
    AuthConfig,
    ProviderAccount,
    get_account_manager,
)
from victor.config.connection_validation import ConnectionValidator, ValidationResult
from victor.config.migration import (
    ConfigMigrator,
    check_migration_needed,
    run_migration,
)

# =============================================================================
# Popular provider models (for quick selection)
# =============================================================================

POPULAR_MODELS: Dict[str, List[str]] = {
    "anthropic": [
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ],
    "openai": [
        "gpt-5.4-mini",
        "gpt-4.1",
        "gpt-4o",
        "o3-mini",
        "o1",
    ],
    "google": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash-exp",
    ],
    "zai": [
        "glm-4.6:coding",  # Coding Plan endpoint
        "glm-4.6:standard",  # Standard endpoint
        "glm-4.6:thinking",  # Thinking mode endpoint
        "glm-5.0",
    ],
    "xai": [
        "grok-2-1212",
        "grok-2-vision-1212",
    ],
    "qwen": [
        "qwen-coder-plus",
        "qwen-max",
        "qwen-plus",
    ],
    "moonshot": [
        "moonshot-v1-128k",
        "moonshot-v1-32k",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-coder",
        "deepseek-reasoner",
    ],
}

# =============================================================================
# OAuth-enabled providers
# =============================================================================

OAUTH_PROVIDERS = ["openai", "qwen"]

# =============================================================================
# Auth app
# =============================================================================

auth_app = typer.Typer(name="auth", help="Manage authentication and provider accounts.")
console = Console()


def _get_oauth_status(provider: str) -> AuthStatus:
    """Check OAuth token status for a provider without triggering login.

    Returns:
        AuthStatus enum: AUTHENTICATED if valid token exists, EXPIRED if token expired, PENDING otherwise.
    """
    try:
        from pathlib import Path

        from victor.config.secure_paths import get_victor_dir

        token_file = get_victor_dir() / "oauth_tokens.yaml"
    except ImportError:
        token_file = Path.home() / ".victor" / "oauth_tokens.yaml"

    if not token_file.exists():
        return AuthStatus.PENDING

    try:
        import yaml

        with open(token_file) as f:
            all_tokens = yaml.safe_load(f) or {}
        data = all_tokens.get(provider)
        if data is None or not data.get("access_token"):
            return AuthStatus.PENDING

        # Check expiry
        expires_at = data.get("expires_at")
        if expires_at:
            from datetime import datetime, timezone

            exp = datetime.fromisoformat(expires_at)
            if exp <= datetime.now(timezone.utc):
                return AuthStatus.EXPIRED

        return AuthStatus.AUTHENTICATED
    except Exception:
        return AuthStatus.PENDING


# =============================================================================
# Setup Command
# =============================================================================


@auth_app.command("setup")
def auth_setup() -> None:
    """Interactive wizard for first-time authentication setup.

    This wizard guides you through:
    1. Detecting local providers (Ollama, LM Studio)
    2. Choosing a cloud provider
    3. Selecting authentication method (API key vs OAuth)
    4. Collecting credentials
    5. Testing the connection
    6. Saving the account

    Example:
        victor auth setup
    """
    wizard = AuthSetupWizard(console)
    exit_code = wizard.run()
    raise typer.Exit(exit_code)


# =============================================================================
# Add Command
# =============================================================================


@auth_app.command("add")
def auth_add(
    provider: str = typer.Option(
        ..., "--provider", "-p", help="Provider name (e.g., anthropic, openai)"
    ),
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g., claude-sonnet-4-5)"),
    name: str = typer.Option("default", "--name", "-n", help="Account name"),
    auth_method: str = typer.Option(
        "api_key", "--auth-method", help="Authentication method (api_key, oauth, none)"
    ),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", "-e", help="Custom endpoint URL"),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key (will prompt if not provided)"
    ),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
) -> None:
    """Quick add a provider account.

    Example:
        victor auth add --provider anthropic --model claude-sonnet-4-5
        victor auth add --provider zai --model glm-4.6:coding --name glm-coding
        victor auth add --provider openai --model gpt-4o --auth-method oauth
    """
    # Parse tags
    tag_list = tags.split(",") if tags else []

    # Get API key if needed
    if auth_method == "api_key" and not api_key:
        api_key = Prompt.ask(f"Enter API key for {provider}", password=True)

    # Create auth config
    auth = AuthConfig(method=auth_method, source="keyring")
    if api_key:
        auth.value = api_key

    # Create account
    account = ProviderAccount(
        name=name,
        provider=provider,
        model=model,
        auth=auth,
        endpoint=endpoint,
        tags=tag_list,
    )

    # Save account
    manager = get_account_manager()
    manager.save_account(account)

    # Save API key to keyring if provided
    if api_key and auth_method == "api_key":
        try:
            from victor.config.api_keys import _set_key_in_keyring

            _set_key_in_keyring(provider, api_key)
            console.print("[green]✓[/] API key saved to keyring")
        except Exception as e:
            console.print(f"[yellow]⚠[/] Could not save to keyring: {e}")
            console.print("[dim]API key stored in config file instead[/]")

    console.print(f"[green]✓[/] Account '{name}' added successfully")
    console.print(f"[dim]Provider: {provider}[/]")
    console.print(f"[dim]Model: {model}[/]")


# =============================================================================
# List Command
# =============================================================================


@auth_app.command("list")
def auth_list() -> None:
    """List all configured accounts.

    Example:
        victor auth list
    """
    manager = get_account_manager()
    accounts = manager.list_accounts()

    if not accounts:
        console.print("[yellow]No accounts configured.[/]")
        console.print("[dim]Run 'victor auth setup' to add your first account.[/]")
        return

    # Determine default account name
    default_name = manager.load_config().defaults.account

    # Check OAuth token status for oauth accounts
    oauth_status: dict[str, str] = {}
    for account in accounts:
        if account.auth.method == "oauth":
            oauth_status[account.name] = _get_oauth_status(account.provider)

    table = Table(title="Configured Accounts", show_header=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Provider", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Auth", style="blue")
    table.add_column("Status", style="dim")
    table.add_column("Tags", style="dim")

    for account in sorted(accounts, key=lambda a: a.name):
        tags_str = ", ".join(account.tags) if account.tags else ""
        # Mark default account
        name_display = account.name
        if account.name == default_name:
            name_display = f"{account.name} ★"

        # Auth display with OAuth status
        auth_display = account.auth.method
        if account.auth.method == "oauth":
            status = oauth_status.get(account.name, "pending")
            status_display = (
                "[green]authenticated[/]"
                if status == "authenticated"
                else "[yellow]pending login[/]" if status == "pending" else f"[red]{status}[/]"
            )
        elif account.auth.method == "none":
            status_display = "[dim]local[/]"
        else:
            status_display = "[green]configured[/]"

        table.add_row(
            name_display,
            account.provider,
            account.model,
            auth_display,
            status_display,
            tags_str,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(accounts)} account(s)[/]")
    console.print("[dim]★ = default account[/]")


# =============================================================================
# Remove Command
# =============================================================================


@auth_app.command("remove")
def auth_remove(
    name: str = typer.Argument(..., help="Account name to remove"),
) -> None:
    """Remove a provider account.

    Example:
        victor auth remove my-account
    """
    manager = get_account_manager()

    # Confirm
    account = manager.get_account(name)
    if not account:
        console.print(f"[red]✗[/] Account '{name}' not found")
        raise typer.Exit(1)

    console.print("[yellow]Removing account:[/]")
    console.print(f"  Name: {account.name}")
    console.print(f"  Provider: {account.provider}")
    console.print(f"  Model: {account.model}")

    if not Confirm.ask("Continue?", default=False):
        console.print("[yellow]Cancelled[/]")
        return

    # Remove account
    if manager.remove_account(name):
        console.print(f"[green]✓[/] Account '{name}' removed")
    else:
        console.print("[red]✗[/] Failed to remove account")
        raise typer.Exit(1)


# =============================================================================
# Migrate Command
# =============================================================================


@auth_app.command("migrate")
def auth_migrate(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done"),
) -> None:
    """Migrate from old configuration format.

    Migrates from:
      ~/.victor/profiles.yaml
      ~/.victor/api_keys.yaml

    To:
      ~/.victor/config.yaml

    Example:
        victor auth migrate
        victor auth migrate --dry-run
        victor auth migrate --force
    """
    if not check_migration_needed():
        console.print("[dim]No old configuration found.[/]")
        console.print("[green]✓[/] Already using new configuration format")
        return

    console.print("[cyan]Migrating to new configuration format...[/]")
    console.print()

    # Show what will be migrated
    migrator = ConfigMigrator(dry_run=True)

    if migrator.old_profiles_file.exists():
        console.print(f"[dim]Found: {migrator.old_profiles_file}[/]")
    if migrator.old_api_keys_file.exists():
        console.print(f"[dim]Found: {migrator.old_api_keys_file}[/]")

    console.print()
    console.print(f"[dim]New config: {migrator.new_config_file}[/]")

    if dry_run:
        console.print()
        console.print("[yellow]DRY RUN[/] - No changes will be made")
        return

    if not Confirm.ask("Continue?", default=True):
        console.print("[yellow]Cancelled[/]")
        return

    # Run migration
    result = run_migration(prompt=False, force=force)

    if result.success:
        console.print("[green]✓[/] Migration successful!")
        console.print(f"[dim]Accounts migrated: {result.migrated_accounts}[/]")
        console.print(f"[dim]API keys migrated: {result.migrated_keys}[/]")
        console.print(f"[dim]Backup: {result.backup_path}[/]")

        if result.warnings:
            console.print()
            console.print("[yellow]Warnings:[/]")
            for warning in result.warnings:
                console.print(f"  ⚠ {warning}")
    else:
        console.print("[red]✗[/] Migration failed")
        for error in result.errors:
            console.print(f"  ✗ {error}")
        raise typer.Exit(1)


# =============================================================================
# Test Command
# =============================================================================


@auth_app.command("test")
def auth_test(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Account name to test"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Provider to test"),
) -> None:
    """Test a provider connection.

    Tests the connection to a provider account, validating:
    - Authentication configuration
    - API endpoint accessibility
    - Model availability

    Example:
        victor auth test
        victor auth test --name my-account
        victor auth test --provider anthropic
    """
    manager = get_account_manager()

    # Get account to test
    if name:
        account = manager.get_account(name)
        if not account:
            console.print(f"[red]✗[/] Account '{name}' not found")
            raise typer.Exit(1)
    elif provider:
        # First try as provider name
        accounts = [a for a in manager.list_accounts() if a.provider == provider]
        if accounts:
            account = accounts[0]
        else:
            # Fall back: treat as account name (user may have used -p instead of -n)
            account = manager.get_account(provider)
            if not account:
                console.print(f"[red]✗[/] No account or provider named '{provider}'")
                console.print("[dim]Hint: use --name/-n to test by account name[/]")
                raise typer.Exit(1)
            console.print(f"[dim]Matched account '{provider}' (hint: use -n for account names)[/]")
    else:
        # Use default account
        account = manager.get_account()
        if not account:
            console.print("[red]✗[/] No default account found")
            console.print("[dim]Run 'victor auth setup' to configure[/]")
            raise typer.Exit(1)

    console.print(f"[cyan]Testing connection to {account.provider}/{account.model}...[/]")
    console.print()

    # Run test
    validator = ConnectionValidator()
    result = validator.test_account_sync(account)

    # Display results
    if result.success:
        console.print("[green]✓[/] Connection successful!")
        console.print()

        # Show validation details
        for validation in result.validations:
            if validation.status == "success":
                console.print(f"  [green]✓[/] {validation.message}")
            elif validation.status == "warning":
                console.print(f"  [yellow]⚠[/] {validation.message}")
                if validation.details:
                    console.print(f"    [dim]{validation.details}[/]")

        if result.latency_ms:
            console.print()
            console.print(f"[dim]Latency: {result.latency_ms}ms[/]")
    else:
        console.print("[red]✗[/] Connection failed")
        console.print()

        if result.error:
            console.print(f"  [red]Error:[/] {result.error}")

        # Show validation details
        for validation in result.validations:
            if validation.status == "failed":
                console.print(f"  [red]✗[/] {validation.message}")
                if validation.details:
                    console.print(f"    [dim]{validation.details}[/]")
            elif validation.status == "warning":
                console.print(f"  [yellow]⚠[/] {validation.message}")

        raise typer.Exit(1)


# =============================================================================
# OAuth Commands (consolidated from victor providers auth)
# =============================================================================

OAUTH_SUPPORTED_PROVIDERS = ["openai", "qwen", "google", "github-copilot"]


@auth_app.command("login")
def auth_login(
    provider: str = typer.Argument(
        ..., help=f"Provider to authenticate ({', '.join(OAUTH_SUPPORTED_PROVIDERS)})"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-authentication even if token is cached"
    ),
) -> None:
    """Log in to a provider via OAuth (opens browser).

    Example:
        victor auth login openai
        victor auth login qwen --force
    """
    provider = provider.lower()
    if provider not in OAUTH_SUPPORTED_PROVIDERS:
        console.print(
            f"[red]\u2717[/] OAuth not supported for '{provider}'. "
            f"Supported: {', '.join(OAUTH_SUPPORTED_PROVIDERS)}"
        )
        raise typer.Exit(1)

    from victor.core.async_utils import run_sync
    from victor.providers.oauth_manager import OAuthTokenManager

    async def _login():
        mgr = OAuthTokenManager(provider)
        if not force:
            cached = mgr._load_cached()
            if cached is not None and not cached.is_expired:
                console.print(f"[green]\u2713[/] Already authenticated with {provider}")
                console.print(
                    f"  Token expires: {cached.expires_at.strftime('%Y-%m-%d %H:%M UTC')}"
                )
                console.print("[dim]Use --force to re-authenticate[/]")
                return
        console.print(f"[cyan]Opening browser for {provider} OAuth login...[/]")
        try:
            token = await mgr.get_valid_token()
            if token:
                console.print(f"[green]\u2713[/] Successfully authenticated with {provider}")
            else:
                console.print(f"[red]\u2717[/] Authentication failed for {provider}")
                raise typer.Exit(1)
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]\u2717[/] OAuth login failed: {e}")
            raise typer.Exit(1)

    run_sync(_login())


@auth_app.command("logout")
def auth_logout(
    provider: str = typer.Argument(
        ..., help=f"Provider to log out from ({', '.join(OAUTH_SUPPORTED_PROVIDERS)})"
    ),
) -> None:
    """Remove cached OAuth tokens for a provider.

    Example:
        victor auth logout openai
    """
    provider = provider.lower()
    if provider not in OAUTH_SUPPORTED_PROVIDERS:
        console.print(
            f"[red]\u2717[/] OAuth not supported for '{provider}'. "
            f"Supported: {', '.join(OAUTH_SUPPORTED_PROVIDERS)}"
        )
        raise typer.Exit(1)

    from victor.providers.oauth_manager import OAuthTokenManager

    mgr = OAuthTokenManager(provider)
    mgr._clear_cached()
    console.print(f"[green]\u2713[/] Logged out from {provider} (tokens cleared)")


@auth_app.command("oauth-status")
def auth_oauth_status(
    provider: Optional[str] = typer.Argument(
        None, help="Check specific provider (or all if omitted)"
    ),
) -> None:
    """Show OAuth authentication status for providers.

    Example:
        victor auth oauth-status
        victor auth oauth-status openai
    """
    from victor.providers.oauth_manager import OAuthTokenManager

    providers_to_check = [provider.lower()] if provider else OAUTH_SUPPORTED_PROVIDERS

    table = Table(title="OAuth Authentication Status", show_header=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Expires")
    table.add_column("Token Preview", style="dim")

    for prov in providers_to_check:
        if prov not in OAUTH_SUPPORTED_PROVIDERS:
            table.add_row(prov, "[red]Not supported[/]", "", "")
            continue

        mgr = OAuthTokenManager(prov)
        cached = mgr._load_cached()

        if cached is None:
            table.add_row(prov, "[dim]Not authenticated[/]", "", "")
        elif cached.is_expired:
            table.add_row(
                prov,
                "[yellow]Expired[/]",
                cached.expires_at.strftime("%Y-%m-%d %H:%M UTC") if cached.expires_at else "",
                "",
            )
        else:
            preview = cached.access_token[:8] + "..." if cached.access_token else ""
            table.add_row(
                prov,
                "[green]\u2713 Active[/]",
                cached.expires_at.strftime("%Y-%m-%d %H:%M UTC") if cached.expires_at else "",
                preview,
            )

    console.print(table)
    console.print("\n[dim]Login: victor auth login <provider>[/]")


# =============================================================================
# Auth Setup Wizard
# =============================================================================


class AuthSetupWizard:
    """Interactive authentication setup wizard."""

    def __init__(self, console: Console):
        """Initialize the wizard.

        Args:
            console: Rich console instance
        """
        self.console = console
        self.account_manager = get_account_manager()
        self.state: Dict[str, Any] = {
            "local_providers": [],
            "selected_provider": None,
            "selected_model": None,
            "auth_method": None,
        }

    def run(self) -> int:
        """Run the setup wizard.

        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            self._show_welcome()

            # Check for migration
            if self._check_migration():
                return 0

            # Step 1: Detect environment
            self._detect_environment()

            # Step 2: Choose provider type
            if not self._choose_provider_type():
                return 0

            # Step 3: Select provider and model
            if not self._select_provider_and_model():
                return 0

            # Step 4: Configure authentication
            if not self._configure_authentication():
                return 0

            # Step 5: Name the account
            if not self._name_account():
                return 0

            # Step 6: Test connection
            if not self._test_connection():
                return 0

            # Step 7: Save account
            self._save_account()

            # Step 8: Complete
            self._show_completion()

            return 0

        except KeyboardInterrupt:
            self.console.print("\n\n[yellow]Setup cancelled.[/]")
            return 0
        except Exception as e:
            self.console.print(f"\n[red]✗[/] An error occurred: {e}")
            return 1

    def _show_welcome(self) -> None:
        """Display welcome screen."""
        welcome_text = Text()
        welcome_text.append("Victor Authentication Setup\n\n", style="bold cyan")
        welcome_text.append(
            "This wizard will help you configure a provider account.\n",
            style="white",
        )

        panel = Panel.fit(
            welcome_text,
            title="[bold cyan]Setup[/]",
            border_style="cyan",
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    def _check_migration(self) -> bool:
        """Check if migration is needed and offer to migrate.

        Returns:
            True if migration was performed, False otherwise
        """
        if not check_migration_needed():
            return False

        self.console.print("[yellow]⚠ Old configuration detected[/]")
        self.console.print("[dim]We've upgraded our configuration system.[/]")
        self.console.print()

        if Confirm.ask("Migrate to new format?", default=True):
            result = run_migration(prompt=False)
            if result.success:
                self.console.print(f"[green]✓[/] Migrated {result.migrated_accounts} accounts")
                self.console.print()
                return True
            else:
                self.console.print("[red]✗[/] Migration failed")
                for error in result.errors:
                    self.console.print(f"  {error}")
                return True

        return False

    def _detect_environment(self) -> None:
        """Detect local providers and environment."""
        self.console.print("[bold cyan]Step 1/6: Environment Detection[/]")
        self.console.print("─" * 50)
        self.console.print()

        # Check Ollama
        self.console.print("[yellow]🔍[/] Checking for Ollama...")
        if self._check_ollama():
            self.console.print("  [green]✓[/] Ollama is running")
            self.state["local_providers"].append("ollama")
        else:
            self.console.print("  [dim]Ollama not detected[/]")

        # Check LM Studio
        self.console.print("[yellow]🔍[/] Checking for LM Studio...")
        if self._check_lmstudio():
            self.console.print("  [green]✓[/] LM Studio is running")
            self.state["local_providers"].append("lmstudio")
        else:
            self.console.print("  [dim]LM Studio not detected[/]")

        self.console.print()

    def _choose_provider_type(self) -> bool:
        """Choose between local and cloud provider.

        Returns:
            False if user cancelled
        """
        self.console.print("[bold cyan]Step 2/6: Provider Type[/]")
        self.console.print("─" * 50)
        self.console.print()

        if self.state["local_providers"]:
            self.console.print("[green]Found local providers:[/]")
            for provider in self.state["local_providers"]:
                self.console.print(f"  • {provider.capitalize()}")
            self.console.print()

            if Confirm.ask("Use a local provider?", default=False):
                self.state["provider_type"] = "local"
                return True

        self.console.print("[cyan]Cloud providers[/]")
        self.console.print("  • Anthropic (Claude)")
        self.console.print("  • OpenAI (GPT)")
        self.console.print("  • Google (Gemini)")
        self.console.print("  • ZhipuAI (GLM)")
        self.console.print("  • And 18 more...")
        self.console.print()

        if not Confirm.ask("Use a cloud provider?", default=True):
            return False

        self.state["provider_type"] = "cloud"
        return True

    def _select_provider_and_model(self) -> bool:
        """Select provider and model.

        Returns:
            False if user cancelled
        """
        self.console.print("[bold cyan]Step 3/6: Provider & Model[/]")
        self.console.print("─" * 50)
        self.console.print()

        if self.state["provider_type"] == "local":
            # Use first available local provider
            provider = self.state["local_providers"][0]
            self.state["selected_provider"] = provider
            self.state["selected_model"] = "default"
            self.console.print(f"[green]✓[/] Using {provider.capitalize()}")
            return True

        # Cloud provider selection
        self.console.print("[cyan]Popular providers:[/]")
        popular = ["anthropic", "openai", "google", "zai"]
        for i, p in enumerate(popular, 1):
            self.console.print(f"  {i}. {p.capitalize()}")

        self.console.print("  0. Other")
        self.console.print()

        choice = Prompt.ask(
            "Select provider",
            choices=[str(i) for i in range(len(popular) + 1)],
            default="1",
        )

        if choice == "0":
            # Show all providers
            provider = Prompt.ask("Enter provider name")
        else:
            provider = popular[int(choice)]

        self.state["selected_provider"] = provider.lower()

        # Model selection
        if provider.lower() in POPULAR_MODELS:
            models = POPULAR_MODELS[provider.lower()]
            self.console.print(f"\n[cyan]Popular {provider.capitalize()} models:[/]")
            for i, model in enumerate(models, 1):
                self.console.print(f"  {i}. {model}")
            self.console.print()

            model_choice = Prompt.ask(
                "Select model",
                choices=[str(i) for i in range(len(models) + 1)],
                default="1",
            )

            if model_choice != "0":
                self.state["selected_model"] = models[int(model_choice)]
            else:
                self.state["selected_model"] = Prompt.ask("Enter model name")
        else:
            self.state["selected_model"] = Prompt.ask("Enter model name")

        self.console.print()
        self.console.print(f"[green]✓[/] Selected: {provider}/{self.state['selected_model']}")
        return True

    def _configure_authentication(self) -> bool:
        """Configure authentication.

        Returns:
            False if user cancelled
        """
        self.console.print()
        self.console.print("[bold cyan]Step 4/6: Authentication[/]")
        self.console.print("─" * 50)
        self.console.print()

        provider = self.state["selected_provider"]

        # Local providers don't need auth
        if provider in AccountManager.LOCAL_PROVIDERS:
            self.state["auth_method"] = "none"
            self.console.print("[green]✓[/] No authentication required")
            return True

        # Check if OAuth is available
        if provider in OAUTH_PROVIDERS:
            self.console.print(f"[cyan]OAuth is available for {provider.capitalize()}[/]")
            use_oauth = Confirm.ask(
                "Use OAuth (recommended)?",
                default=True,
            )

            if use_oauth:
                self.state["auth_method"] = "oauth"
                self.console.print("[green]✓[/] Will use OAuth")
                return True

        # Default to API key
        self.state["auth_method"] = "api_key"
        self.console.print("[cyan]Enter your API key[/]")
        self.console.print("[dim]Your key will be stored securely in the system keyring[/]")

        api_key = Prompt.ask("API key", password=True)
        self.state["api_key"] = api_key

        self.console.print("[green]✓[/] API key received")
        return True

    def _name_account(self) -> bool:
        """Name the account.

        Returns:
            False if user cancelled
        """
        self.console.print()
        self.console.print("[bold cyan]Step 5/6: Account Name[/]")
        self.console.print("─" * 50)
        self.console.print()

        # Suggest a name
        provider = self.state["selected_provider"]
        model = self.state["selected_model"]

        if ":" in model:
            base_model, variant = model.rsplit(":", 1)
            suggestion = f"{provider}-{variant}"
        else:
            suggestion = provider

        name = Prompt.ask("Account name", default=suggestion)
        self.state["account_name"] = name

        # Ask for tags
        tags_input = Prompt.ask(
            "Tags (comma-separated, optional)",
            default="",
        )
        self.state["tags"] = [t.strip() for t in tags_input.split(",") if t.strip()]

        self.console.print(f"[green]✓[/] Account name: {name}")
        return True

    def _test_connection(self) -> bool:
        """Test the connection.

        Returns:
            False if test failed or user wants to retry
        """
        self.console.print()
        self.console.print("[bold cyan]Step 6/6: Test Connection[/]")
        self.console.print("─" * 50)
        self.console.print()

        provider = self.state["selected_provider"]
        model = self.state["selected_model"]
        auth_method = self.state["auth_method"]

        self.console.print(f"[cyan]Testing {provider}/{model}...[/]")
        self.console.print()

        # Create test account
        auth = AuthConfig(method=auth_method, source="keyring")
        if auth_method == "api_key" and "api_key" in self.state:
            auth.value = self.state["api_key"]

        account = ProviderAccount(
            name="test",
            provider=provider,
            model=model,
            auth=auth,
            tags=self.state.get("tags", []),
        )

        # Run test
        validator = ConnectionValidator()
        result = validator.test_account_sync(account)

        # Show results
        if result.success:
            self.console.print("[green]✓ Connection successful![/]")
            return True
        else:
            self.console.print("[red]✗ Connection failed[/]")
            if result.error:
                self.console.print(f"  Error: {result.error}")

            self.console.print()
            retry = Confirm.ask("Try again?", default=False)
            return not retry  # If not retrying, continue

    def _save_account(self) -> None:
        """Save the account."""
        self.console.print()
        self.console.print("[bold cyan]Saving Account[/]")
        self.console.print("─" * 50)
        self.console.print()

        # Create account
        auth = AuthConfig(method=self.state["auth_method"], source="keyring")
        if self.state["auth_method"] == "api_key" and "api_key" in self.state:
            auth.value = self.state["api_key"]

        account = ProviderAccount(
            name=self.state["account_name"],
            provider=self.state["selected_provider"],
            model=self.state["selected_model"],
            auth=auth,
            tags=self.state.get("tags", []),
        )

        # Save to config
        self.account_manager.save_account(account)

        # Save API key to keyring
        if self.state["auth_method"] == "api_key" and "api_key" in self.state:
            try:
                from victor.config.api_keys import _set_key_in_keyring

                _set_key_in_keyring(
                    self.state["selected_provider"],
                    self.state["api_key"],
                )
                self.console.print("[green]✓[/] API key saved to keyring")
            except Exception as e:
                self.console.print(f"[yellow]⚠[/] Could not save to keyring: {e}")

        # Set as default if it's the first account
        accounts = self.account_manager.list_accounts()
        if len(accounts) == 1:
            config = self.account_manager.load_config()
            config.defaults.account = account.name
            self.account_manager.save_config(config)
            self.console.print("[green]✓[/] Set as default account")

        self.console.print(f"[green]✓[/] Account '{account.name}' saved")

    def _show_completion(self) -> None:
        """Show completion message."""
        self.console.print()
        self.console.print("[bold green]✓ Setup Complete![/]")
        self.console.print()
        self.console.print("[cyan]Next steps:[/]")
        self.console.print("  • Start chatting: victor chat")
        self.console.print("  • List accounts: victor auth list")
        self.console.print("  • Test connection: victor auth test")
        self.console.print()

    # ========================================================================
    # Helper methods
    # ========================================================================

    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                timeout=2,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _check_lmstudio(self) -> bool:
        """Check if LM Studio is available."""
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", 1234))
            sock.close()
            return result == 0
        except Exception:
            return False
