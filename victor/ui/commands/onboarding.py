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

"""Interactive onboarding wizard for first-time Victor setup.

This module provides an interactive wizard that guides new users through:
1. Welcome and introduction to Victor
2. Environment detection (Ollama, API keys, etc.)
3. Profile selection based on experience level
4. Provider configuration
5. Configuration validation
6. Testing the setup
7. Starting first chat

The wizard uses Rich for beautiful terminal UI with panels, tables, and prompts.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm, Prompt
from rich.text import Text

from victor.config.profiles import (
    ProfileLevel,
    PROFILES,
    get_profile,
    get_recommended_profile,
    install_profile,
)


class OnboardingWizard:
    """Interactive onboarding wizard for Victor setup."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the wizard.

        Args:
            console: Optional Rich console instance (creates default if None)
        """
        self.console = console or Console()
        self.config_dir = Path.home() / ".victor"
        self.state: Dict[str, Any] = {
            "step": 0,
            "selected_profile": None,
            "provider": None,
            "model": None,
            "api_keys_configured": False,
            "ollama_available": False,
            "has_cloud_keys": False,
        }

    def run(self) -> int:
        """Run the complete onboarding wizard.

        Returns:
            Exit code (0 for success, 1 for cancellation/error)
        """
        try:
            # Removed console.clear() - jarring UX, destroys user context
            self._show_welcome()

            if not self._confirm_start():
                self.console.print("\n[yellow]Onboarding cancelled.[/]")
                return 0

            # Step 1: Environment Detection
            self._detect_environment()

            # Step 2: Profile Selection
            if not self._select_profile():
                return 0

            # Step 3: Provider Configuration
            if not self._configure_provider():
                return 0

            # Step 4: Apply Configuration
            self._apply_configuration()

            # Step 5: Validate Configuration
            if not self._validate_configuration():
                return 0

            # Step 6: Complete and Next Steps
            self._show_completion()

            return 0

        except KeyboardInterrupt:
            self.console.print("\n\n[yellow]Onboarding cancelled.[/]")
            return 0
        except Exception as e:
            self.console.print(f"\n[red]✗[/] An error occurred: {e}")
            return 1

    def _show_welcome(self) -> None:
        """Display welcome screen."""
        welcome_text = Text()
        welcome_text.append("Welcome to ", style="white")
        welcome_text.append("Victor", style="bold cyan")
        welcome_text.append("! ", style="white")
        welcome_text.append("Open-source Agentic AI Framework\n\n", style="dim")

        features = Table(show_header=False, box=None, padding=(0, 2))
        features.add_column("", style="cyan")
        features.add_column("", style="white")

        features.add_row("✦", "22 LLM provider adapters")
        features.add_row("✦", "33 tool modules")
        features.add_row("✦", "Multi-agent coordination")
        features.add_row("✦", "Domain-specific verticals")
        features.add_row("✦", "YAML workflow engine")

        panel = Panel.fit(
            welcome_text + "\n" + features,
            title="[bold cyan]Victor Setup Wizard[/]",
            border_style="cyan",
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    def _confirm_start(self) -> bool:
        """Ask user to confirm starting onboarding.

        Returns:
            True if user wants to continue, False otherwise
        """
        # Check if config already exists
        profiles_path = self.config_dir / "profiles.yaml"
        if profiles_path.exists():
            self.console.print("[yellow]⚠[/] Configuration already exists!")
            self.console.print(f"[dim]Found: {profiles_path}[/]")
            self.console.print()

            if not Confirm.ask("Would you like to reconfigure Victor?", default=False):
                return False

        return Confirm.ask("Ready to set up Victor?", default=True)

    def _detect_environment(self) -> None:
        """Step 1: Detect the user's environment."""
        self.console.print("\n[bold cyan]Step 1/5: Environment Detection[/]")
        self.console.print("─" * 50)

        # Check Ollama
        self.console.print("\n[yellow]🔍[/] Checking for Ollama (local models)...")
        self.state["ollama_available"] = self._check_ollama()

        if self.state["ollama_available"]:
            self.console.print("  [green]✓[/] Ollama is running")
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    models = result.stdout.decode().strip().split("\n")[1:]  # Skip header
                    if models:
                        self.console.print(f"  [dim]Found {len(models)} model(s)[/]")
            except Exception:
                pass
        else:
            self.console.print("  [dim]Ollama not running[/]")
            self.console.print("  [dim]Install: https://ollama.com[/]")

        # Check for API keys
        self.console.print("\n[yellow]🔍[/] Checking for cloud provider API keys...")
        api_keys = self._check_api_keys()
        self.state["has_cloud_keys"] = bool(api_keys)

        if api_keys:
            self.console.print(f"  [green]✓[/] Found API keys for: {', '.join(api_keys)}")
        else:
            self.console.print("  [dim]No API keys configured[/]")

        # Recommendation
        recommended = get_recommended_profile()
        self.console.print(f"\n💡 [yellow]Recommended profile:[/] {recommended.display_name}")

    def _check_ollama(self) -> bool:
        """Check if Ollama is available and running.

        Returns:
            True if Ollama is running
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=3,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        except Exception:
            return False

    def _check_api_keys(self) -> List[str]:
        """Check for configured API keys.

        Returns:
            List of provider names with configured keys
        """
        providers = {
            "ANTHROPIC_API_KEY": "Anthropic",
            "OPENAI_API_KEY": "OpenAI",
            "GOOGLE_API_KEY": "Google",
            "AZURE_API_KEY": "Azure",
            "XAI_API_KEY": "xAI",
            "COHERE_API_KEY": "Cohere",
        }

        found = []
        for env_var, provider in providers.items():
            if os.getenv(env_var):
                found.append(provider)

        return found

    def _select_profile(self) -> bool:
        """Step 2: Profile selection.

        Returns:
            False if user cancels, True otherwise
        """
        self.console.print("\n[bold cyan]Step 2/5: Profile Selection[/]")
        self.console.print("─" * 50)

        # Show available profiles
        self.console.print("\n[bold]Available Profiles:[/]\n")

        for level in [ProfileLevel.BASIC, ProfileLevel.ADVANCED, ProfileLevel.EXPERT]:
            profiles = [p for p in PROFILES.values() if p.level == level]
            if not profiles:
                continue

            level_style = {
                ProfileLevel.BASIC: "green",
                ProfileLevel.ADVANCED: "yellow",
                ProfileLevel.EXPERT: "red",
            }.get(level, "white")

            for profile in profiles:
                self.console.print(f"  [{level_style}]{profile.display_name}[/]")
                self.console.print(f"  [dim]{profile.description}[/]")
                self.console.print()

        # Get recommendation
        recommended = get_recommended_profile()

        # Prompt for selection
        self.console.print(f"💡 [yellow]Recommended for you:[/] {recommended.display_name}\n")

        choices = [p.name for p in PROFILES.values()]
        choice = Prompt.ask(
            "Choose a profile",
            choices=choices,
            default=recommended.name,
            show_choices=True,
        )

        if choice is None:
            return False

        self.state["selected_profile"] = get_profile(choice)
        return True

    def _configure_provider(self) -> bool:
        """Step 3: Provider configuration.

        Returns:
            False if user cancels, True otherwise
        """
        self.console.print("\n[bold cyan]Step 3/5: Provider Configuration[/]")
        self.console.print("─" * 50)

        provider_options = [
            "ollama",
            "anthropic",
            "openai",
            "google",
            "lmstudio",
            "auto",
        ]

        # Default to auto or based on environment
        if self.state["has_cloud_keys"]:
            default_provider = "auto"
        elif self.state["ollama_available"]:
            default_provider = "ollama"
        else:
            default_provider = "auto"

        self.console.print("\n[bold]Available providers:[/]")
        self.console.print("  [cyan]ollama[/] - Local models (free, private)")
        self.console.print("  [cyan]anthropic[/] - Claude (API key required)")
        self.console.print("  [cyan]openai[/] - GPT (API key required)")
        self.console.print("  [cyan]google[/] - Gemini (API key required)")
        self.console.print("  [cyan]auto[/] - Auto-detect best option\n")

        provider = Prompt.ask(
            "Select provider",
            choices=provider_options,
            default=default_provider,
            show_choices=True,
        )

        if provider is None:
            return False

        if provider == "auto":
            provider = self._auto_detect_provider()

        self.state["provider"] = provider

        # Model selection
        self.console.print(f"\n[bold]Select model for {provider}:[/]")
        models = self._get_models_for_provider(provider)

        if models:
            # Default to qwen3.5:27b-q4_K_M if available (first in list)
            default_model = "qwen3.5:27b-q4_K_M"
            for i, model in enumerate(models[:5], 1):  # Show first 5
                desc = model.get("description", "")
                self.console.print(f"  {i}. [cyan]{model['id']}[/] - {desc}")

            model_choice = Prompt.ask(
                "Choose model (1-5 or model name)",
                default=default_model,
            )

            # Try to parse as number
            try:
                model_num = int(model_choice)
                if 1 <= model_num <= len(models):
                    self.state["model"] = models[model_num - 1]["id"]
                else:
                    self.state["model"] = model_choice
            except ValueError:
                self.state["model"] = model_choice
        else:
            # Use default model
            self.state["model"] = self._get_default_model(provider)
            self.console.print(f"\n[dim]Using default model: {self.state['model']}[/]")

        return True

    def _auto_detect_provider(self) -> str:
        """Auto-detect the best available provider.

        Returns:
            Provider name
        """
        if self.state["has_cloud_keys"]:
            # Prefer Anthropic, then OpenAI
            if os.getenv("ANTHROPIC_API_KEY"):
                return "anthropic"
            if os.getenv("OPENAI_API_KEY"):
                return "openai"

        # Default to Ollama
        return "ollama"

    def _get_models_for_provider(self, provider: str) -> List[Dict[str, str]]:
        """Get available models for a provider.

        Args:
            provider: Provider name

        Returns:
            List of model dictionaries with 'id' and 'description'
        """
        models = {
            "ollama": [
                {
                    "id": "qwen3.5:27b-q4_K_M",
                    "description": "MoE model, fast + knowledgeable (recommended)",
                },
                {"id": "qwen2.5-coder:7b", "description": "Coding-focused, 7B params"},
                {
                    "id": "qwen2.5-coder:14b",
                    "description": "Coding-focused, 14B params",
                },
                {"id": "qwen2.5:7b", "description": "General purpose, 7B params"},
                {"id": "llama3.2:3b", "description": "Fast, efficient, 3B params"},
            ],
            "anthropic": [
                {"id": "claude-3-5-sonnet-20241022", "description": "Powerful, newer"},
            ],
            "openai": [
                {"id": "gpt-4o", "description": "Fast, multimodal"},
                {"id": "gpt-4o-mini", "description": "Very fast, efficient"},
            ],
            "google": [
                {"id": "gemini-2.0-flash-exp", "description": "Very fast, experimental"},
                {"id": "gemini-1.5-pro", "description": "Balanced"},
            ],
            "lmstudio": [
                {"id": "local-model", "description": "Your loaded model"},
            ],
            "vllm": [
                {"id": "local-model", "description": "Your loaded model"},
            ],
            "deepseek": [
                {"id": "deepseek-chat", "description": "Fast, efficient"},
            ],
            "xai": [
                {"id": "grok-beta", "description": "Fast, capable"},
            ],
            "zai": [
                {"id": "glm-4.7", "description": "Fast, cost-effective"},
            ],
            "cohere": [
                {"id": "command-r-plus", "description": "Command-optimized"},
            ],
        }
        return models.get(provider, [])

    def _get_default_model(self, provider: str) -> str:
        """Get default model for a provider.

        Args:
            provider: Provider name

        Returns:
            Default model identifier
        """
        defaults = {
            "ollama": "qwen3.5:27b-q4_K_M",  # Fast MoE model
            "anthropic": "claude-sonnet-4-5-20250514",
            "openai": "gpt-4o",
            "google": "gemini-2.0-flash-exp",
            "lmstudio": "local-model",
        }
        return defaults.get(provider, "qwen3.5:27b-q4_K_M")

    def _apply_configuration(self) -> None:
        """Step 4: Apply configuration."""
        self.console.print("\n[bold cyan]Step 4/5: Apply Configuration[/]")
        self.console.print("─" * 50)

        profile = self.state["selected_profile"]
        provider = self.state["provider"]
        model = self.state["model"]

        self.console.print("\n[yellow]⚙️[/] Applying configuration...")
        self.console.print(f"  Profile: [cyan]{profile.display_name}[/]")
        self.console.print(f"  Provider: [cyan]{provider}[/]")
        self.console.print(f"  Model: [cyan]{model}[/]")

        try:
            profiles_path = install_profile(
                profile,
                config_dir=self.config_dir,
                provider_override=provider,
                model_override=model,
            )
            self.console.print("\n  [green]✓[/] Configuration saved to:")
            self.console.print(f"  [dim]{profiles_path}[/]")

            # Create onboarding completion marker
            marker_file = self.config_dir / ".onboarding_completed"
            completed_at = datetime.now().isoformat()
            marker_file.write_text(
                f"# Onboarding completed successfully\n"
                f"# Completed at: {completed_at}\n"
                f"# Profile: {profile.name}\n"
                f"# Provider: {provider}\n"
                f"# Model: {model}\n"
            )

        except Exception as e:
            self.console.print(f"\n  [red]✗[/] Failed to save configuration: {e}")
            raise

    def _validate_configuration(self) -> bool:
        """Step 5: Validate configuration.

        Returns:
            False if validation fails, True otherwise
        """
        self.console.print("\n[bold cyan]Step 5/5: Validate Configuration[/]")
        self.console.print("─" * 50)

        self.console.print("\n[yellow]🔍[/] Validating configuration...")

        # Import validation
        try:
            from victor.config.validation import validate_configuration
            from victor.config.settings import load_settings
        except ImportError:
            self.console.print("  [yellow]⚠[/] Validation module not available")
            return True

        try:
            settings = load_settings()
            result = validate_configuration(settings)

            if result.is_valid:
                self.console.print("  [green]✓[/] Configuration is valid!")
            else:
                self.console.print("  [yellow]⚠[/] Configuration has warnings:")
                for error in result.errors:
                    self.console.print(f"    [dim]• {error.message}[/]")
                # Continue despite warnings

        except Exception as e:
            self.console.print(f"  [yellow]⚠[/] Validation skipped: {e}")

        # Test provider connection
        self.console.print("\n[yellow]🔍[/] Testing provider connection...")
        provider = self.state["provider"]

        if provider == "ollama":
            if self.state["ollama_available"]:
                self.console.print("  [green]✓[/] Ollama connection OK")
            else:
                self.console.print("  [yellow]⚠[/] Ollama not running - start with: ollama serve")
        elif provider in ["anthropic", "openai", "google"]:
            if self.state["has_cloud_keys"]:
                self.console.print(f"  [green]✓[/] {provider.title()} API key configured")
            else:
                self.console.print(f"  [yellow]⚠[/] {provider.title()} API key not set")

        return True

    def _show_completion(self) -> None:
        """Show completion screen and next steps."""
        self.console.print("\n[bold cyan]✓ Setup Complete![/]")
        self.console.print("═" * 50)

        # Summary table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("", style="yellow")
        table.add_column("", style="white")

        profile = self.state["selected_profile"]
        table.add_row("Profile", profile.display_name)
        table.add_row("Provider", self.state["provider"])
        table.add_row("Model", self.state["model"])
        table.add_row("Config", str(self.config_dir / "profiles.yaml"))

        self.console.print("\n[bold]Your Configuration:[/]")
        self.console.print(table)

        # Next steps
        self.console.print("\n[bold]Next Steps:[/]")
        self.console.print("  1. [cyan]victor doctor[/] - Run diagnostics")
        self.console.print("  2. [cyan]victor chat[/] - Start chatting")
        self.console.print("  3. [cyan]victor profile list[/] - See other profiles")
        self.console.print("\n[dim]For more information:[/]")
        self.console.print("  [dim]https://github.com/victor-ai/victor[/]")

        # Offer to start chat
        self.console.print()
        if Confirm.ask("Start your first chat now?", default=False):
            self._start_first_chat()

    def _start_first_chat(self) -> None:
        """Start the first chat session."""
        self.console.print("\n[yellow]Starting Victor chat...[/]\n")
        self.console.print("[dim]Type your message and press Enter. Type 'quit' to exit.[/]\n")

        try:
            from victor.ui.commands.chat import _run_default_interactive

            _run_default_interactive()
        except Exception as e:
            self.console.print(f"\n[yellow]Chat ended: {e}[/]")


def run_onboarding() -> int:
    """Run the onboarding wizard.

    This is the entry point for 'victor init' command.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    console = Console()
    try:
        wizard = OnboardingWizard(console)
        return wizard.run()
    except Exception as e:
        console.print(f"\n[red]✗[/] Onboarding failed: {e}")
        return 1
