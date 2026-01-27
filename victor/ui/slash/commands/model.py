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

"""Model and provider slash commands: model, profile, provider."""

from __future__ import annotations

import logging

from rich.panel import Panel
from rich.table import Table

from victor.ui.common.constants import FIRST_ARG_INDEX, FIRST_MATCH_INDEX, SECOND_ARG_INDEX
from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class ModelCommand(BaseSlashCommand):
    """List available models or switch model."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="model",
            description="List available models or switch model",
            usage="/model [model_name]",
            aliases=["models"],
            category="model",
            requires_agent=True,
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        if ctx.args:
            # Switch model
            model_name = ctx.args[FIRST_ARG_INDEX]
            if ctx.agent and getattr(ctx.agent, 'switch_model', lambda x: False)(model_name):
                info = ctx.agent.get_current_provider_info()  # type: ignore[attr-defined]
                ctx.console.print(f"[green]Switched to model:[/] [cyan]{model_name}[/]")
                ctx.console.print(
                    f"  [dim]Native tools: {info['native_tool_calls']}, "
                    f"Thinking: {info['thinking_mode']}[/]"
                )
            else:
                ctx.console.print(f"[red]Failed to switch model to {model_name}[/]")
            return

        # List available models
        ctx.console.print("[dim]Fetching available models...[/]")

        try:
            from victor.providers.ollama_provider import OllamaProvider

            provider_settings = ctx.settings.get_provider_settings("ollama")
            ollama = OllamaProvider(**provider_settings)

            models_list = await ollama.list_models()

            if not models_list:
                ctx.console.print("[yellow]No models found[/]")
                ctx.console.print("Pull a model: [bold]ollama pull qwen2.5-coder:7b[/]")
                await ollama.close()
                return

            table = Table(title="Available Models", show_header=True)
            table.add_column("Model", style="cyan")
            table.add_column("Size", style="yellow")
            table.add_column("Status", style="green")

            current_model = getattr(ctx.agent, 'model', None) if ctx.agent else None

            for model in models_list:
                name = model.get("name", "unknown")
                size = model.get("size", 0)
                size_gb = size / (1024**3) if size else 0

                status = "current" if name == current_model else ""
                table.add_row(
                    name,
                    f"{size_gb:.1f} GB" if size_gb > 0 else "?",
                    status,
                )

            ctx.console.print(table)
            ctx.console.print("\n[dim]Switch model: /model <model_name>[/]")

            await ollama.close()

        except Exception as e:
            ctx.console.print(f"[red]Error listing models:[/] {e}")
            ctx.console.print("Make sure Ollama is running")


@register_command
class ProfileCommand(BaseSlashCommand):
    """Show or switch profile."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="profile",
            description="Show or switch profile",
            usage="/profile [profile_name]",
            aliases=["profiles"],
            category="model",
        )

    def execute(self, ctx: CommandContext) -> None:
        profiles = ctx.settings.load_profiles()

        if ctx.args:
            # Switch to specified profile
            profile_name = ctx.args[FIRST_ARG_INDEX]
            if profile_name not in profiles:
                ctx.console.print(f"[red]Profile not found:[/] {profile_name}")
                ctx.console.print(f"Available: {', '.join(profiles.keys())}")
                return

            if ctx.agent is None:
                ctx.console.print("[yellow]No active session to switch profile[/]")
                return

            profile_config = profiles[profile_name]

            ctx.console.print(
                f"[dim]Switching to profile '{profile_name}' "
                f"({profile_config.provider}:{profile_config.model_name})...[/]"
            )

            if ctx.agent and getattr(ctx.agent, 'switch_provider', lambda p, m=None: False)(
                profile_config.provider,
                profile_config.model_name,
            ):
                info = ctx.agent.get_current_provider_info()  # type: ignore[attr-defined]
                ctx.console.print(f"[green]Switched to profile:[/] [cyan]{profile_name}[/]")
                ctx.console.print(f"  [dim]Provider: {info['provider']}, Model: {info['model']}[/]")
                ctx.console.print(
                    f"  [dim]Native tools: {info['native_tool_calls']}, "
                    f"Thinking: {info['thinking_mode']}[/]"
                )
            else:
                ctx.console.print(f"[red]Failed to switch to profile {profile_name}[/]")
                ctx.console.print(
                    "[yellow]You may need to restart: "
                    f"[bold]victor --profile {profile_name}[/][/]"
                )
            return

        # Show profiles
        current_provider = getattr(ctx.agent, 'provider_name', None) if ctx.agent else None
        current_model = getattr(ctx.agent, 'model', None) if ctx.agent else None

        # Get RL Q-values for provider ranking
        rl_rankings = {}
        rl_best_provider = None
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if learner:
                rankings = learner.get_provider_rankings()  # type: ignore[attr-defined]
                rl_rankings = {r["provider"].lower(): r["q_value"] for r in rankings}
                if rankings:
                    rl_best_provider = rankings[FIRST_MATCH_INDEX]["provider"].lower()
        except Exception:
            pass

        table = Table(title="Configured Profiles", show_header=True)
        table.add_column("Profile", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Model", style="yellow")
        table.add_column("Q-Value", style="magenta", justify="right")
        table.add_column("Status", style="dim")

        for name, config in profiles.items():
            is_current = config.provider == current_provider and config.model == current_model
            status = "current" if is_current else ""

            provider_lower = config.provider.lower()
            if provider_lower == rl_best_provider and not is_current:
                status = "RL best" if not status else f"{status}, RL best"

            q_val = rl_rankings.get(provider_lower)
            q_str = f"{q_val:.2f}" if q_val is not None else "-"

            table.add_row(name, config.provider, config.model, q_str, status)

        ctx.console.print(table)
        ctx.console.print("\n[dim]Switch profile: /profile <name>[/]")
        if rl_rankings:
            ctx.console.print("[dim]RL recommends based on historical performance[/]")


@register_command
class ProviderCommand(BaseSlashCommand):
    """Show current provider info or switch provider."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="provider",
            description="Show current provider info or switch provider",
            usage="/provider [provider_name]",
            aliases=["providers"],
            category="model",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        from victor.providers.registry import ProviderRegistry

        info = ctx.agent.get_current_provider_info() if ctx.agent else {}  # type: ignore[attr-defined]
        available_providers = ProviderRegistry.list_providers()

        if not ctx.args:
            # Show current provider info
            ctx.console.print(
                Panel(
                    f"[bold]Provider:[/] [cyan]{info.get('provider', 'unknown')}[/]\n"
                    f"[bold]Model:[/] [yellow]{info.get('model', 'unknown')}[/]\n"
                    f"[bold]Supports Tools:[/] {info.get('supports_tools', False)}\n"
                    f"[bold]Native Tool Calls:[/] {info.get('native_tool_calls', False)}\n"
                    f"[bold]Thinking Mode:[/] {info.get('thinking_mode', False)}\n"
                    f"[bold]Tool Budget:[/] {info.get('tool_budget', 'N/A')}",
                    title="Current Provider",
                    border_style="cyan",
                )
            )

            # Show RL rankings
            try:
                from victor.framework.rl.coordinator import get_rl_coordinator

                coordinator = get_rl_coordinator()
                learner = coordinator.get_learner("model_selector")
                if learner:
                    rankings = learner.get_provider_rankings()  # type: ignore[attr-defined]
                    if rankings:
                        ctx.console.print("\n[bold]Provider Rankings (RL):[/]")
                        for r in rankings[:5]:
                            indicator = (
                                "current"
                                if r["provider"].lower() == info.get("provider", "").lower()
                                else ""
                            )
                            ctx.console.print(
                                f"  {r['provider']}: Q={r['q_value']:.2f} "
                                f"(samples: {r['sample_count']}) {indicator}"
                            )
            except Exception:
                pass

            ctx.console.print(f"\n[dim]Available providers: {', '.join(available_providers)}[/]")
            ctx.console.print("[dim]Switch provider: /provider <name>[/]")
            return

        # Switch provider
        provider_name = ctx.args[FIRST_ARG_INDEX]
        model = ctx.args[SECOND_ARG_INDEX] if len(ctx.args) > 1 else None

        # Handle provider:model syntax
        if ":" in provider_name and model is None:
            provider_name, model = provider_name.split(":", 1)

        if provider_name not in available_providers:
            ctx.console.print(f"[red]Unknown provider:[/] {provider_name}")
            ctx.console.print(f"Available: {', '.join(available_providers)}")
            return

        ctx.console.print(f"[dim]Switching to {provider_name}...[/]")

        if ctx.agent and getattr(ctx.agent, 'switch_provider', lambda p, m=None: False)(provider_name, model):
            info = ctx.agent.get_current_provider_info()  # type: ignore[attr-defined]
            ctx.console.print(f"[green]Switched to:[/] {info['provider']}:{info['model']}")
            ctx.console.print(
                f"  [dim]Native tools: {info['native_tool_calls']}, "
                f"Thinking: {info['thinking_mode']}[/]"
            )
        else:
            ctx.console.print(f"[red]Failed to switch to {provider_name}[/]")
