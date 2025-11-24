"""Command-line interface for CodingAgent."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from codingagent import __version__
from codingagent.agent.orchestrator import AgentOrchestrator
from codingagent.config.settings import load_settings

app = typer.Typer(
    name="codingagent",
    help="Universal terminal-based coding agent supporting multiple LLM providers",
    add_completion=False,
)

console = Console()


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"CodingAgent v{__version__}")
        raise typer.Exit()


@app.command()
def main(
    message: Optional[str] = typer.Argument(
        None,
        help="Message to send to the agent (starts interactive mode if not provided)",
    ),
    profile: str = typer.Option(
        "default",
        "--profile",
        "-p",
        help="Profile to use from profiles.yaml",
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Stream responses",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """CodingAgent - Universal terminal-based coding agent.

    Examples:
        # Interactive mode
        codingagent

        # One-shot command
        codingagent "Write a Python function to calculate Fibonacci numbers"

        # Use specific profile
        codingagent --profile claude "Explain how async/await works"
    """
    # Load settings
    settings = load_settings()

    if message:
        # One-shot mode
        asyncio.run(run_oneshot(message, settings, profile, stream))
    else:
        # Interactive REPL mode
        asyncio.run(run_interactive(settings, profile, stream))


async def run_oneshot(
    message: str,
    settings: any,
    profile: str,
    stream: bool,
) -> None:
    """Run a single message and exit.

    Args:
        message: Message to send
        settings: Application settings
        profile: Profile name
        stream: Whether to stream response
    """
    try:
        agent = await AgentOrchestrator.from_settings(settings, profile)

        if stream and agent.provider.supports_streaming():
            async for chunk in agent.stream_chat(message):
                if chunk.content:
                    console.print(chunk.content, end="")
            console.print()  # New line at end
        else:
            response = await agent.chat(message)
            console.print(Markdown(response.content))

        await agent.provider.close()

    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(1)


async def run_interactive(
    settings: any,
    profile: str,
    stream: bool,
) -> None:
    """Run interactive REPL mode.

    Args:
        settings: Application settings
        profile: Profile name
        stream: Whether to stream responses
    """
    try:
        # Load profile info
        profiles = settings.load_profiles()
        profile_config = profiles.get(profile)

        if not profile_config:
            console.print(f"[bold red]Error:[/] Profile '{profile}' not found")
            raise typer.Exit(1)

        # Create agent
        agent = await AgentOrchestrator.from_settings(settings, profile)

        # Welcome message
        console.print(
            Panel(
                f"[bold]CodingAgent v{__version__}[/]\n\n"
                f"Provider: [cyan]{profile_config.provider}[/]\n"
                f"Model: [cyan]{profile_config.model}[/]\n\n"
                f"Type your message and press Enter to chat.\n"
                f"Type [bold]exit[/] or [bold]quit[/] to leave.\n"
                f"Type [bold]clear[/] to reset conversation.",
                title="Welcome",
                border_style="blue",
            )
        )

        # REPL loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold green]You[/]")

                # Handle special commands
                if user_input.lower() in ["exit", "quit"]:
                    console.print("[dim]Goodbye![/]")
                    break

                if user_input.lower() == "clear":
                    agent.reset_conversation()
                    console.print("[dim]Conversation cleared[/]")
                    continue

                if not user_input.strip():
                    continue

                # Send message and get response
                console.print("\n[bold blue]Assistant[/]")

                if stream and agent.provider.supports_streaming():
                    async for chunk in agent.stream_chat(user_input):
                        if chunk.content:
                            console.print(chunk.content, end="")
                    console.print()  # New line at end
                else:
                    response = await agent.chat(user_input)
                    console.print(Markdown(response.content))

            except KeyboardInterrupt:
                console.print("\n[dim]Use 'exit' or 'quit' to leave[/]")
                continue

            except EOFError:
                break

        await agent.provider.close()

    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def init() -> None:
    """Initialize configuration files."""
    from pathlib import Path
    import shutil

    config_dir = Path.home() / ".codingagent"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Copy example profiles if they don't exist
    profiles_file = config_dir / "profiles.yaml"
    if not profiles_file.exists():
        console.print(f"Creating default configuration at {profiles_file}")

        # Create a basic default profile
        default_config = """profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7
    max_tokens: 4096

providers:
  ollama:
    base_url: http://localhost:11434
"""
        profiles_file.write_text(default_config)
        console.print("[green]✓[/] Configuration created successfully!")
    else:
        console.print("[yellow]Configuration already exists[/]")

    console.print(f"\nConfiguration directory: {config_dir}")
    console.print(f"Edit {profiles_file} to customize profiles")


if __name__ == "__main__":
    app()
