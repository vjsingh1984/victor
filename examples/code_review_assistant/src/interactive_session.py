"""
Interactive review session for real-time code analysis.
"""

import asyncio
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.syntax import Syntax

from .review_engine import ReviewEngine


console = Console()


class InteractiveSession:
    """Interactive code review session."""

    def __init__(self, orchestrator, config):
        """Initialize interactive session.

        Args:
            orchestrator: Victor AI agent orchestrator
            config: Review configuration
        """
        self.orchestrator = orchestrator
        self.config = config
        self.engine = ReviewEngine(orchestrator, config)
        self.running = True

    async def run(self):
        """Run interactive session."""
        console.print("\n[green]Interactive session started![/green]")
        console.print(
            "Commands: [bold]review[/bold], [bold]ask[/bold], [bold]help[/bold], [bold]quit[/bold]\n"
        )

        while self.running:
            try:
                # Get user input
                command = Prompt.ask("[bold cyan]victor-review[/bold cyan]", default="help")

                # Process command
                await self._process_command(command)

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'quit' to exit[/yellow]")
            except EOFError:
                self.running = False

    async def _process_command(self, command: str):
        """Process user command."""
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "quit" or cmd == "exit":
            self.running = False
            console.print("[yellow]Goodbye![/yellow]")

        elif cmd == "help":
            self._show_help()

        elif cmd == "review":
            await self._review_command(args)

        elif cmd == "ask":
            await self._ask_command(args)

        else:
            console.print(f"[red]Unknown command: {cmd}[/red]")
            console.print("Type 'help' for available commands")

    def _show_help(self):
        """Display help information."""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

  [bold]review <path>[/bold]
    Review a file or directory
    Example: review src/main.py

  [bold]ask <question>[/bold]
    Ask a question about your code
    Example: ask How can I improve the error handling?

  [bold]help[/bold]
    Show this help message

  [bold]quit[/bold]
    Exit the interactive session
        """
        console.print(help_text)

    async def _review_command(self, args: str):
        """Handle review command."""
        if not args:
            console.print("[red]Please specify a file or directory to review[/red]")
            return

        path = Path(args.strip())
        if not path.exists():
            console.print(f"[red]Path not found: {args}[/red]")
            return

        console.print(f"\n[cyan]Reviewing: {path}[/cyan]\n")

        results = await self.engine.review(path)

        # Display summary
        console.print(
            Panel(
                f"[bold]Files Analyzed:[/bold] {results['files_analyzed']}\n"
                f"[bold]Total Issues:[/bold] {results['total_issues']}\n"
                f"[red][bold]Critical:[/bold] {results['critical']}[/red]\n"
                f"[orange1][bold]High:[/bold] {results['high']}[/orange1]\n"
                f"[yellow][bold]Medium:[/bold] {results['medium']}[/yellow]\n"
                f"[blue][bold]Low:[/bold] {results['low']}[/blue]",
                title="Review Results",
                border_style="cyan",
            )
        )

        # Show top issues
        if results["top_issues"]:
            console.print("\n[bold yellow]Top Issues:[/bold yellow]\n")
            for issue in results["top_issues"][:5]:
                console.print(f"[{issue['severity']}] {issue['message']}")
                console.print(f"  → {issue['file']}:{issue['line']}\n")

    async def _ask_command(self, args: str):
        """Handle ask command."""
        if not args:
            console.print("[red]Please provide a question[/red]")
            return

        question = args.strip()
        console.print(f"\n[cyan]Question: {question}[/cyan]\n")
        console.print("[dim]Thinking...[/dim]\n")

        # Use Victor AI orchestrator to answer
        response = await self.orchestrator.process_request(
            question, context={"mode": "code_review"}
        )

        # Display response
        console.print(Panel(response, title="Victor AI Response", border_style="green"))
