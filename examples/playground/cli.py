"""Victor Playground CLI - Interactive learning environment."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from victor import Agent

app = typer.Typer(help="Victor Playground - Interactive learning environment")
console = Console()


@app.command()
def demo(
    example: str = typer.Option("hello", help="Example to run"),
    provider: str = typer.Option("openai", help="LLM provider"),
    model: Optional[str] = typer.Option(None, help="Model to use"),
):
    """Run a playground example."""

    examples = {
        "hello": "A simple greeting",
        "joke": "Tell a joke",
        "code": "Generate Python code",
        "explain": "Explain a concept",
        "creative": "Creative writing",
    }

    if example not in examples:
        console.print(f"[red]Unknown example: {example}[/red]")
        console.print(f"Available examples: {', '.join(examples.keys())}")
        raise typer.Exit(1)

    console.print(Panel(f"[bold blue]{example.upper()}[/bold blue]", expand=False))
    console.print(f"[dim]{examples[example]}[/dim]\n")

    # Create agent
    agent = Agent.create(provider=provider, model=model)

    # Run example
    prompts = {
        "hello": "Hello! Please introduce yourself.",
        "joke": "Tell me a short programming joke.",
        "code": "Write a Python function to calculate fibonacci numbers.",
        "explain": "Explain async/await in Python like I'm 12.",
        "creative": "Write a haiku about artificial intelligence.",
    }

    result = asyncio.run(agent.run(prompts[example]))

    console.print(Panel(result.content, title="[bold green]Response[/bold green]", expand=False))


@app.command()
def interactive(
    provider: str = typer.Option("openai", help="LLM provider"),
    tools: str = typer.Option("default", help="Tool preset (minimal/default/full)"),
):
    """Start an interactive session."""

    console.print(Panel(
        "[bold blue]Victor Playground - Interactive Mode[/bold blue]\n"
        "Type your messages below. Press Ctrl+D to exit.",
        title="Welcome"
    ))

    agent = Agent.create(provider=provider, tools=tools)

    console.print("\n[dim]Example prompts to try:[/dim]")
    console.print("  • What is the capital of France?")
    console.print("  • Write a Python hello world")
    console.print("  • Explain quantum computing")
    console.print("  • Tell me a joke\n")

    while True:
        try:
            # Get user input
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")

            if not user_input.strip():
                continue

            # Run agent
            with console.status("[dim]Thinking...[/dim]"):
                result = asyncio.run(agent.run(user_input))

            # Display response
            console.print(f"\n[bold green]Agent:[/bold green]")
            console.print(result.content)

        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


@app.command()
def examples():
    """List available examples."""

    table = Table(title="Available Examples")
    table.add_column("Example", style="cyan")
    table.add_column("Description", style="green")

    examples_list = [
        ("hello", "A simple greeting"),
        ("joke", "Tell a joke"),
        ("code", "Generate Python code"),
        ("explain", "Explain a concept"),
        ("creative", "Creative writing"),
    ]

    for name, desc in examples_list:
        table.add_row(name, desc)

    console.print(table)


@app.command()
def info():
    """Show playground information."""

    info_text = """
[bold blue]Victor Playground[/bold blue]

An interactive learning environment for the Victor AI Framework.

[bold]Commands:[/bold]
  [cyan]victor-examples demo[/cyan]      Run a pre-configured example
  [cyan]victor-examples interactive[/cyan] Start interactive chat mode
  [cyan]victor-examples examples[/cyan]   List available examples
  [cyan]victor-examples info[/cyan]       Show this information

[bold]Examples:[/bold]
  [cyan]demo hello[/cyan]       - Simple greeting
  [cyan]demo joke[/cyan]        - Tell a joke
  [cyan]demo code[/cyan]        - Generate Python code
  [cyan]demo explain[/cyan]     - Explain a concept
  [cyan]demo creative[/cyan]    - Creative writing

[bold]Providers:[/bold]
  [cyan]openai[/cyan]    - OpenAI (GPT-4, GPT-3.5)
  [cyan]anthropic[/cyan] - Anthropic (Claude)
  [cyan]ollama[/cyan]     - Local models (no API key)

For more information, visit: https://github.com/vjsingh1984/victor
"""

    console.print(info_text)


if __name__ == "__main__":
    app()
