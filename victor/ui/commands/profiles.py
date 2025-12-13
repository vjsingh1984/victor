import typer
from rich.console import Console
from rich.table import Table

from victor.config.settings import load_settings

profiles_app = typer.Typer(name="profiles", help="List configured profiles.")
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

    for name, profile in profiles.items():
        table.add_row(
            name,
            profile.provider,
            profile.model,
            f"{profile.temperature}",
            f"{profile.max_tokens}",
        )

    console.print(table)
    console.print(f"\n[dim]Config file: {settings.get_config_dir() / 'profiles.yaml'}[/]")
