import typer
import asyncio
from rich.console import Console
from rich.prompt import Confirm, Prompt
from pathlib import Path
from typing import Optional, List

from victor.config.settings import get_project_paths, VICTOR_CONTEXT_FILE, VICTOR_DIR_NAME
from victor.core.database import get_database, get_project_database

init_app = typer.Typer(name="init", help="Initialize project context and configuration.")
console = Console()


@init_app.callback(invoke_without_command=True)
def init(
    ctx: typer.Context,
    update: bool = typer.Option(
        False, "--update", "-u", help="Update existing init.md preserving user edits"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing init.md completely"
    ),
    learn: bool = typer.Option(
        True, "--learn/--no-learn", "-L", help="Include conversation history insights (default: on)"
    ),
    index: bool = typer.Option(
        False, "--index", "-i", help="Use SQLite symbol store only (no LLM)"
    ),
    deep: bool = typer.Option(
        True, "--deep/--no-deep", "-d", help="Use LLM for deep analysis (default: on)"
    ),
    quick: bool = typer.Option(
        False, "--quick", "-q", help="Fast regex-only analysis (no LLM, no indexing)"
    ),
    symlinks: bool = typer.Option(
        False, "--symlinks", "-l", help="Create CLAUDE.md and other tool aliases"
    ),
    config_only: bool = typer.Option(
        False, "--config", "-c", help="Only setup global config, skip project analysis"
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", "-I", help="Use interactive wizard for scoping"
    ),
):
    """Initialize project context and configuration."""
    if ctx.invoked_subcommand is None:
        paths = get_project_paths()

        # Step 1: Global config setup
        config_dir = paths.global_victor_dir
        config_dir.mkdir(parents=True, exist_ok=True)

        profiles_file = config_dir / "profiles.yaml"
        if not profiles_file.exists():
            console.print(f"[dim]Creating default configuration at {profiles_file}[/]")

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
            console.print(f"[green]✓[/] Global config created at {config_dir}")
        else:
            console.print(f"[dim]Global config exists at {config_dir}[/]")

        # Step 1.5: Initialize databases
        console.print("[dim]Initializing databases...[/]")
        try:
            # Initialize global database (user-level: RL, sessions, signatures)
            global_db = get_database()
            console.print(f"  [green]✓[/] Global database: {global_db.db_path}")

            # Initialize project database (repo-level: graph, conversations, entities)
            project_db = get_project_database()
            console.print(f"  [green]✓[/] Project database: {project_db.db_path}")
        except Exception as e:
            console.print(f"  [yellow]![/] Database initialization warning: {e}")

        if config_only:
            console.print(f"\nEdit {profiles_file} to customize profiles")
            return

        # Step 2: Project codebase analysis
        console.print("")
        target_path = paths.project_context_file
        paths.project_victor_dir.mkdir(parents=True, exist_ok=True)

        existing_content = None
        if target_path.exists():
            existing_content = target_path.read_text(encoding="utf-8")

            if not force and not update and interactive:
                if not Confirm.ask(
                    f"[yellow]{VICTOR_CONTEXT_FILE} already exists. Do you want to overwrite it with new analysis options?[/]"
                ):  # default=False
                    console.print("[dim]Aborted.[/dim]")
                    return
            elif not force and not update:
                console.print(f"[yellow]{VICTOR_CONTEXT_FILE} already exists at {target_path}[/]")
                console.print("")
                console.print("[bold]Options:[/]")
                console.print(
                    "  [cyan]victor init --update[/]   Merge new analysis (preserves your edits)"
                )
                console.print("  [cyan]victor init --force[/]    Overwrite completely")
                console.print(
                    "  [cyan]victor init --learn[/]    Enhance with conversation history insights"
                )
                console.print("  [cyan]victor init --index[/]    Multi-language symbol indexing")
                console.print("  [cyan]victor init --deep[/]     LLM-powered deep analysis")
                return

        # Handle --quick flag (overrides everything)
        if quick:
            deep = False
            learn = False
            index = False

        include_dirs: List[str] = []
        exclude_dirs: List[str] = []

        if interactive and not quick:
            console.print("\n[bold]Interactive Project Scoping[/]")

            # Get current working directory (cross-platform)
            cwd = Path.cwd()
            all_dirs = [
                d.name for d in cwd.iterdir() if d.is_dir() and not d.name.startswith(".")
            ]  # noqa
            common_src_dirs = ["src", "app", "lib", "victor", "server", "client"]
            suggested_src = [d for d in common_src_dirs if d in all_dirs]

            if not suggested_src and all_dirs:
                project_dir = cwd.name
                if project_dir in all_dirs:
                    suggested_src.append(project_dir)
                # If no common src dirs found, default to "." (entire project)
                # The user can narrow down if needed

            # Default to "." if no suggestions - scan entire project (cross-platform)
            # "." means current working directory on all platforms
            # Default to scanning the whole project (.) even if we detect likely src dirs;
            # users can narrow scope by entering a comma-separated list.
            default_include = "."

            include_str = Prompt.ask(
                f"[cyan]Enter comma-separated source directories to include[/]",
                default=default_include,
            )
            # Normalize "." to current working directory path for downstream processing
            include_dirs = []
            for d in include_str.split(","):
                d = d.strip()
                if d == ".":
                    # Use absolute path for current directory (cross-platform)
                    include_dirs.append(str(cwd))
                else:
                    include_dirs.append(d)

            default_exclude = [
                "__pycache__",
                ".git",
                ".pytest_cache",
                "venv",
                "env",
                ".venv",
                "node_modules",
                ".tox",
                "build",
                "dist",
                "egg-info",
                "htmlcov",
                "htmlcov_lang",
                ".mypy_cache",
                ".ruff_cache",
                # IDE/editor artifacts
                ".vscode-test",
                ".idea",
                # Coverage
                "coverage",
                # Third party / vendor
                "vendor",
                "third_party",
            ]

            exclude_str = Prompt.ask(
                f"[cyan]Enter comma-separated directories to exclude[/]",
                default=", ".join(default_exclude),
            )
            exclude_dirs = [d.strip() for d in exclude_str.split(",")]

        # Determine analysis mode and print status
        if deep and learn:
            console.print("[dim]Comprehensive analysis: Index → Learn → LLM (default)...[/]")
        elif deep:
            console.print("[dim]Analysis: Index → LLM (no conversation insights)...[/]")
        elif learn:
            console.print("[dim]Analysis: Index → Learn (no LLM)...[/]")
        elif index:
            console.print("[dim]Analysis: Index only (no LLM, no conversation)...[/]")
        else:
            console.print("[dim]Quick regex analysis (no indexing)...[/]")

        try:

            def on_progress(stage: str, msg: str):
                console.print(f"[dim]  {msg}[/]")

            if deep or learn:
                from victor.context.codebase_analyzer import generate_enhanced_init_md

                new_content = asyncio.run(
                    generate_enhanced_init_md(
                        use_llm=deep,
                        include_conversations=learn,
                        on_progress=on_progress,
                        force=force,
                        include_dirs=include_dirs or None,
                        exclude_dirs=exclude_dirs or None,
                    )
                )
            elif index:
                from victor.context.codebase_analyzer import generate_victor_md_from_index

                console.print("[dim]  Building symbol index...[/]")
                new_content = asyncio.run(
                    generate_victor_md_from_index(
                        force=force,
                        include_dirs=include_dirs or None,
                        exclude_dirs=exclude_dirs or None,
                    )
                )
            else:
                from victor.context.codebase_analyzer import generate_smart_victor_md

                new_content = generate_smart_victor_md(
                    include_dirs=include_dirs or None, exclude_dirs=exclude_dirs or None
                )

            if update and existing_content:
                from victor.ui.slash.commands.codebase import InitCommand

                # Use InitCommand's public merge function
                init_cmd = InitCommand()
                content = init_cmd.merge_init_content(existing_content, new_content)
                console.print("[dim]  Merged with existing content[/]")
            else:
                content = new_content

            target_path.write_text(content, encoding="utf-8")
            console.print(f"[green]✓[/] Created {target_path}")

            component_count = content.count("| `")
            pattern_count = content.count(". **") + content.count("Pattern:")
            console.print(f"[dim]  - Detected {component_count} key components[/]")
            console.print(f"[dim]  - Found {pattern_count} architecture patterns[/]")

            if symlinks:
                from victor.context.codebase_analyzer import (
                    CONTEXT_FILE_ALIASES,
                    create_context_symlinks,
                )

                console.print("\n[dim]Creating symlinks for other AI tools...[/]")
                results = create_context_symlinks()

                for alias, status in results.items():
                    tool_name = CONTEXT_FILE_ALIASES.get(alias, "Unknown")
                    if status == "created":
                        console.print(
                            f"  [green]✓[/] {alias} -> {VICTOR_DIR_NAME}/{VICTOR_CONTEXT_FILE} ({tool_name})"
                        )
                    elif status == "exists":
                        console.print(f"  [dim]○[/] {alias} (already linked)")
                    elif status == "exists_file":
                        console.print(f"  [yellow]![/] {alias} (file exists, not a symlink)")

            console.print(f"\n[dim]Review and customize {target_path} as needed.[/]")

        except Exception as e:
            console.print(f"[red]Failed to create {VICTOR_DIR_NAME}/{VICTOR_CONTEXT_FILE}:[/] {e}")
            import traceback

            traceback.print_exc()
