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

"""Workflow CLI commands for Victor.

Provides commands for validating, rendering, listing, and executing
YAML-defined workflows.

Commands:
    validate  - Validate YAML workflow files
    render    - Render workflow DAG as diagram (ascii, mermaid, d2, dot, svg, png)
    list      - List available workflows in a directory
    run       - Execute a workflow

Example:
    victor workflow validate ./workflows/analysis.yaml
    victor workflow render ./workflows/analysis.yaml --format svg -o diagram.svg
    victor workflow list ./workflows/
    victor workflow run ./workflows/analysis.yaml --context '{"symbol": "AAPL"}'
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

workflow_app = typer.Typer(
    name="workflow",
    help="Manage and execute YAML workflows.",
)

console = Console()


def _load_workflow_file(workflow_path: Path) -> dict:
    """Load and parse a workflow YAML file.

    Args:
        workflow_path: Path to the YAML file

    Returns:
        Dictionary of workflow name -> WorkflowDefinition

    Raises:
        typer.Exit: If file not found or invalid
    """
    from victor.workflows import load_workflow_from_file, YAMLWorkflowError
    from victor.workflows.definition import WorkflowDefinition

    if not workflow_path.exists():
        console.print(f"[bold red]Error:[/] Workflow file not found: {workflow_path}")
        raise typer.Exit(1)

    if workflow_path.suffix not in {".yaml", ".yml"}:
        console.print(f"[bold red]Error:[/] File must be .yaml or .yml: {workflow_path}")
        raise typer.Exit(1)

    try:
        loaded = load_workflow_from_file(str(workflow_path))

        workflows: dict[str, WorkflowDefinition] = {}
        if isinstance(loaded, dict):
            workflows = loaded
        else:
            workflows = {loaded.name: loaded}

        if not workflows:
            console.print("[bold red]Error:[/] No workflows found in file")
            raise typer.Exit(1)

        return workflows

    except YAMLWorkflowError as e:
        console.print(f"[bold red]Error:[/] Failed to parse workflow: {e}")
        raise typer.Exit(1)


def _display_workflow_info(workflow, wf_name: str) -> None:
    """Display workflow information in a table."""
    table = Table(show_header=False, box=None)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Description", workflow.description or "(none)")
    table.add_row("Nodes", str(len(workflow.nodes)))
    table.add_row("Start Node", workflow.start_node)

    # Count node types
    node_types: dict[str, int] = {}
    for node in workflow.nodes.values():
        node_type = type(node).__name__
        node_types[node_type] = node_types.get(node_type, 0) + 1

    types_str = ", ".join(f"{k}: {v}" for k, v in sorted(node_types.items()))
    table.add_row("Node Types", types_str)

    console.print(table)


@workflow_app.command("validate")
def validate_workflow(
    workflow_path: str = typer.Argument(
        ...,
        help="Path to YAML workflow file to validate",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed node information",
    ),
) -> None:
    """Validate a YAML workflow file.

    Checks that the workflow file is valid YAML, contains valid node
    definitions, and can be compiled to a state graph.

    Example:
        victor workflow validate ./workflows/analysis.yaml
        victor workflow validate ./workflows/analysis.yaml -v
    """
    from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler

    path = Path(workflow_path)
    console.print(f"\n[bold blue]Validating:[/] {path.name}")
    console.print("[dim]" + "─" * 50 + "[/]")

    workflows = _load_workflow_file(path)

    compiler = YAMLToStateGraphCompiler()
    all_valid = True

    for wf_name, workflow in workflows.items():
        console.print(f"\n[bold cyan]Workflow:[/] {wf_name}")
        _display_workflow_info(workflow, wf_name)

        if verbose:
            console.print("\n[dim]Nodes:[/]")
            for node_id, node in workflow.nodes.items():
                node_type = type(node).__name__
                next_nodes = getattr(node, "next_nodes", []) or []
                next_str = " -> " + ", ".join(next_nodes) if next_nodes else ""
                console.print(f"  [dim]{node_id}[/] ({node_type}){next_str}")

        try:
            _compiled = compiler.compile(workflow)
            console.print("[bold green]✓[/] Validation passed")
        except Exception as e:
            console.print(f"[bold red]✗[/] Validation failed: {e}")
            all_valid = False

    console.print()
    if all_valid:
        console.print("[bold green]All workflows validated successfully.[/]")
    else:
        console.print("[bold red]Some workflows failed validation.[/]")
        raise typer.Exit(1)


@workflow_app.command("render")
def render_workflow(
    workflow_path: str = typer.Argument(
        ...,
        help="Path to YAML workflow file to render",
    ),
    format: str = typer.Option(
        "ascii",
        "--format",
        "-f",
        help="Output format: ascii, mermaid, d2, dot, plantuml, svg, png",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (required for svg/png, optional for others)",
    ),
    workflow_name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Workflow name to render (if file contains multiple workflows)",
    ),
) -> None:
    """Render a workflow DAG as a diagram.

    Supports multiple output formats:
    - ascii: Terminal-friendly ASCII art
    - mermaid: Mermaid markdown (for docs)
    - d2: D2 diagram language
    - dot: Graphviz DOT format
    - plantuml: PlantUML format
    - svg: SVG image (requires d2, graphviz, or matplotlib)
    - png: PNG image (requires d2 or graphviz)

    Example:
        victor workflow render ./workflows/analysis.yaml
        victor workflow render ./workflows/analysis.yaml -f mermaid
        victor workflow render ./workflows/analysis.yaml -f svg -o diagram.svg
    """
    from victor.workflows.visualization import (
        WorkflowVisualizer,
        OutputFormat as VizFormat,
        RenderBackend,
    )

    path = Path(workflow_path)
    workflows = _load_workflow_file(path)

    # Select workflow
    if workflow_name:
        if workflow_name not in workflows:
            console.print(f"[bold red]Error:[/] Workflow '{workflow_name}' not found")
            console.print(f"Available: {', '.join(workflows.keys())}")
            raise typer.Exit(1)
        workflow = workflows[workflow_name]
    else:
        workflow = next(iter(workflows.values()))
        if len(workflows) > 1:
            console.print(f"[dim]Multiple workflows found, rendering '{workflow.name}'[/]")
            console.print(f"[dim]Use --name to specify: {', '.join(workflows.keys())}[/]")

    # Map format string to enum
    format_map = {
        "ascii": VizFormat.ASCII,
        "mermaid": VizFormat.MERMAID,
        "plantuml": VizFormat.PLANTUML,
        "dot": VizFormat.DOT,
        "d2": VizFormat.D2,
        "svg": VizFormat.SVG,
        "png": VizFormat.PNG,
    }

    fmt = format_map.get(format.lower())
    if not fmt:
        console.print(f"[bold red]Error:[/] Unknown format: {format}")
        console.print(f"Supported: {', '.join(format_map.keys())}")
        raise typer.Exit(1)

    # SVG/PNG require output path or generate default
    if fmt in {VizFormat.SVG, VizFormat.PNG} and not output:
        output = str(path.with_suffix(f".{format.lower()}"))
        console.print(f"[dim]Output: {output}[/]")

    viz = WorkflowVisualizer(workflow)

    try:
        result = viz.render(fmt, output, RenderBackend.AUTO)

        if output:
            console.print(f"[bold green]✓[/] Saved to {output}")
        else:
            console.print()
            console.print(result)

    except ImportError as e:
        console.print(f"[bold red]Error:[/] Rendering backend not available: {e}")
        console.print("[dim]For SVG/PNG, install: pip install matplotlib networkx[/]")
        console.print("[dim]Or install d2: brew install d2[/]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/] Rendering failed: {e}")
        raise typer.Exit(1)


@workflow_app.command("list")
def list_workflows(
    directory: str = typer.Argument(
        ".",
        help="Directory containing workflow YAML files",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Search subdirectories recursively",
    ),
) -> None:
    """List available workflows in a directory.

    Scans for .yaml/.yml files and displays workflow information.

    Example:
        victor workflow list ./workflows/
        victor workflow list ./workflows/ -r
    """
    from victor.workflows import load_workflow_from_file, YAMLWorkflowError

    path = Path(directory)
    if not path.exists():
        console.print(f"[bold red]Error:[/] Directory not found: {directory}")
        raise typer.Exit(1)

    if not path.is_dir():
        console.print(f"[bold red]Error:[/] Not a directory: {directory}")
        raise typer.Exit(1)

    # Find YAML files
    pattern = "**/*.yaml" if recursive else "*.yaml"
    yaml_files = list(path.glob(pattern)) + list(path.glob(pattern.replace(".yaml", ".yml")))

    if not yaml_files:
        console.print(f"[dim]No workflow files found in {directory}[/]")
        return

    table = Table(title="Available Workflows")
    table.add_column("File", style="cyan")
    table.add_column("Workflow", style="green")
    table.add_column("Description")
    table.add_column("Nodes", justify="right")
    table.add_column("Types")

    total_workflows = 0

    for yaml_file in sorted(yaml_files):
        try:
            loaded = load_workflow_from_file(str(yaml_file))

            if isinstance(loaded, dict):
                workflows = loaded
            else:
                workflows = {loaded.name: loaded}

            for wf_name, workflow in workflows.items():
                node_types = {}
                for node in workflow.nodes.values():
                    node_type = type(node).__name__.replace("Node", "")
                    node_types[node_type] = node_types.get(node_type, 0) + 1

                types_str = ", ".join(f"{v}{k[0]}" for k, v in sorted(node_types.items()))

                table.add_row(
                    yaml_file.name,
                    wf_name,
                    (workflow.description or "")[:40],
                    str(len(workflow.nodes)),
                    types_str,
                )
                total_workflows += 1

        except YAMLWorkflowError:
            table.add_row(yaml_file.name, "[red]Error[/]", "Failed to parse", "-", "-")
        except Exception as e:
            table.add_row(yaml_file.name, "[red]Error[/]", str(e)[:40], "-", "-")

    console.print(table)
    console.print(f"\n[dim]Total: {total_workflows} workflow(s) in {len(yaml_files)} file(s)[/]")


@workflow_app.command("run")
def run_workflow(
    workflow_path: str = typer.Argument(
        ...,
        help="Path to YAML workflow file to execute",
    ),
    context: Optional[str] = typer.Option(
        None,
        "--context",
        "-c",
        help='Initial context as JSON string (e.g., \'{"symbol": "AAPL"}\')',
    ),
    context_file: Optional[str] = typer.Option(
        None,
        "--context-file",
        "-f",
        help="Path to JSON file with initial context",
    ),
    workflow_name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Workflow name to run (if file contains multiple workflows)",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Profile to use for agent nodes",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and show execution plan without running",
    ),
) -> None:
    """Execute a workflow.

    Runs the workflow with the provided initial context. Agent nodes
    will use the specified profile for LLM interactions.

    Example:
        victor workflow run ./workflows/analysis.yaml -c '{"symbol": "AAPL"}'
        victor workflow run ./workflows/analysis.yaml -f context.json
        victor workflow run ./workflows/analysis.yaml --dry-run
    """
    from victor.workflows import StateGraphExecutor, ExecutorConfig
    from victor.workflows.definition import AgentNode
    from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler
    from victor.config.settings import load_settings
    from victor.agent import AgentOrchestrator
    from victor.ui.commands.utils import graceful_shutdown

    path = Path(workflow_path)
    workflows = _load_workflow_file(path)

    # Select workflow
    if workflow_name:
        if workflow_name not in workflows:
            console.print(f"[bold red]Error:[/] Workflow '{workflow_name}' not found")
            raise typer.Exit(1)
        workflow = workflows[workflow_name]
    else:
        workflow = next(iter(workflows.values()))

    # Parse context
    initial_context = {}
    if context:
        try:
            initial_context = json.loads(context)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Error:[/] Invalid JSON in --context: {e}")
            raise typer.Exit(1)

    if context_file:
        context_path = Path(context_file)
        if not context_path.exists():
            console.print(f"[bold red]Error:[/] Context file not found: {context_file}")
            raise typer.Exit(1)
        try:
            with open(context_path) as f:
                file_context = json.load(f)
                initial_context.update(file_context)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Error:[/] Invalid JSON in context file: {e}")
            raise typer.Exit(1)

    console.print(f"\n[bold blue]Workflow:[/] {workflow.name}")
    console.print("[dim]" + "─" * 50 + "[/]")
    _display_workflow_info(workflow, workflow.name)

    if initial_context:
        console.print(f"\n[dim]Context: {json.dumps(initial_context, default=str)[:100]}...[/]")

    if dry_run:
        console.print("\n[bold yellow]Dry run mode[/] - showing execution plan:")

        # Show execution order
        compiler = YAMLToStateGraphCompiler()
        try:
            compiled = compiler.compile(workflow)
            console.print("[bold green]✓[/] Workflow can be compiled")
            console.print(f"[dim]Start node: {workflow.start_node}[/]")
        except Exception as e:
            console.print(f"[bold red]✗[/] Compilation failed: {e}")
            raise typer.Exit(1)
        return

    # Execute workflow
    async def _execute():
        settings = load_settings()

        # Create orchestrator if agent nodes exist
        orchestrator = None
        has_agent_nodes = any(isinstance(node, AgentNode) for node in workflow.nodes.values())

        if has_agent_nodes:
            orchestrator = await AgentOrchestrator.create(
                profile=profile,
                settings=settings,
            )

        executor = StateGraphExecutor(
            config=ExecutorConfig(
                max_parallel=4,
                timeout_seconds=3600,
            ),
            orchestrator=orchestrator,
        )

        console.print(f"\n[bold]Executing workflow '{workflow.name}'...[/]\n")

        try:
            result = await executor.execute(workflow, initial_context)

            console.print("[dim]" + "─" * 50 + "[/]")

            if result.success:
                console.print("[bold green]✓[/] Workflow completed successfully")
                console.print(f"  [dim]Duration: {result.duration_seconds:.2f}s[/]")
                console.print(f"  [dim]Nodes executed: {', '.join(result.nodes_executed)}[/]")

                # Show final state summary
                if result.final_state:
                    console.print("\n[bold]Final State:[/]")
                    # Filter out large/internal keys
                    display_state = {
                        k: v
                        for k, v in result.final_state.items()
                        if not k.startswith("_") and k not in {"messages", "history"}
                    }
                    console.print(json.dumps(display_state, indent=2, default=str)[:2000])
            else:
                console.print("[bold red]✗[/] Workflow failed")
                if result.error:
                    console.print(f"  [red]{result.error}[/]")
                raise typer.Exit(1)

        finally:
            if orchestrator:
                await graceful_shutdown(orchestrator)

    asyncio.run(_execute())


@workflow_app.command("backends")
def list_backends() -> None:
    """Show available rendering backends.

    Lists which rendering backends are installed and available
    for SVG/PNG output.

    Example:
        victor workflow backends
    """
    from victor.workflows.visualization import get_available_backends

    backends = get_available_backends()

    table = Table(title="Rendering Backends")
    table.add_column("Backend", style="cyan")
    table.add_column("Status")
    table.add_column("Install Command")

    install_hints = {
        "d2": "brew install d2",
        "graphviz": "brew install graphviz",
        "mermaid-cli": "npm install -g @mermaid-js/mermaid-cli",
        "plantuml": "brew install plantuml",
        "matplotlib": "pip install matplotlib networkx",
        "kroki": "(cloud API - always available)",
        "ascii": "(built-in - always available)",
    }

    for name, available in backends.items():
        status = "[green]✓ Available[/]" if available else "[red]✗ Not installed[/]"
        hint = install_hints.get(name, "")
        table.add_row(name, status, hint if not available else "[dim]installed[/]")

    console.print(table)
    console.print("\n[dim]For SVG/PNG rendering, at least one backend is required.[/]")
