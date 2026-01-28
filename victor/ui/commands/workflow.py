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
    validate  - Validate YAML workflow files (with escape hatch and handler checks)
    render    - Render workflow DAG as diagram (ascii, mermaid, d2, dot, svg, png)
    list      - List available workflows in a directory
    run       - Execute a workflow

Validation checks:
    - YAML syntax and structure
    - Node references (next_nodes, branches, parallel_nodes)
    - Escape hatch references (CONDITIONS/TRANSFORMS in condition/transform nodes)
    - Handler references (registered compute handlers)
    - HITL node configuration
    - Capability provider integration hints

Example:
    victor workflow validate ./workflows/analysis.yaml
    victor workflow validate ./workflows/analysis.yaml --check-handlers
    victor workflow render ./workflows/analysis.yaml --format svg -o diagram.svg
    victor workflow list ./workflows/
    victor workflow run ./workflows/analysis.yaml --context '{"symbol": "AAPL"}'
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import typer
from rich.console import Console
from rich.panel import Panel
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


def _detect_vertical_from_path(workflow_path: Path) -> Optional[str]:
    """Detect vertical from workflow file path.

    Looks for vertical names in the path structure, e.g.:
    - victor/coding/workflows/feature.yaml -> coding
    - victor/research/workflows/paper.yaml -> research
    """
    known_verticals = {"coding", "devops", "rag", "dataanalysis", "research", "benchmark"}
    parts = workflow_path.parts

    for part in parts:
        if part in known_verticals:
            return part
    return None


def _load_escape_hatches(vertical: Optional[str]) -> tuple[Set[str], Set[str]]:
    """Load escape hatch registries (CONDITIONS and TRANSFORMS) for a vertical.

    Args:
        vertical: Vertical name (coding, research, etc.)

    Returns:
        Tuple of (condition_names, transform_names)
    """
    conditions: Set[str] = set()
    transforms: Set[str] = set()

    if not vertical:
        return conditions, transforms

    try:
        # Import the escape_hatches module for the vertical
        module_name = f"victor.{vertical}.escape_hatches"
        import importlib

        module = importlib.import_module(module_name)

        if hasattr(module, "CONDITIONS"):
            conditions = set(module.CONDITIONS.keys())
        if hasattr(module, "TRANSFORMS"):
            transforms = set(module.TRANSFORMS.keys())

    except ImportError:
        # No escape hatches module for this vertical
        pass

    return conditions, transforms


def _load_registered_handlers(vertical: Optional[str]) -> Set[str]:
    """Load registered compute handler names.

    First ensures vertical handlers are registered, then returns all known handlers.
    """
    handlers: Set[str] = set()

    # Try to register vertical handlers first
    if vertical:
        try:
            module_name = f"victor.{vertical}.handlers"
            import importlib

            module = importlib.import_module(module_name)

            if hasattr(module, "register_handlers"):
                module.register_handlers()
            if hasattr(module, "HANDLERS"):
                handlers.update(module.HANDLERS.keys())
        except ImportError:
            pass

    # Get all registered handlers from executor
    try:
        from victor.workflows.executor import list_compute_handlers

        handlers.update(list_compute_handlers())
    except ImportError:
        pass

    return handlers


def _validate_escape_hatches_and_handlers(
    workflow,
    vertical: Optional[str],
    check_handlers: bool = False,
) -> tuple[List[str], List[str]]:
    """Validate escape hatch and handler references in workflow.

    Args:
        workflow: WorkflowDefinition to validate
        vertical: Vertical name for loading escape hatches
        check_handlers: Whether to validate handler references

    Returns:
        Tuple of (errors, warnings)
    """
    from victor.workflows.definition import (
        ConditionNode,
        TransformNode,
        ComputeNode,
        ParallelNode,
    )
    from victor.workflows.hitl import HITLNode

    errors: List[str] = []
    warnings: List[str] = []

    # Load registries
    conditions, transforms = _load_escape_hatches(vertical)
    handlers = _load_registered_handlers(vertical) if check_handlers else set()

    # Track what's used
    used_conditions: Set[str] = set()
    used_transforms: Set[str] = set()
    used_handlers: Set[str] = set()

    # Names that indicate simple/inline conditions (not escape hatch references)
    simple_condition_names = {
        "<lambda>",
        "truthy_condition",
        "condition",
        "in_condition",
        "literal_transform",
        "ref_transform",
    }

    for node_id, node in workflow.nodes.items():
        # Check condition nodes for escape hatch references
        if isinstance(node, ConditionNode):
            # Try to determine if this is a simple expression or escape hatch reference
            # Simple expressions get compiled to functions with generic names
            condition_name = getattr(node.condition, "__name__", None)
            if condition_name and condition_name not in simple_condition_names:
                used_conditions.add(condition_name)
                if conditions and condition_name not in conditions:
                    warnings.append(
                        f"Condition node '{node_id}' uses escape hatch '{condition_name}' "
                        f"not found in {vertical}.escape_hatches.CONDITIONS"
                    )

        # Check transform nodes for escape hatch references
        if isinstance(node, TransformNode):
            transform_name = getattr(node.transform, "__name__", None)
            if transform_name and transform_name not in {
                "<lambda>",
                "literal_transform",
                "ref_transform",
            }:
                used_transforms.add(transform_name)
                if transforms and transform_name not in transforms:
                    warnings.append(
                        f"Transform node '{node_id}' uses escape hatch '{transform_name}' "
                        f"not found in {vertical}.escape_hatches.TRANSFORMS"
                    )

        # Check compute nodes for handler references
        if isinstance(node, ComputeNode):
            if node.handler:
                used_handlers.add(node.handler)
                if check_handlers and node.handler not in handlers:
                    errors.append(
                        f"Compute node '{node_id}' references unregistered handler '{node.handler}'"
                    )

        # Check parallel nodes
        if isinstance(node, ParallelNode):
            if not node.parallel_nodes:
                warnings.append(f"Parallel node '{node_id}' has no parallel_nodes defined")

        # Check HITL nodes
        if isinstance(node, HITLNode):
            if not node.prompt:
                warnings.append(f"HITL node '{node_id}' has no prompt defined")
            if node.hitl_type.value == "choice" and not node.choices:
                errors.append(f"HITL choice node '{node_id}' has no choices defined")

    return errors, warnings


def _display_validation_details(
    workflow,
    verbose: bool = False,
    check_handlers: bool = False,
    vertical: Optional[str] = None,
) -> tuple[bool, List[str], List[str]]:
    """Display detailed validation results for a workflow.

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    from victor.workflows.definition import (
        AgentNode,
        ConditionNode,
        TransformNode,
        ComputeNode,
        ParallelNode,
    )

    errors: List[str] = []
    warnings: List[str] = []

    # Basic validation
    basic_errors = workflow.validate()
    errors.extend(basic_errors)

    # Extended validation for escape hatches and handlers
    ext_errors, ext_warnings = _validate_escape_hatches_and_handlers(
        workflow, vertical, check_handlers
    )
    errors.extend(ext_errors)
    warnings.extend(ext_warnings)

    if verbose:
        console.print("\n[dim]Nodes:[/]")
        for node_id, node in workflow.nodes.items():
            node_type = type(node).__name__
            next_nodes = getattr(node, "next_nodes", []) or []
            next_str = " -> " + ", ".join(next_nodes) if next_nodes else ""

            # Add additional info based on node type
            info_parts = []
            if isinstance(node, AgentNode):
                info_parts.append(f"role={node.role}")
            elif isinstance(node, ComputeNode) and node.handler:
                info_parts.append(f"handler={node.handler}")
            elif isinstance(node, ConditionNode):
                branches = list(node.branches.keys())[:3]
                info_parts.append(f"branches={branches}")
            elif isinstance(node, ParallelNode):
                info_parts.append(f"parallel={node.parallel_nodes}")

            info_str = f" [{', '.join(info_parts)}]" if info_parts else ""
            console.print(f"  [dim]{node_id}[/] ({node_type}){info_str}{next_str}")

    return len(errors) == 0, errors, warnings


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
    check_handlers: bool = typer.Option(
        False,
        "--check-handlers",
        help="Validate that referenced handlers are registered",
    ),
    check_escape_hatches: bool = typer.Option(
        True,
        "--check-escape-hatches/--no-check-escape-hatches",
        help="Validate escape hatch references against vertical registries",
    ),
    vertical: Optional[str] = typer.Option(
        None,
        "--vertical",
        help="Vertical name for escape hatch validation (auto-detected from path if not specified)",
    ),
) -> None:
    """Validate a YAML workflow file.

    Checks that the workflow file is valid YAML, contains valid node
    definitions, and can be compiled to a state graph.

    Extended validation includes:
    - Escape hatch references (CONDITIONS/TRANSFORMS)
    - Handler references (when --check-handlers is specified)
    - HITL node configuration
    - Parallel node structure

    Example:
        victor workflow validate ./workflows/analysis.yaml
        victor workflow validate ./workflows/analysis.yaml -v
        victor workflow validate ./workflows/analysis.yaml --check-handlers
        victor workflow validate ./workflows/analysis.yaml --vertical coding
    """
    from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler

    path = Path(workflow_path)
    console.print(f"\n[bold blue]Validating:[/] {path.name}")
    console.print("[dim]" + "─" * 50 + "[/]")

    # Auto-detect vertical if not specified
    detected_vertical = vertical or _detect_vertical_from_path(path)
    if detected_vertical:
        console.print(f"[dim]Vertical: {detected_vertical}[/]")

    workflows = _load_workflow_file(path)

    compiler = YAMLToStateGraphCompiler()
    all_valid = True
    total_warnings = 0

    for wf_name, workflow in workflows.items():
        console.print(f"\n[bold cyan]Workflow:[/] {wf_name}")
        _display_workflow_info(workflow, wf_name)

        # Run extended validation
        is_valid, errors, warnings = _display_validation_details(
            workflow,
            verbose=verbose,
            check_handlers=check_handlers,
            vertical=detected_vertical if check_escape_hatches else None,
        )

        # Display warnings
        for warning in warnings:
            console.print(f"[yellow]  Warning:[/] {warning}")
            total_warnings += 1

        # Display errors
        for error in errors:
            console.print(f"[red]  Error:[/] {error}")

        # Try to compile to state graph
        try:
            _compiled = compiler.compile(workflow)
            if is_valid:
                console.print("[bold green]✓[/] Validation passed")
            else:
                console.print("[bold yellow]✓[/] Compiled with errors")
                all_valid = False
        except Exception as e:
            console.print(f"[bold red]✗[/] Compilation failed: {e}")
            all_valid = False

    console.print()

    # Show escape hatch and handler info if verbose
    if verbose and detected_vertical:
        conditions, transforms = _load_escape_hatches(detected_vertical)
        if conditions or transforms:
            console.print(f"\n[dim]Available escape hatches for '{detected_vertical}':[/]")
            if conditions:
                console.print(f"  CONDITIONS: {', '.join(sorted(conditions))}")
            if transforms:
                console.print(f"  TRANSFORMS: {', '.join(sorted(transforms))}")

        if check_handlers:
            handlers = _load_registered_handlers(detected_vertical)
            if handlers:
                console.print(f"  HANDLERS: {', '.join(sorted(handlers))}")
            console.print()

    if all_valid:
        msg = "[bold green]All workflows validated successfully.[/]"
        if total_warnings > 0:
            msg += f" [yellow]({total_warnings} warning(s))[/]"
        console.print(msg)
    else:
        console.print("[bold red]Some workflows failed validation.[/]")
        raise typer.Exit(1)


@workflow_app.command("lint")
def lint_workflow(
    path: str = typer.Argument(
        ...,
        help="Path to workflow file or directory to lint",
    ),
    format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: text, json, markdown",
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save report to file",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Enable strict validation (all rules)",
    ),
    disable_rules: Optional[List[str]] = typer.Option(
        None,
        "--disable-rule",
        help="Disable specific rules by ID",
    ),
    enable_rules: Optional[List[str]] = typer.Option(
        None,
        "--enable-rule",
        help="Enable specific rules by ID",
    ),
    include_suggestions: bool = typer.Option(
        True,
        "--include-suggestions/--no-suggestions",
        help="Include suggestions in report",
    ),
    include_context: bool = typer.Option(
        False,
        "--include-context",
        help="Include context data in report",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Recursively lint directory",
    ),
) -> None:
    """Lint workflow files for best practices and issues.

    Performs comprehensive validation including:
    - YAML syntax and structure
    - Node schema validation (required fields, types)
    - Connection reference checking
    - Circular dependency detection
    - Team node configuration
    - Best practices (goal quality, tool budgets)
    - Complexity analysis

    Rules can be enabled/disabled individually. Use --strict to enable all rules.

    Examples:
        victor workflow lint workflow.yaml
        victor workflow lint workflows/ --recursive
        victor workflow lint workflow.yaml --format json -o report.json
        victor workflow lint workflow.yaml --strict --disable-rule complexity_analysis
        victor workflow lint . --format markdown -o lint_report.md
    """
    from victor.workflows.linter import WorkflowLinter

    path_obj = Path(path)

    if not path_obj.exists():
        console.print(f"[bold red]Error:[/] Path not found: {path}")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]Linting:[/] {path_obj.name}")
    console.print("[dim]" + "─" * 50 + "[/]")

    # Create linter
    linter = WorkflowLinter()

    # Apply strict mode
    if strict:
        from victor.workflows.validation_rules import Severity

        console.print("[dim]Strict mode enabled[/]")
        for rule in linter.get_rules():
            if rule.severity in {Severity.INFO, Severity.SUGGESTION}:
                rule.enabled = True

    # Disable specific rules
    if disable_rules:
        for rule_id in disable_rules:
            linter.disable_rule(rule_id)
            console.print(f"[dim]Disabled rule: {rule_id}[/]")

    # Enable specific rules
    if enable_rules:
        for rule_id in enable_rules:
            linter.enable_rule(rule_id)
            console.print(f"[dim]Enabled rule: {rule_id}[/]")

    console.print()

    # Run linter
    if path_obj.is_file():
        result = linter.lint_file(path_obj)
    elif path_obj.is_dir():
        result = linter.lint_directory(path_obj, recursive=recursive)
    else:
        console.print(f"[bold red]Error:[/] Not a file or directory: {path}")
        raise typer.Exit(1)

    # Display results
    report = result.generate_report(
        format=format,
        include_suggestions=include_suggestions,
        include_context=include_context,
    )

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.write_text(report)
        console.print(f"[bold green]✓[/] Report saved to {output_file}")
    else:
        console.print()

        # Colorize text output
        if format == "text":
            # Add color indicators
            for line in report.split("\n"):
                if line.startswith("✗"):
                    console.print(f"[bold red]{line}[/]")
                elif line.startswith("⚠"):
                    console.print(f"[bold yellow]{line}[/]")
                elif line.startswith("ℹ"):
                    console.print(f"[bold cyan]{line}[/]")
                elif line.startswith("💡"):
                    console.print(f"[bold dim]{line}[/]")
                elif line.startswith("✓"):
                    console.print(f"[bold green]{line}[/]")
                else:
                    console.print(line)
        else:
            console.print(report)

    # Exit with error if validation failed
    if result.has_errors:
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
    summary: bool = typer.Option(
        False,
        "--summary",
        "-s",
        help="Show workflow summary with node type counts before rendering",
    ),
) -> None:
    """Render a workflow DAG as a diagram.

    Supports multiple output formats:
    - ascii: Terminal-friendly ASCII art (shows all node types with distinct icons)
    - mermaid: Mermaid markdown (for docs, supports HITL/parallel node shapes)
    - d2: D2 diagram language (modern styling with capability hints)
    - dot: Graphviz DOT format
    - plantuml: PlantUML format (with fork/join for parallel nodes)
    - svg: SVG image (requires d2, graphviz, or matplotlib)
    - png: PNG image (requires d2 or graphviz)

    Node type visualization:
    - AgentNode: Stadium shape, blue (@)
    - ComputeNode: Parallelogram, green (#)
    - ConditionNode: Diamond, orange (?)
    - ParallelNode: Double-border subroutine, purple (=)
    - TransformNode: Rectangle, gray (>)
    - HITLNode: Circle, red (!)

    Example:
        victor workflow render ./workflows/analysis.yaml
        victor workflow render ./workflows/analysis.yaml -f mermaid
        victor workflow render ./workflows/analysis.yaml -f svg -o diagram.svg
        victor workflow render ./workflows/analysis.yaml --summary
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

    # Show summary if requested
    if summary:
        from victor.workflows.definition import (
            AgentNode,
            ComputeNode,
            ConditionNode,
            ParallelNode,
            TransformNode,
        )

        try:
            from victor.workflows.hitl import HITLNode

            has_hitl = True
        except ImportError:
            # Create a dummy HITLNode type for type checking
            class HITLNode:  # type: ignore
                pass

            has_hitl = False

        console.print(f"\n[bold cyan]Workflow:[/] {workflow.name}")
        _display_workflow_info(workflow, workflow.name)

        # Show node breakdown with icons
        console.print("\n[dim]Node breakdown:[/]")
        for node_id, node in workflow.nodes.items():
            node_type = type(node).__name__
            if node_type == "AgentNode":
                icon = "@"
                extra = f"role={node.role}"
            elif node_type == "ComputeNode":
                icon = "#"
                extra = f"handler={node.handler}" if node.handler else f"tools={len(node.tools)}"
            elif node_type == "ConditionNode":
                icon = "?"
                extra = f"branches={list(node.branches.keys())}"
            elif node_type == "ParallelNode":
                icon = "="
                extra = f"parallel={node.parallel_nodes}"
            elif node_type == "TransformNode":
                icon = ">"
                extra = ""
            elif has_hitl and isinstance(node, HITLNode):
                icon = "!"
                extra = f"type={node.hitl_type.value}"
            else:
                icon = "*"
                extra = ""

            extra_str = f" [{extra}]" if extra else ""
            console.print(f"  [{icon}] {node_id}{extra_str}")
        console.print()

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
                node_types: dict[str, int] = {}
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
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
) -> None:
    """Execute a workflow.

    Runs the workflow with the provided initial context. Agent nodes
    will use the specified profile for LLM interactions.

    Example:
        victor workflow run ./workflows/analysis.yaml -c '{"symbol": "AAPL"}'
        victor workflow run ./workflows/analysis.yaml -f context.json
        victor workflow run ./workflows/analysis.yaml --dry-run
        victor workflow run ./workflows/analysis.yaml --log-level DEBUG
    """
    from victor.workflows import StateGraphExecutor, ExecutorConfig
    from victor.workflows.definition import AgentNode
    from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler
    from victor.config.settings import load_settings
    from victor.agent import AgentOrchestrator
    from victor.ui.commands.utils import graceful_shutdown

    # Set log level if specified
    if log_level:
        import logging

        level = getattr(logging, log_level.upper(), None)
        if not isinstance(level, int):
            console.print(f"[bold red]Error:[/] Invalid log level: {log_level}")
            raise typer.Exit(1)

        # Configure logging
        logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

        # Filter out noisy HTTP third-party logs at DEBUG level
        # Only show WARNING and above for HTTP libraries
        if level <= logging.DEBUG:
            for http_logger in ["httpcore", "httpx", "urllib3"]:
                logging.getLogger(http_logger).setLevel(logging.WARNING)

        console.print(f"[dim]Log level set to: {log_level}[/]")

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
            compiler.compile(workflow)  # Validate compilation
            console.print("[bold green]✓[/] Workflow can be compiled")
            console.print(f"[dim]Start node: {workflow.start_node}[/]")
        except Exception as e:
            console.print(f"[bold red]✗[/] Compilation failed: {e}")
            raise typer.Exit(1)
        return

    # Execute workflow
    async def _execute():
        from victor.framework.shim import FrameworkShim
        import logging

        logger = logging.getLogger(__name__)
        settings = load_settings()

        # Create orchestrators if agent nodes exist
        orchestrators = {}
        has_agent_nodes = any(isinstance(node, AgentNode) for node in workflow.nodes.values())

        if has_agent_nodes:
            # Collect all unique profiles from agent nodes
            node_profiles = set()
            profile_to_nodes = {}  # Map profile to list of nodes using it

            for node in workflow.nodes.values():
                if isinstance(node, AgentNode) and hasattr(node, "profile") and node.profile:
                    node_profiles.add(node.profile)
                    if node.profile not in profile_to_nodes:
                        profile_to_nodes[node.profile] = []
                    profile_to_nodes[node.profile].append(node.id)
                elif isinstance(node, AgentNode):
                    # Nodes without explicit profile use default
                    if "default" not in profile_to_nodes:
                        profile_to_nodes["default"] = []
                    profile_to_nodes["default"].append(node.id)

            # Always include the default profile
            all_profiles = node_profiles | {profile}

            console.print(
                f"[dim]Creating orchestrators for profiles: {', '.join(sorted(all_profiles))}[/]"
            )

            # Debug: Show profile to node mapping
            for prof, nodes in sorted(profile_to_nodes.items()):
                logger.debug(f"Profile '{prof}' assigned to nodes: {', '.join(nodes)}")

            # Create an orchestrator for each unique profile
            for profile_name in sorted(all_profiles):
                try:
                    logger.debug(f"Creating orchestrator for profile: {profile_name}")
                    shim = FrameworkShim(
                        settings,
                        profile_name=profile_name,
                        vertical=None,  # Workflows may use different verticals per node
                    )
                    orchestrator = await shim.create_orchestrator()
                    orchestrators[profile_name] = orchestrator
                    logger.debug(f"Successfully created orchestrator for profile: {profile_name}")
                    console.print(f"  [dim]✓ Profile '{profile_name}' initialized[/]")
                except Exception as e:
                    logger.error(
                        f"Failed to create orchestrator for profile '{profile_name}': {e}",
                        exc_info=True,
                    )
                    console.print(f"  [red]✗ Profile '{profile_name}' failed: {e}[/]")
                    if profile_name == profile:
                        # If default profile fails, we can't continue
                        raise

        executor = StateGraphExecutor(
            config=ExecutorConfig(
                enable_checkpointing=True,
                max_iterations=workflow.max_iterations or 15,
                timeout=workflow.max_execution_timeout_seconds or 3600.0,
                default_profile=profile,
            ),
            orchestrators=orchestrators,
        )

        # Set project root from workflow metadata if specified
        project_root_override = workflow.metadata.get("project_root")
        if project_root_override:
            from pathlib import Path
            from victor.config.settings import set_project_root

            project_path = Path(project_root_override).expanduser().resolve()
            set_project_root(project_path)
            console.print(f"[dim]📁 Project root set to: {project_path}[/]\n")
            logger.debug(f"Project root set from workflow metadata: {project_path}")

        console.print(f"\n[bold]Executing workflow '{workflow.name}'...[/]\n")
        logger.debug(f"\n{'='*80}")
        logger.debug(f"🚀 WORKFLOW EXECUTION START: {workflow.name}")
        logger.debug(f"   Initial Context: {list(initial_context.keys())}")
        logger.debug(f"   Total Nodes: {len(workflow.nodes)}")
        logger.debug(f"   Node IDs: {', '.join(workflow.nodes.keys())}")
        logger.debug(f"{'='*80}\n")

        try:
            result = await executor.execute(workflow, initial_context)
            logger.debug(f"\n{'='*80}")
            logger.debug(f"✅ WORKFLOW EXECUTION COMPLETE: {workflow.name}")
            logger.debug(f"   Success: {result.success}")
            logger.debug(f"   Duration: {result.duration_seconds:.2f}s")
            logger.debug(f"   Nodes Executed: {len(result.nodes_executed)}")
            logger.debug(f"   Final State Keys: {list(result.state.keys())}")
            logger.debug(f"{'='*80}\n")

            console.print("[dim]" + "─" * 50 + "[/]")

            if result.success:
                console.print("[bold green]✓[/] Workflow completed successfully")
                console.print(f"  [dim]Duration: {result.duration_seconds:.2f}s[/]")
                console.print(f"  [dim]Nodes executed: {', '.join(result.nodes_executed)}[/]")

                # Show final state summary
                if result.state:
                    console.print("\n[bold]Final State:[/]")
                    # Filter out large/internal keys
                    display_state = {
                        k: v
                        for k, v in result.state.items()
                        if not k.startswith("_") and k not in {"messages", "history"}
                    }
                    console.print(json.dumps(display_state, indent=2, default=str)[:2000])
            else:
                console.print("[bold red]✗[/] Workflow failed")
                if result.error:
                    console.print(f"  [red]{result.error}[/]")
                raise typer.Exit(1)

        finally:
            # Shutdown all orchestrators
            if orchestrators:
                for orch_name, orch in orchestrators.items():
                    try:
                        await graceful_shutdown(orch)
                        console.print(f"[dim]✓ Shut down orchestrator '{orch_name}'[/]")
                    except Exception as e:
                        console.print(f"[red]✗ Failed to shutdown '{orch_name}': {e}[/]")

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


@workflow_app.command("presets")
def list_presets(
    preset_type: str = typer.Option(
        "all",
        "--type",
        "-t",
        help="Preset type to list: all, agents, workflows",
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category (for workflows: code_review, research, etc.)",
    ),
) -> None:
    """List available preset agents and workflows.

    Shows predefined agent configurations and workflow templates that can be
    used as starting points for custom workflows.

    Example:
        victor workflow presets
        victor workflow presets --type agents
        victor workflow presets --type workflows --category code_review
    """
    from victor.workflows.presets import (
        list_agent_presets,
        list_workflow_presets,
        get_agent_preset,
        get_workflow_preset,
        WorkflowPreset,
        AgentPreset,
    )

    preset_type = preset_type.lower()

    if preset_type in ("all", "agents"):
        # List agent presets
        console.print("\n[bold cyan]Agent Presets:[/]\n")

        preset_names = list_agent_presets()
        for preset_name in sorted(preset_names):
            preset = get_agent_preset(preset_name)
            if preset:
                console.print(f"  [bold]{preset.role.title()}[/]")
                console.print(f"    • [cyan]{preset.name}[/]: {preset.description}")

            console.print()

    if preset_type in ("all", "workflows"):
        # List workflow presets
        console.print("[bold cyan]Workflow Presets:[/]\n")

        if category:
            # Filter by category
            preset_names = list_workflow_presets()
            category_presets: list[WorkflowPreset] = []
            for preset_name in preset_names:
                workflow_preset = get_workflow_preset(preset_name)
                if workflow_preset and getattr(workflow_preset, 'category', None) == category:
                    category_presets.append(workflow_preset)

            if not category_presets:
                console.print(f"[bold red]Error:[/] No presets found for category: {category}")
                # Get available categories
                all_presets = [get_workflow_preset(n) for n in preset_names]
                available_categories = set(getattr(p, 'category', None) for p in all_presets if p)
                if available_categories:
                    console.print(f"Available: {', '.join(sorted(available_categories))}")
                raise typer.Exit(1)

            for workflow_preset in sorted(category_presets, key=lambda x: x.name):
                console.print(f"  • [cyan]{workflow_preset.name}[/]: {workflow_preset.description}")
                complexity = getattr(workflow_preset, 'complexity', None)
                duration = getattr(workflow_preset, 'estimated_duration_minutes', None)
                if complexity is not None and duration is not None:
                    console.print(
                        f"    [dim]Complexity: {complexity}, "
                        f"~{duration}min[/]"
                    )
        else:
            # Show all by category
            preset_names = list_workflow_presets()
            by_category: dict[str, list[WorkflowPreset]] = {}
            for preset_name in preset_names:
                workflow_preset = get_workflow_preset(preset_name)
                if workflow_preset:
                    cat = getattr(workflow_preset, 'category', 'uncategorized')
                    if cat not in by_category:
                        by_category[cat] = []
                    by_category[cat].append(workflow_preset)

            for cat, category_presets in sorted(by_category.items()):
                console.print(f"  [bold]{cat.title()}[/]")
                for workflow_preset in sorted(category_presets, key=lambda x: x.name):
                    console.print(f"    • [cyan]{workflow_preset.name}[/]: {workflow_preset.description}")

                console.print()

    console.print("\n[dim]Usage: Use preset names in workflow YAML files or WorkflowBuilder API[/]")


@workflow_app.command("preset-info")
def show_preset_info(
    preset_name: str = typer.Argument(
        ...,
        help="Name of the preset to show details for",
    ),
    preset_type: str = typer.Option(
        "workflow",
        "--type",
        "-t",
        help="Preset type: agent or workflow",
    ),
) -> None:
    """Show detailed information about a preset.

    Displays full details including configuration, example usage, and parameters.

    Example:
        victor workflow preset-info code_reviewer --type agent
        victor workflow preset-info code_review --type workflow
    """
    from victor.workflows.presets import get_agent_preset, get_workflow_preset

    preset_type = preset_type.lower()

    console.print(f"\n[bold cyan]Preset:[/] {preset_name}\n")
    console.print("[dim]" + "─" * 50 + "[/]\n")

    if preset_type == "agent":
        preset = get_agent_preset(preset_name)
        if not preset:
            console.print(f"[bold red]Error:[/] Agent preset '{preset_name}' not found")
            console.print(
                f"[dim]Use 'victor workflow presets --type agents' to list available presets[/]"
            )
            raise typer.Exit(1)

        # Display agent preset details
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Name", preset.name)
        table.add_row("Role", preset.role)
        table.add_row("Description", preset.description)
        table.add_row("Tool Budget", str(preset.default_tool_budget))
        table.add_row("Goal Template", preset.goal_template)

        if preset.allowed_tools:
            table.add_row(
                "Allowed Tools",
                (
                    ", ".join(preset.allowed_tools[:10]) + "..."
                    if len(preset.allowed_tools) > 10
                    else ", ".join(preset.allowed_tools)
                ),
            )

        if preset.llm_config:
            table.add_row("LLM Config", ", ".join(f"{k}={v}" for k, v in preset.llm_config.items()))

        console.print(table)

        if preset.example_goals:
            console.print("\n[bold]Example Goals:[/]")
            for goal in preset.example_goals:
                console.print(f"  • {goal}")

        console.print("\n[bold]Usage Example:[/]")
        console.print(
            f"  [dim]# In WorkflowBuilder:[/]\n"
            f"  from victor.workflows.presets import get_agent_preset\n\n"
            f"  preset = get_agent_preset('{preset_name}')\n"
            f"  params = preset.to_workflow_builder_params('Custom goal')\n"
            f"  workflow = WorkflowBuilder('my_workflow')\n"
            f"              .add_agent('agent1', **params)\n"
            f"              .build()"
        )

    elif preset_type == "workflow":
        workflow_preset = get_workflow_preset(preset_name)
        if not workflow_preset:
            console.print(f"[bold red]Error:[/] Workflow preset '{preset_name}' not found")
            console.print(
                f"[dim]Use 'victor workflow presets --type workflows' to list available presets[/]"
            )
            raise typer.Exit(1)

        # Display workflow preset details
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Name", workflow_preset.name)
        table.add_row("Category", workflow_preset.category)
        table.add_row("Description", workflow_preset.description)
        table.add_row("Complexity", workflow_preset.complexity)
        table.add_row("Estimated Duration", f"{workflow_preset.estimated_duration_minutes} minutes")

        console.print(table)

        if workflow_preset.example_context:
            console.print("\n[bold]Example Context:[/]")
            import json

            console.print(json.dumps(workflow_preset.example_context, indent=2, default=str)[:500])

        console.print("\n[bold]Usage Example:[/]")
        console.print(
            f"  [dim]# Create from preset:[/]\n"
            f"  from victor.workflows.presets import create_workflow_from_preset\n\n"
            f"  workflow = create_workflow_from_preset('{preset_name}')\n"
            f"  result = await executor.execute(workflow, context)"
        )

    else:
        console.print(f"[bold red]Error:[/] Invalid preset type: {preset_type}")
        console.print("Valid types: agent, workflow")
        raise typer.Exit(1)

    console.print()


@workflow_app.command("generate")
def generate_workflow(
    description: str = typer.Argument(
        ...,
        help="Natural language description of the workflow to generate",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path for generated YAML (defaults to stdout)",
    ),
    vertical: str = typer.Option(
        "coding",
        "--vertical",
        "-V",
        help="Target vertical (coding, devops, research, rag, dataanalysis)",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Profile to use for LLM generation",
    ),
    strategy: str = typer.Option(
        "multi_stage",
        "--strategy",
        "-s",
        help="Generation strategy: multi_stage, single_stage, or template",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Enable interactive clarification for ambiguous requirements",
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate generated workflow after generation",
    ),
    max_retries: int = typer.Option(
        3,
        "--max-retries",
        help="Maximum LLM generation retry attempts",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Extract requirements only, don't generate workflow",
    ),
) -> None:
    """Generate a workflow YAML from natural language description.

    Uses LLM-based extraction to convert natural language descriptions
    into structured workflow definitions compatible with StateGraph.

    The generation pipeline:
    1. Extract requirements from natural language
    2. Optionally clarify ambiguous requirements (--interactive)
    3. Generate workflow schema using chosen strategy
    4. Validate generated workflow
    5. Output YAML file or display

    Strategies:
    - multi_stage: Most accurate, uses 3-stage LLM generation
    - single_stage: Faster, single LLM call
    - template: Fastest, matches to pre-defined templates

    Example:
        victor workflow generate "Analyze code, find bugs, fix them, run tests"
        victor workflow generate "Research topic, summarize findings, create report" -V research
        victor workflow generate "Build and deploy container" -V devops -o deploy.yaml
        victor workflow generate "Analyze data, visualize results" --interactive
    """
    from victor.config.settings import load_settings
    from victor.framework.shim import FrameworkShim
    from victor.workflows.generation import (
        WorkflowGenerationPipeline,
        GenerationStrategy,
        PipelineMode,
        RequirementPipeline,
        WorkflowValidator,
    )

    # Map strategy string to enum
    strategy_map = {
        "multi_stage": GenerationStrategy.LLM_MULTI_STAGE,
        "single_stage": GenerationStrategy.LLM_SINGLE_STAGE,
        "template": GenerationStrategy.TEMPLATE_BASED,
    }

    gen_strategy = strategy_map.get(strategy.lower())
    if not gen_strategy:
        console.print(f"[bold red]Error:[/] Unknown strategy: {strategy}")
        console.print(f"Valid options: {', '.join(strategy_map.keys())}")
        raise typer.Exit(1)

    # Validate vertical
    valid_verticals = {"coding", "devops", "research", "rag", "dataanalysis", "benchmark"}
    if vertical.lower() not in valid_verticals:
        console.print(f"[bold red]Error:[/] Unknown vertical: {vertical}")
        console.print(f"Valid options: {', '.join(valid_verticals)}")
        raise typer.Exit(1)

    console.print("\n[bold blue]Generating workflow from description...[/]")
    console.print("[dim]" + "─" * 50 + "[/]")
    console.print(
        f"[dim]Description:[/] {description[:100]}{'...' if len(description) > 100 else ''}"
    )
    console.print(f"[dim]Vertical:[/] {vertical}")
    console.print(f"[dim]Strategy:[/] {strategy}")

    async def _generate():
        settings = load_settings()

        # Create orchestrator for LLM access
        profile_name = profile or "default"
        try:
            shim = FrameworkShim(
                settings,
                profile_name=profile_name,
                vertical=vertical,
            )
            orchestrator = await shim.create_orchestrator()
            console.print(f"[dim]Using profile: {profile_name}[/]")
        except Exception as e:
            console.print(f"[bold red]Error:[/] Failed to create orchestrator: {e}")
            raise typer.Exit(1)

        try:
            # Step 1: Extract requirements
            console.print("\n[bold]Step 1:[/] Extracting requirements...")

            req_pipeline = RequirementPipeline(orchestrator, vertical=vertical)
            requirements = await req_pipeline.extract_and_validate(description)

            console.print(f"  [green]✓[/] Extracted {len(requirements.functional.tasks)} tasks")
            console.print(f"  [dim]Tools: {list(requirements.functional.tools.keys())}[/]")
            console.print(f"  [dim]Execution order: {requirements.structural.execution_order}[/]")

            # Check for ambiguities
            if requirements.metadata.ambiguities:
                console.print(
                    f"  [yellow]⚠[/] {len(requirements.metadata.ambiguities)} ambiguities detected"
                )
                for amb in requirements.metadata.ambiguities[:3]:
                    console.print(f"    - {amb.description}")

                if interactive:
                    console.print("\n[bold]Interactive clarification:[/]")
                    # In a real implementation, this would prompt the user
                    # For now, we proceed with defaults
                    console.print("  [dim](Interactive mode not yet fully implemented)[/]")

            if dry_run:
                console.print("\n[bold yellow]Dry run mode[/] - requirements extracted:")
                console.print(
                    Panel(
                        json.dumps(requirements.to_dict(), indent=2, default=str)[:2000],
                        title="Requirements",
                        border_style="blue",
                    )
                )
                return

            # Step 2: Generate workflow
            console.print(f"\n[bold]Step 2:[/] Generating workflow (strategy: {strategy})...")

            pipeline = WorkflowGenerationPipeline(
                orchestrator=orchestrator,
                vertical=vertical,
                generation_strategy=gen_strategy,
                max_retries=max_retries,
            )

            result = await pipeline.run(
                requirements,
                mode=PipelineMode.GENERATE_ONLY,
            )

            if not result.success:
                console.print(f"[bold red]✗[/] Generation failed: {result.error}")
                if result.validation_result and not result.validation_result.is_valid:
                    for err in result.validation_result.all_errors[:5]:
                        console.print(f"  - {err.message}")
                raise typer.Exit(1)

            console.print("  [green]✓[/] Generated workflow schema")
            console.print(f"  [dim]Duration: {result.metadata.duration_seconds:.2f}s[/]")

            # Step 3: Convert to YAML
            console.print("\n[bold]Step 3:[/] Converting to YAML...")

            import yaml

            workflow_schema = result.workflow_schema
            if workflow_schema is None:
                console.print("[bold red]✗[/] No workflow schema generated")
                raise typer.Exit(1)

            # Format for YAML output
            yaml_data = {
                "workflows": {
                    workflow_schema.get("workflow_name", "generated_workflow"): {
                        "description": workflow_schema.get("description", description[:100]),
                        "metadata": {
                            "vertical": vertical,
                            "generated_by": "victor workflow generate",
                            "generation_strategy": strategy,
                        },
                        "nodes": _convert_nodes_to_yaml(workflow_schema.get("nodes", [])),
                    }
                }
            }

            yaml_output = yaml.dump(
                yaml_data,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

            # Step 4: Validate if requested
            if validate:
                console.print("\n[bold]Step 4:[/] Validating generated workflow...")
                validator = WorkflowValidator()
                val_result = validator.validate(workflow_schema)

                if val_result.is_valid:
                    console.print("  [green]✓[/] Validation passed")
                else:
                    console.print(
                        f"  [yellow]⚠[/] Validation warnings: {len(val_result.all_errors)}"
                    )
                    for err in val_result.all_errors[:3]:
                        console.print(f"    - {err.message}")

            # Output
            console.print("\n[dim]" + "─" * 50 + "[/]")

            if output:
                output_path = Path(output)
                output_path.write_text(yaml_output)
                console.print(f"[bold green]✓[/] Saved to {output}")
            else:
                console.print("[bold]Generated Workflow:[/]\n")
                console.print(yaml_output)

            console.print("\n[bold green]✓[/] Workflow generation complete")

        finally:
            # Cleanup
            from victor.ui.commands.utils import graceful_shutdown

            try:
                await graceful_shutdown(orchestrator)
            except Exception:
                pass

    def _convert_nodes_to_yaml(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert node list to YAML-compatible format."""
        yaml_nodes = []
        for node in nodes:
            yaml_node: Dict[str, Any] = {
                "id": node.get("id", "unknown"),
                "type": node.get("type", "agent"),
            }

            # Add type-specific fields
            node_type = node.get("type", "agent")
            if node_type == "agent":
                yaml_node["role"] = node.get("role", "executor")
                yaml_node["goal"] = node.get("goal", "")
                if "tool_budget" in node:
                    yaml_node["tool_budget"] = node["tool_budget"]
                if "tools" in node:
                    yaml_node["tools"] = node["tools"]

            elif node_type == "compute":
                if "handler" in node:
                    yaml_node["handler"] = node["handler"]
                if "input_mapping" in node:
                    yaml_node["inputs"] = node["input_mapping"]

            elif node_type == "condition":
                yaml_node["condition"] = node.get("condition", "")
                yaml_node["branches"] = node.get("branches", {})

            elif node_type == "parallel":
                yaml_node["parallel_nodes"] = node.get("parallel_nodes", [])

            elif node_type == "transform":
                yaml_node["transform"] = node.get("transform", "")

            # Add output key if present
            if "output_key" in node:
                yaml_node["output"] = node["output_key"]

            # Add next nodes
            if "next_nodes" in node:
                yaml_node["next"] = node["next_nodes"]
            elif "edges" in node:
                # Extract targets from edges
                targets = [e.get("target") for e in node.get("edges", [])]
                if targets:
                    yaml_node["next"] = targets

            yaml_nodes.append(yaml_node)

        return yaml_nodes

    asyncio.run(_generate())


@workflow_app.command("debug")
def debug_workflow(
    workflow_path: str = typer.Argument(
        ...,
        help="Path to YAML workflow file to debug",
    ),
    context: Optional[str] = typer.Option(
        None,
        "--context",
        "-c",
        help='Initial context as JSON string (e.g., \'{"symbol": "AAPL"}\')',
    ),
    breakpoints: Optional[str] = typer.Option(
        None,
        "--breakpoints",
        "-b",
        help="Comma-separated list of node IDs to set breakpoints on",
    ),
    workflow_name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Workflow name to debug (if file contains multiple workflows)",
    ),
    stop_on_entry: bool = typer.Option(
        False,
        "--stop-on-entry",
        help="Stop before executing first node",
    ),
    log_level: Optional[str] = typer.Option(
        "DEBUG",
        "--log-level",
        help="Set logging level (default: DEBUG for debugging)",
    ),
) -> None:
    """Debug a workflow with breakpoints and step execution.

    Provides interactive debugging capabilities including:
    - Set breakpoints on specific nodes
    - Step through execution (step over, into, out)
    - Inspect state at any point
    - View variable values
    - Continue to next breakpoint

    Example:
        victor workflow debug ./workflows/analysis.yaml -c '{"symbol": "AAPL"}'
        victor workflow debug ./workflows/analysis.yaml -b node_1,node_3
        victor workflow debug ./workflows/analysis.yaml --stop-on-entry
    """
    from victor.workflows.execution_engine import WorkflowExecutor
    from victor.workflows.debugger import WorkflowDebugger

    # Set debug logging
    import logging

    level = getattr(logging, log_level.upper(), logging.DEBUG) if log_level else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    path = Path(workflow_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/] Workflow file not found: {path}")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]Debugging:[/] {path.name}")
    console.print("[dim]" + "─" * 50 + "[/]")

    # Parse context
    initial_context = {}
    if context:
        try:
            initial_context = json.loads(context)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Error:[/] Invalid JSON in --context: {e}")
            raise typer.Exit(1)

    # Parse breakpoints
    breakpoint_list = []
    if breakpoints:
        breakpoint_list = [b.strip() for b in breakpoints.split(",")]
        console.print(f"[dim]Breakpoints: {', '.join(breakpoint_list)}[/]")

    console.print("\n[bold yellow]Debug mode:[/] Starting debugger...")
    console.print("[dim](Note: Interactive debugging is under development)[/]\n")

    # Execute with debug mode
    async def _debug():
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_yaml(path, workflow_name)

        # Create executor with debug mode
        executor = WorkflowExecutor(debug_mode=True, trace_mode=True)

        try:
            result = await executor.execute(
                compiled,
                initial_context,
                breakpoints=breakpoint_list,
            )

            console.print("\n[dim]" + "─" * 50 + "[/]")
            console.print("[bold green]✓[/] Debugging completed")

            # Show trace summary
            trace = executor.get_trace()
            if trace:
                summary = trace.get_summary()
                console.print(f"\n[bold]Execution Summary:[/]")
                console.print(f"  Duration: {summary['duration_seconds']:.3f}s")
                console.print(f"  Total events: {summary['total_events']}")
                console.print(f"  Error count: {summary['error_count']}")

        except Exception as e:
            console.print(f"\n[bold red]✗[/] Debugging failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_debug())


@workflow_app.command("trace")
def trace_workflow(
    workflow_path: str = typer.Argument(
        ...,
        help="Path to YAML workflow file to trace",
    ),
    context: Optional[str] = typer.Option(
        None,
        "--context",
        "-c",
        help="Initial context as JSON string",
    ),
    trace_file: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output trace file path (JSON format)",
    ),
    workflow_name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Workflow name to trace (if file contains multiple workflows)",
    ),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Trace output format: json, csv, html",
    ),
) -> None:
    """Execute workflow with execution tracing.

    Captures detailed trace logs including:
    - All node executions with inputs/outputs
    - Execution time for each node
    - Tool calls and their results
    - Errors and failures
    - Performance metrics

    Example:
        victor workflow trace ./workflows/analysis.yaml -c '{"symbol": "AAPL"}'
        victor workflow trace ./workflows/analysis.yaml -o trace.json
        victor workflow trace ./workflows/analysis.yaml --format html -o report.html
    """
    from victor.workflows.execution_engine import WorkflowExecutor

    path = Path(workflow_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/] Workflow file not found: {path}")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]Tracing:[/] {path.name}")
    console.print("[dim]" + "─" * 50 + "[/]")

    # Parse context
    initial_context = {}
    if context:
        try:
            initial_context = json.loads(context)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Error:[/] Invalid JSON in --context: {e}")
            raise typer.Exit(1)

    console.print("\n[bold]Starting workflow with tracing...[/]\n")

    # Execute with trace mode
    async def _trace():
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_yaml(path, workflow_name)

        # Create executor with trace mode
        executor = WorkflowExecutor(trace_mode=True)

        try:
            result = await executor.execute(compiled, initial_context)

            console.print("\n[dim]" + "─" * 50 + "[/]")
            console.print("[bold green]✓[/] Workflow execution completed")

            # Export trace
            trace = executor.get_trace()
            if trace:
                if trace_file:
                    output_path = Path(trace_file)
                    executor.export_trace(output_path)
                    console.print(f"[bold]Trace saved to:[/] {output_path}")
                else:
                    # Show trace summary
                    summary = trace.get_summary()
                    console.print(f"\n[bold]Trace Summary:[/]")
                    console.print(f"  Duration: {summary['duration_seconds']:.3f}s")
                    console.print(f"  Total events: {summary['total_events']}")
                    console.print(f"  Nodes executed: {len(summary['node_counts'])}")
                    console.print(f"  Error count: {summary['error_count']}")

        except Exception as e:
            console.print(f"\n[bold red]✗[/] Tracing failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_trace())


@workflow_app.command("history")
def show_history(
    workflow_name: Optional[str] = typer.Option(
        None,
        "--workflow",
        "-w",
        help="Filter by workflow name",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Maximum number of executions to show",
    ),
    export_file: Optional[str] = typer.Option(
        None,
        "--export",
        "-e",
        help="Export history to file",
    ),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Export format: json, csv",
    ),
) -> None:
    """Show workflow execution history.

    Displays past workflow executions from history, including:
    - Execution ID and timestamp
    - Workflow name
    - Success/failure status
    - Execution duration
    - Node execution counts

    Example:
        victor workflow history
        victor workflow history --workflow my_workflow --limit 20
        victor workflow history --export history.json
        victor workflow history --format csv --export report.csv
    """
    from victor.workflows.execution_engine import WorkflowExecutor

    executor = WorkflowExecutor()
    history = executor.get_history(limit=limit)

    if workflow_name:
        history = [h for h in history if h.get("workflow_name") == workflow_name]

    if not history:
        console.print("[dim]No execution history found[/]")
        return

    # Display history
    table = Table(title="Workflow Execution History")
    table.add_column("Execution ID", style="cyan")
    table.add_column("Workflow", style="green")
    table.add_column("Timestamp")
    table.add_column("Duration", justify="right")
    table.add_column("Success")
    table.add_column("Nodes", justify="right")

    for record in history:
        timestamp = datetime.fromtimestamp(record.get("start_time", 0)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        success_str = "[green]✓[/]" if record.get("success", True) else "[red]✗[/]"

        table.add_row(
            record.get("execution_id", "unknown")[:8],
            record.get("workflow_name", "unknown"),
            timestamp,
            f"{record.get('duration', 0):.2f}s",
            success_str,
            str(record.get("nodes_executed", 0)),
        )

    console.print(table)

    # Export if requested
    if export_file:
        import json

        output_path = Path(export_file)

        if format == "json":
            output_path.write_text(json.dumps(history, indent=2, default=str))
        elif format == "csv":
            import csv

            with open(output_path, "w", newline="") as f:
                if history:
                    writer = csv.DictWriter(f, fieldnames=history[0].keys())
                    writer.writeheader()
                    writer.writerows(history)

        console.print(f"\n[bold green]✓[/] History exported to {output_path}")


@workflow_app.command("replay")
def replay_execution(
    execution_id: str = typer.Argument(
        ...,
        help="Execution ID to replay",
    ),
    export_trace: bool = typer.Option(
        False,
        "--export-trace",
        "-e",
        help="Export trace from replayed execution",
    ),
) -> None:
    """Replay a previous workflow execution.

    Re-runs a workflow with the same inputs as a previous execution,
    allowing you to compare results or debug issues.

    Example:
        victor workflow replay exec_abc123
        victor workflow replay exec_abc123 --export-trace
    """
    from victor.workflows.execution_engine import WorkflowExecutor

    console.print(f"\n[bold blue]Replaying:[/] {execution_id}")
    console.print("[dim]" + "─" * 50 + "[/]")

    executor = WorkflowExecutor()
    history = executor.get_history()

    # Find execution
    execution = None
    for record in history:
        if record.get("execution_id", "").startswith(execution_id):
            execution = record
            break

    if not execution:
        console.print(f"[bold red]Error:[/] Execution '{execution_id}' not found in history")
        console.print("[dim]Use 'victor workflow history' to list executions[/]")
        raise typer.Exit(1)

    console.print(f"Workflow: {execution.get('workflow_name')}")
    console.print(f"Original duration: {execution.get('duration', 0):.2f}s\n")

    console.print("[bold yellow]Replay is under development[/]")
    console.print("[dim](Use 'victor workflow run' with same inputs for now)[/]")
    console.print(f"\n[dim]Original inputs:[/]")
    console.print(json.dumps(execution.get("inputs", {}), indent=2)[:500])


@workflow_app.command("inspect")
def inspect_state(
    execution_id: str = typer.Argument(
        ...,
        help="Execution ID to inspect",
    ),
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Query specific state variable (e.g., 'user.name')",
    ),
    snapshot: Optional[str] = typer.Option(
        None,
        "--snapshot",
        "-s",
        help="Snapshot ID to inspect (default: latest)",
    ),
) -> None:
    """Inspect workflow execution state.

    View the state of a workflow execution at a specific point,
    including variables, node results, and metadata.

    Example:
        victor workflow inspect exec_abc123
        victor workflow inspect exec_abc123 --query user.name
        victor workflow inspect exec_abc123 --snapshot snap_xyz
    """
    from victor.workflows.execution_engine import WorkflowExecutor

    console.print(f"\n[bold blue]Inspecting:[/] {execution_id}")
    console.print("[dim]" + "─" * 50 + "[/]")

    console.print("[bold yellow]State inspection is under development[/]")
    console.print("[dim](Use 'victor workflow trace --output trace.json' for detailed state)[/]")


@workflow_app.command("metrics")
def show_metrics(
    execution_id: Optional[str] = typer.Option(
        None,
        "--execution",
        "-e",
        help="Execution ID to show metrics for",
    ),
    workflow_name: Optional[str] = typer.Option(
        None,
        "--workflow",
        "-w",
        help="Show metrics for workflow (all executions)",
    ),
    export_file: Optional[str] = typer.Option(
        None,
        "--export",
        "-o",
        help="Export metrics to file",
    ),
) -> None:
    """Show workflow execution metrics.

    Displays performance metrics and analytics:
    - Total executions
    - Success/failure rates
    - Average duration
    - Node execution counts
    - Tool usage statistics

    Example:
        victor workflow metrics --workflow my_workflow
        victor workflow metrics --execution exec_abc123
        victor workflow metrics --export metrics.json
    """
    from victor.workflows.execution_engine import WorkflowExecutor

    if not execution_id and not workflow_name:
        console.print("[bold red]Error:[/] Specify --execution or --workflow")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]Metrics:[/]")
    console.print("[dim]" + "─" * 50 + "[/]")

    if workflow_name:
        executor = WorkflowExecutor()
        history = executor.get_history(limit=1000)

        # Filter by workflow
        workflow_executions = [h for h in history if h.get("workflow_name") == workflow_name]

        if not workflow_executions:
            console.print(f"[dim]No executions found for workflow '{workflow_name}'[/]")
            return

        # Calculate metrics
        total = len(workflow_executions)
        successful = sum(1 for h in workflow_executions if h.get("success", True))
        failed = total - successful
        durations = [h.get("duration", 0) for h in workflow_executions]

        console.print(f"[bold]Workflow:[/] {workflow_name}")
        console.print(f"  Total executions: {total}")
        console.print(f"  Successful: [green]{successful}[/] ({successful/total*100:.1f}%)")
        console.print(f"  Failed: [red]{failed}[/] ({failed/total*100:.1f}%)")
        console.print(f"  Avg duration: {sum(durations)/total:.2f}s")
        console.print(f"  Min duration: {min(durations):.2f}s")
        console.print(f"  Max duration: {max(durations):.2f}s")

    elif execution_id:
        # Show specific execution metrics
        executor = WorkflowExecutor()
        history = executor.get_history()

        execution = None
        for h in history:
            if h.get("execution_id", "").startswith(execution_id):
                execution = h
                break

        if not execution:
            console.print(f"[bold red]Error:[/] Execution '{execution_id}' not found")
            raise typer.Exit(1)

        console.print(f"[bold]Execution:[/] {execution_id}")
        console.print(f"  Workflow: {execution.get('workflow_name')}")
        console.print(f"  Duration: {execution.get('duration', 0):.2f}s")
        console.print(
            f"  Success: [green]Yes[/]"
            if execution.get("success", True)
            else "  Success: [red]No[/]"
        )
        console.print(f"  Nodes executed: {execution.get('nodes_executed', 0)}")

        metrics = execution.get("metrics", {})
        if metrics:
            console.print(f"\n[bold]Detailed Metrics:[/]")
            for key, value in metrics.items():
                console.print(f"  {key}: {value}")

    console.print("\n[dim](Detailed metrics export coming soon)[/]")

    # Export if requested
    if export_file:
        import json
        import time

        output_path = Path(export_file)
        metrics_data = {
            "workflow_name": workflow_name,
            "timestamp": time.time(),
            "summary": {
                "total_executions": len(workflow_executions) if workflow_name else 1,
                "successful": successful if workflow_name else None,
                "failed": failed if workflow_name else None,
                "average_duration": sum(durations) / total if workflow_name else None,
            },
        }

        output_path.write_text(json.dumps(metrics_data, indent=2, default=str))
        console.print(f"\n[bold green]✓[/] Metrics exported to {output_path}")
