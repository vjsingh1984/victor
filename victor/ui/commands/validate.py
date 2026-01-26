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

"""CLI command for validating code files using the unified language capability system."""

import typer
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table

validate_app = typer.Typer(
    name="validate",
    help="Validate code files for syntax errors using the unified language capability system.",
)
console = Console()


@validate_app.command("files")
def validate_files(
    files: List[Path] = typer.Argument(
        ...,
        help="Files to validate.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Strict mode - exit with error on any validation failure",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Override language detection (e.g., python, javascript, json)",
    ),
    show_warnings: bool = typer.Option(
        True,
        "--warnings/--no-warnings",
        "-w/-W",
        help="Show warning messages",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output with validator details",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
):
    """Validate code files for syntax errors.

    Examples:
        victor validate files main.py
        victor validate files src/*.py --strict
        victor validate files config.json --language json
        victor validate files --json app.ts
    """
    # Import here to avoid slow startup
    from victor.core.language_capabilities import LanguageCapabilityRegistry
    from victor.core.language_capabilities.validators import UnifiedLanguageValidator
    from victor.core.language_capabilities.types import ValidationSeverity

    registry = LanguageCapabilityRegistry.instance()
    validator = UnifiedLanguageValidator(registry=registry)

    results = []
    has_errors = False
    total_errors = 0
    total_warnings = 0

    for file_path in files:
        if not file_path.exists():
            console.print(f"[red]Error:[/red] File not found: {file_path}")
            has_errors = True
            continue

        if not file_path.is_file():
            console.print(f"[yellow]Skipping directory:[/yellow] {file_path}")
            continue

        try:
            content = file_path.read_text()
        except UnicodeDecodeError:
            console.print(f"[yellow]Skipping binary file:[/yellow] {file_path}")
            continue
        except Exception as e:
            console.print(f"[red]Error reading {file_path}:[/red] {e}")
            has_errors = True
            continue

        # Validate the file
        result = validator.validate(content, file_path, language=language)

        file_result = {
            "file": str(file_path),
            "valid": result.is_valid,
            "language": result.language,
            "tier": result.tier.name if result.tier else "unknown",
            "validators": result.validators_used,
            "errors": [],
            "warnings": [],
        }

        for issue in result.errors:
            file_result["errors"].append(
                {
                    "line": issue.line,
                    "column": issue.column,
                    "message": issue.message,
                    "source": issue.source,
                }
            )
            total_errors += 1

        for warning in result.warnings:
            file_result["warnings"].append(
                {
                    "line": warning.line,
                    "column": warning.column,
                    "message": warning.message,
                    "source": warning.source,
                }
            )
            total_warnings += 1

        results.append(file_result)

        if not result.is_valid:
            has_errors = True

    # Output results
    if json_output:
        import json

        output = {
            "results": results,
            "summary": {
                "total_files": len(results),
                "valid_files": sum(1 for r in results if r["valid"]),
                "invalid_files": sum(1 for r in results if not r["valid"]),
                "total_errors": total_errors,
                "total_warnings": total_warnings,
            },
        }
        console.print(json.dumps(output, indent=2))
    else:
        _print_results(results, verbose, show_warnings, console)

    if strict and has_errors:
        raise typer.Exit(1)
    elif has_errors:
        raise typer.Exit(0)  # Non-strict mode: exit 0 even with errors


def _print_results(
    results: List[dict],
    verbose: bool,
    show_warnings: bool,
    console: Console,
) -> None:
    """Print validation results in human-readable format."""
    valid_count = 0
    invalid_count = 0

    for result in results:
        file_path = result["file"]
        is_valid = result["valid"]
        language = result["language"]
        errors = result["errors"]
        warnings = result["warnings"]

        if is_valid:
            valid_count += 1
            if verbose:
                validators = ", ".join(result["validators"]) or "none"
                console.print(
                    f"[green]✓[/green] {file_path} "
                    f"[dim]({language}, tier={result['tier']}, validators=[{validators}])[/dim]"
                )
            else:
                console.print(f"[green]✓[/green] {file_path}")
        else:
            invalid_count += 1
            console.print(f"[red]✗[/red] {file_path}")

            for error in errors:
                line_info = f":{error['line']}" if error["line"] else ""
                col_info = f":{error['column']}" if error["column"] else ""
                source_info = f" [{error['source']}]" if verbose and error["source"] else ""
                console.print(
                    f"  [red]error{line_info}{col_info}:[/red] {error['message']}{source_info}"
                )

        if show_warnings and warnings:
            for warning in warnings:
                line_info = f":{warning['line']}" if warning["line"] else ""
                col_info = f":{warning['column']}" if warning["column"] else ""
                source_info = f" [{warning['source']}]" if verbose and warning["source"] else ""
                console.print(
                    f"  [yellow]warning{line_info}{col_info}:[/yellow] {warning['message']}{source_info}"
                )

    # Summary
    total = valid_count + invalid_count
    if total > 1:
        console.print()
        if invalid_count == 0:
            console.print(f"[green]All {total} files validated successfully[/green]")
        else:
            console.print(
                f"[dim]Summary:[/dim] {valid_count} valid, "
                f"[red]{invalid_count} invalid[/red] ({total} total)"
            )


@validate_app.command()
def languages(
    tier: Optional[int] = typer.Option(
        None,
        "--tier",
        "-t",
        help="Filter by tier (1, 2, or 3)",
    ),
):
    """List supported languages and their validation capabilities."""
    from victor.core.language_capabilities import LanguageCapabilityRegistry
    from victor.core.language_capabilities.types import LanguageTier

    registry = LanguageCapabilityRegistry.instance()

    tier_filter = None
    if tier:
        tier_map = {1: LanguageTier.TIER_1, 2: LanguageTier.TIER_2, 3: LanguageTier.TIER_3}
        tier_filter = tier_map.get(tier)

    languages = registry.list_supported_languages(tier=tier_filter)

    table = Table(title="Supported Languages for Validation")
    table.add_column("Language", style="cyan")
    table.add_column("Tier", style="green")
    table.add_column("Extensions", style="dim")
    table.add_column("Native AST", style="yellow")
    table.add_column("Tree-sitter", style="blue")
    table.add_column("LSP", style="magenta")

    for lang in languages:
        cap = registry.get(lang)
        if cap:
            native = "✓" if cap.native_ast else "-"
            ts = "✓" if cap.tree_sitter else "-"
            lsp = "✓" if cap.lsp else "-"
            exts = ", ".join(cap.extensions[:3])
            if len(cap.extensions) > 3:
                exts += f" (+{len(cap.extensions) - 3})"

            table.add_row(
                lang,
                f"Tier {cap.tier.value}" if cap.tier else "?",
                exts,
                native,
                ts,
                lsp,
            )

    console.print(table)
    console.print(f"\n[dim]Total: {len(languages)} languages[/dim]")


@validate_app.command()
def check(
    file_path: Path = typer.Argument(..., help="File to check validation support for"),
):
    """Check if validation is supported for a specific file."""
    from victor.core.language_capabilities import LanguageCapabilityRegistry
    from victor.core.language_capabilities.validators import UnifiedLanguageValidator

    registry = LanguageCapabilityRegistry.instance()
    validator = UnifiedLanguageValidator(registry=registry)

    cap = registry.get_for_file(file_path)

    if not cap:
        console.print(f"[yellow]Unknown file type:[/yellow] {file_path}")
        console.print("[dim]No language capability registered for this file extension.[/dim]")
        raise typer.Exit(1)

    console.print(f"[cyan]File:[/cyan] {file_path}")
    console.print(f"[cyan]Language:[/cyan] {cap.name}")
    console.print(f"[cyan]Tier:[/cyan] {cap.tier.name if cap.tier else 'unknown'}")

    can_validate = validator.can_validate(file_path)
    method = validator.get_validation_method(file_path)

    console.print(f"[cyan]Can validate:[/cyan] {'Yes' if can_validate else 'No'}")
    if method:
        console.print(f"[cyan]Validation method:[/cyan] {method.value}")

    console.print(f"[cyan]Validation enabled:[/cyan] {'Yes' if cap.validation_enabled else 'No'}")

    if cap.native_ast:
        console.print(f"[cyan]Native AST:[/cyan] {cap.native_ast.library}")
    if cap.tree_sitter:
        console.print(f"[cyan]Tree-sitter:[/cyan] {cap.tree_sitter.grammar_package}")
    if cap.lsp:
        console.print(f"[cyan]LSP server:[/cyan] {cap.lsp.server_name}")
