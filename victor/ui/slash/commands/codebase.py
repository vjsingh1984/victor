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

"""Codebase management slash commands: reindex, init."""

from __future__ import annotations

import logging

from rich.markdown import Markdown
from rich.panel import Panel

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class ReindexCommand(BaseSlashCommand):
    """Reindex codebase for semantic search."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="reindex",
            description="Reindex codebase for semantic search",
            usage="/reindex [--force] [--stats]",
            aliases=["index"],
            category="codebase",
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        force = self._has_flag(ctx, "--force", "-f")
        show_stats = self._has_flag(ctx, "--stats", "-s")

        try:
            from victor_coding.codebase.embeddings.manager import get_embedding_manager

            manager = get_embedding_manager()

            if show_stats:
                stats = manager.get_index_stats()
                content = (
                    f"[bold]Index Statistics[/]\n\n"
                    f"[bold]Total Documents:[/] {stats.get('total_documents', 0)}\n"
                    f"[bold]Total Chunks:[/] {stats.get('total_chunks', 0)}\n"
                    f"[bold]Last Updated:[/] {stats.get('last_updated', 'Never')}\n"
                    f"[bold]Index Size:[/] {stats.get('size_mb', 0):.1f} MB\n"
                )
                ctx.console.print(Panel(content, title="Embedding Index", border_style="cyan"))
                return

            ctx.console.print("[dim]Indexing codebase for semantic search...[/]")

            if force:
                ctx.console.print("[dim]Force reindex: rebuilding from scratch[/]")
                manager.clear_index()

            # Run indexing
            result = await manager.index_directory(
                directory=".",
                incremental=not force,
            )

            ctx.console.print(
                Panel(
                    f"[green]Indexing complete![/]\n\n"
                    f"[bold]Files Indexed:[/] {result.get('files_indexed', 0)}\n"
                    f"[bold]Chunks Created:[/] {result.get('chunks_created', 0)}\n"
                    f"[bold]Time:[/] {result.get('duration_seconds', 0):.1f}s",
                    title="Reindex Complete",
                    border_style="green",
                )
            )

        except ImportError:
            ctx.console.print("[yellow]Embedding manager not available[/]")
            ctx.console.print("[dim]Make sure sentence-transformers is installed[/]")
        except Exception as e:
            ctx.console.print(f"[red]Indexing failed:[/] {e}")
            logger.exception("Reindex error")


@register_command
class InitCommand(BaseSlashCommand):
    """Initialize or update .victor/init.md with codebase analysis."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="init",
            description="Initialize or update .victor/init.md with codebase analysis",
            usage="/init [--index] [--update] [--force] [--deep] [--symlinks]",
            category="codebase",
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        from victor.config.settings import get_project_paths

        force = self._has_flag(ctx, "--force", "-f")
        update = self._has_flag(ctx, "--update", "-u")
        deep = self._has_flag(ctx, "--deep", "-d", "--smart")
        use_index = self._has_flag(ctx, "--index", "-i")
        use_learn = self._has_flag(ctx, "--learn", "-L")
        symlinks = self._has_flag(ctx, "--symlinks", "-l")

        paths = get_project_paths()
        target_path = paths.project_context_file
        paths.project_victor_dir.mkdir(parents=True, exist_ok=True)

        existing_content = None
        if target_path.exists():
            existing_content = target_path.read_text(encoding="utf-8")

            if not force and not update:
                ctx.console.print(f"[yellow]{target_path.name} already exists[/]")
                ctx.console.print("")
                ctx.console.print("[bold]Options:[/]")
                ctx.console.print(
                    "  [cyan]/init --update[/]   Merge new analysis (preserves your edits)"
                )
                ctx.console.print("  [cyan]/init --force[/]    Overwrite completely")
                ctx.console.print(
                    "  [cyan]/init --learn[/]    Enhance with conversation history insights"
                )
                ctx.console.print("  [cyan]/init --deep[/]     LLM-powered analysis (any language)")
                ctx.console.print("  [cyan]/init --symlinks[/] Create CLAUDE.md and other aliases")

                preview = existing_content[:500]
                ctx.console.print(
                    Panel(
                        Markdown(preview + "\n\n..." if len(existing_content) > 500 else preview),
                        title=f"Current {target_path.name}",
                        border_style="dim",
                    )
                )
                return

        if use_learn:
            ctx.console.print("[dim]Analyzing codebase + learning from conversation history...[/]")
        elif use_index:
            ctx.console.print("[dim]Indexing codebase (multi-language symbol analysis)...[/]")
        elif deep:
            ctx.console.print("[dim]Analyzing codebase with LLM (deep mode)...[/]")
        elif update and existing_content:
            ctx.console.print("[dim]Updating codebase analysis (preserving user sections)...[/]")
        else:
            ctx.console.print("[dim]Analyzing codebase (quick mode)...[/]")

        try:
            from victor.context.codebase_analyzer import (
                generate_enhanced_init_md,
                generate_smart_victor_md,
                generate_victor_md_from_index,
            )

            if deep or use_learn:
                new_content = await generate_enhanced_init_md(
                    use_llm=deep,
                    include_conversations=use_learn or deep,
                )
            elif use_index:
                new_content = await generate_victor_md_from_index()
            else:
                new_content = generate_smart_victor_md()

            # Handle update mode
            if update and existing_content:
                content = self.merge_init_content(existing_content, new_content)
                ctx.console.print("[dim]  Merged with existing content[/]")
            else:
                content = new_content

            # Write the file
            target_path.write_text(content, encoding="utf-8")
            ctx.console.print(f"[green]Created {target_path}[/]")

            # Show what was detected
            lines = content.split("\n")
            component_count = content.count("| `")
            pattern_count = content.count(". **") + content.count("Pattern:")

            ctx.console.print(
                f"[dim]  Lines: {len(lines)}, Components: {component_count}, Patterns: {pattern_count}[/]"
            )

            # Create symlinks if requested
            if symlinks:
                self._create_symlinks(target_path, ctx)

        except ImportError as e:
            ctx.console.print(f"[red]Missing dependency:[/] {e}")
        except Exception as e:
            ctx.console.print(f"[red]Analysis failed:[/] {e}")
            logger.exception("Init error")

    def merge_init_content(self, existing: str, new: str) -> str:
        """Merge new analysis with existing content, preserving user sections.

        This is a public method for use by CLI commands that also need merging
        functionality.
        """
        # Simple merge: keep user-added sections, update auto-generated ones
        # Look for markers like "<!-- AUTO-GENERATED -->" or "## Project Overview"

        # For now, a simple strategy: append new sections that don't exist
        existing_sections = set()
        for line in existing.split("\n"):
            if line.startswith("## "):
                existing_sections.add(line.strip())

        new_lines = []
        current_section = None
        skip_section = False

        for line in new.split("\n"):
            if line.startswith("## "):
                current_section = line.strip()
                skip_section = current_section in existing_sections
                if skip_section:
                    continue

            if not skip_section:
                new_lines.append(line)

        if new_lines:
            return existing + "\n\n" + "\n".join(new_lines)
        return existing

    def _create_symlinks(self, target: "Path", ctx: CommandContext) -> None:
        """Create symlinks for other tools (CLAUDE.md, etc.)."""
        from pathlib import Path

        aliases = ["CLAUDE.md", "CURSOR.md"]
        for alias in aliases:
            alias_path = target.parent.parent / alias
            if not alias_path.exists():
                try:
                    alias_path.symlink_to(target)
                    ctx.console.print(f"[dim]  Created symlink: {alias}[/]")
                except Exception:
                    # Fall back to copying
                    alias_path.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
                    ctx.console.print(f"[dim]  Created copy: {alias}[/]")
