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

"""Markdown generation for codebase analysis output.

Handles:
- VictorMDBuilder for composing markdown sections
- Smart init.md generation (AST-based and generic)
- LLM-powered init.md generation
- Enhanced init.md generation with graph/conversation insights
- Project context gathering for LLM analysis
- Helper functions (readme extraction, command inference, etc.)
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from victor.config.settings import VICTOR_CONTEXT_FILE, get_project_paths

from .scanner import (
    DEFAULT_SKIP_DIRS,
    should_ignore_path,
)

logger = logging.getLogger(__name__)


class VictorMDBuilder:
    """Lightweight helper for composing Victor context markdown."""

    def __init__(self, context_file: str = VICTOR_CONTEXT_FILE):
        self._context_file = context_file
        self.sections: List[str] = []
        self._started = False

    def start_document(self, project_name: str, description: str) -> None:
        if self._started:
            raise ValueError("VictorMDBuilder.start_document already called")
        self._started = True
        self.sections.append(f"# {self._context_file}\n")
        self.sections.append(
            "This file provides guidance to Victor when working with code in this repository.\n"
        )
        self.sections.append("## Project Overview\n")
        self.sections.append(f"**{project_name}**: {description}\n")

    def add_table(
        self,
        title: str,
        headers: List[str],
        rows: List[List[str]],
        separator: Optional[str] = None,
    ) -> None:
        if not rows:
            return
        if title:
            self.sections.append(title if title.endswith("\n") else f"{title}\n")
        header_line = "| " + " | ".join(headers) + " |"
        self.sections.append(header_line)
        if separator is not None:
            self.sections.append(separator)
        else:
            sep_cells = ["-" * max(3, len(h)) for h in headers]
            self.sections.append("|" + "|".join(sep_cells) + "|")
        for row in rows:
            self.sections.append("| " + " | ".join(row) + " |")
        self.sections.append("")

    def add_code_block(self, lines: List[str], language: str = "bash") -> None:
        if not lines:
            return
        self.sections.append(f"```{language}")
        self.sections.extend(lines)
        self.sections.append("```\n")

    def blank_line(self) -> None:
        self.sections.append("")

    def add_package_layout(self, rows: List[List[str]], note: Optional[str] = None) -> None:
        if not rows:
            return
        self.append("## Package Layout\n")
        if note:
            self.append(note if note.endswith("\n") else f"{note}\n")
        self.add_table(
            "",
            ["Path", "Type", "Description"],
            rows,
            separator="|------|------|-------------|",
        )

    def add_key_components_table(self, rows: List[List[str]]) -> None:
        if not rows:
            return
        self.add_table(
            "## Key Components\n",
            ["Component", "Type", "Path", "Description"],
            rows,
            separator="|-----------|------|------|-------------|",
        )

    def add_command_section(self, title: str, commands: List[str], language: str = "bash") -> None:
        if not commands:
            return
        self.append(title if title.endswith("\n") else f"{title}\n")
        self.add_code_block(commands, language=language)

    def append(self, text: str) -> None:
        self.sections.append(text)

    def build(self) -> str:
        return "\n".join(self.sections)


# Supported context file aliases for other AI coding tools
CONTEXT_FILE_ALIASES = {
    "CLAUDE.md": "Claude Code (Anthropic)",
    "GEMINI.md": "Gemini (Google AI Studio)",
    ".cursorrules": "Cursor IDE",
    ".windsurfrules": "Windsurf IDE",
    "AGENTS.md": "Generic AI agents",
}


def generate_smart_victor_md(
    root_path: Optional[str] = None,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Generate comprehensive project context using codebase analysis.

    Works with Python projects (AST-based analysis) and falls back to
    language-agnostic analysis for non-Python projects.
    """
    # Import here to avoid circular imports
    from . import CodebaseAnalyzer

    analyzer = CodebaseAnalyzer(root_path, include_dirs=include_dirs, exclude_dirs=exclude_dirs)
    analysis = analyzer.analyze()

    if not analysis.main_package and not analysis.key_components:
        return _generate_generic_victor_md(
            root_path, include_dirs=include_dirs, exclude_dirs=exclude_dirs
        )

    builder = VictorMDBuilder()
    readme_desc = (
        _extract_readme_description(analysis.root_path) or "[Add project description here]"
    )
    builder.start_document(analysis.project_name, readme_desc)
    sections = builder.sections

    layout_rows: List[List[str]] = []
    if analysis.main_package:
        layout_rows.append(
            [f"`{analysis.main_package}/`", "**ACTIVE**", "Main package - all source code"]
        )

    for deprecated in analysis.deprecated_paths:
        layout_rows.append([f"`{deprecated}`", "**DEPRECATED**", "Legacy - DO NOT USE"])

    if (analysis.root_path / "tests").is_dir():
        layout_rows.append(["`tests/`", "Active", "Unit and integration tests"])

    if (analysis.root_path / "docs").is_dir():
        layout_rows.append(["`docs/`", "Active", "Documentation"])

    builder.add_package_layout(
        layout_rows, note="**IMPORTANT**: Use the correct directory paths:\n"
    )

    # Key Components
    if analysis.key_components:
        component_rows = []
        for comp in analysis.key_components[:10]:
            desc = comp.docstring or f"{comp.category.title() if comp.category else 'Class'}"
            component_rows.append([comp.name, f"`{comp.file_path}:{comp.line_number}`", desc[:60]])

        builder.add_table(
            "## Key Components\n",
            ["Component", "Path", "Description"],
            component_rows,
            separator="|-----------|------|-------------|",
        )

    # Common Commands
    command_lines: List[str] = [
        "# Install with dev dependencies",
        'pip install -e ".[dev]"',
    ]
    if analysis.entry_points:
        command_lines.append("")
        command_lines.append("# Run the application")
        command_lines.extend(list(analysis.entry_points.keys())[:2])
    if analysis.cli_commands:
        command_lines.append("")
        command_lines.append("# Development")
        command_lines.extend(analysis.cli_commands)
    builder.add_command_section("## Common Commands\n", command_lines)

    # Architecture Notes
    if analysis.architecture_patterns:
        sections.append("## Architecture\n")
        for i, pattern in enumerate(analysis.architecture_patterns, 1):
            sections.append(f"{i}. {pattern}")
        sections.append("")

    # Package Structure
    if analysis.packages:
        sections.append("## Package Structure\n")
        for pkg_name, modules in sorted(analysis.packages.items()):
            if pkg_name == "root":
                continue
            class_count = sum(len(m.classes) for m in modules)
            sections.append(f"- **{pkg_name}/**: {len(modules)} modules, {class_count} classes")
        sections.append("")

    # Important Notes
    sections.append("## Important Notes\n")
    if analysis.deprecated_paths:
        sections.append(
            f"- **ALWAYS** use `{analysis.main_package}/` not `{analysis.deprecated_paths[0]}`"
        )
    sections.append("- Check component paths above for exact file:line references")

    if analysis.config_files:
        sections.append(
            "- Key config files: " + ", ".join(f"`{f}`" for f, _ in analysis.config_files[:4])
        )

    sections.append("")

    return builder.build()


def _generate_generic_victor_md(
    root_path: Optional[str] = None,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Generate init.md for non-Python projects using language-agnostic analysis."""
    context = gather_project_context(
        root_path, max_files=100, include_dirs=include_dirs, exclude_dirs=exclude_dirs
    )
    _root = Path(root_path).resolve() if root_path else Path.cwd()

    sections = []

    sections.append(f"# {VICTOR_CONTEXT_FILE}\n")
    sections.append(
        "This file provides guidance to Victor when working with code in this repository.\n"
    )

    sections.append("## Project Overview\n")
    if context["readme_content"]:
        paragraphs = context["readme_content"].split("\n\n")
        for para in paragraphs:
            stripped = para.strip()
            if stripped and not stripped.startswith(("#", "![", "<", "[!", "---", "```", "|")):
                sections.append(f"**{context['project_name']}**: {stripped[:300]}\n")
                break
        else:
            sections.append(f"**{context['project_name']}**: [Add project description here]\n")
    else:
        sections.append(f"**{context['project_name']}**: [Add project description here]\n")

    if context["detected_languages"]:
        sections.append(f"**Languages**: {', '.join(context['detected_languages'][:5])}\n")

    sections.append("## Package Layout\n")
    sections.append("| Path | Description |")
    sections.append("|------|-------------|")

    for dir_path in context["directory_structure"][:15]:
        if "/" not in dir_path.rstrip("/"):
            desc = _infer_directory_purpose(dir_path.rstrip("/"))
            sections.append(f"| `{dir_path}` | {desc} |")

    sections.append("")

    if context["source_files"]:
        sections.append("## Key Files\n")
        for f in context["source_files"][:15]:
            sections.append(f"- `{f}`")
        sections.append("")

    sections.append("## Common Commands\n")
    sections.append("```bash")

    if "pyproject.toml" in context["config_files"] or "requirements.txt" in list(
        context["config_files"]
    ):
        sections.append("# Python project")
        sections.append("pip install -r requirements.txt")
        sections.append("python main.py")
    elif "package.json" in context["config_files"]:
        sections.append("# Node.js project")
        sections.append("npm install")
        sections.append("npm start")
    elif "Cargo.toml" in context["config_files"]:
        sections.append("# Rust project")
        sections.append("cargo build")
        sections.append("cargo run")
    elif "go.mod" in context["config_files"]:
        sections.append("# Go project")
        sections.append("go build")
        sections.append("go run .")
    else:
        sections.append("# Add your build/run commands here")

    sections.append("```\n")

    if context["config_files"]:
        sections.append("## Configuration\n")
        sections.append(
            "Key config files: " + ", ".join(f"`{f}`" for f in context["config_files"][:5])
        )
        sections.append("")

    sections.append("## Important Notes\n")
    sections.append("- Review and customize this file based on your project specifics")
    sections.append("- Use `/init --deep` for LLM-powered comprehensive analysis")
    sections.append("")

    return "\n".join(sections)


def _infer_directory_purpose(dirname: str) -> str:
    """Infer the purpose of a directory from its name."""
    purposes = {
        "src": "Source code",
        "lib": "Library code",
        "app": "Application code",
        "api": "API endpoints",
        "components": "UI components",
        "pages": "Page components",
        "views": "View templates",
        "models": "Data models",
        "utils": "Utility functions",
        "helpers": "Helper functions",
        "config": "Configuration",
        "configs": "Configuration",
        "tests": "Test files",
        "test": "Test files",
        "spec": "Test specifications",
        "docs": "Documentation",
        "public": "Public/static assets",
        "static": "Static files",
        "assets": "Asset files",
        "scripts": "Script files",
        "bin": "Executable scripts",
        "data": "Data files",
        "migrations": "Database migrations",
        "styles": "Stylesheets",
        "css": "CSS styles",
    }
    return purposes.get(dirname.lower(), "Project files")


def create_context_symlinks(
    root_path: Optional[str] = None,
    source_file: Optional[str] = None,
    aliases: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Create symlinks from .victor/init.md to other context file names."""
    root = Path(root_path).resolve() if root_path else Path.cwd()
    if source_file is None:
        paths = get_project_paths(root)
        source = paths.project_context_file
        source_file = str(source.relative_to(root))
    else:
        source = root / source_file
    results: Dict[str, str] = {}

    if not source.exists():
        logger.warning(f"Source file {source} does not exist")
        return {"error": f"Source file {source_file} not found"}

    target_aliases = aliases if aliases is not None else list(CONTEXT_FILE_ALIASES.keys())

    for alias in target_aliases:
        target = root / alias
        try:
            if target.exists():
                if target.is_symlink():
                    if target.resolve() == source.resolve():
                        results[alias] = "exists"
                    else:
                        results[alias] = "exists_different"
                else:
                    results[alias] = "exists_file"
            else:
                target.symlink_to(source_file)
                results[alias] = "created"
                logger.info(f"Created symlink: {alias} -> {source_file}")
        except OSError as e:
            results[alias] = f"failed: {e}"
            logger.warning(f"Failed to create symlink {alias}: {e}")

    return results


def remove_context_symlinks(
    root_path: Optional[str] = None,
    aliases: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Remove symlinks to context files."""
    root = Path(root_path).resolve() if root_path else Path.cwd()
    results: Dict[str, str] = {}

    target_aliases = aliases if aliases is not None else list(CONTEXT_FILE_ALIASES.keys())

    for alias in target_aliases:
        target = root / alias
        try:
            if not target.exists() and not target.is_symlink():
                results[alias] = "not_found"
            elif target.is_symlink():
                target.unlink()
                results[alias] = "removed"
                logger.info(f"Removed symlink: {alias}")
            else:
                results[alias] = "not_symlink"
        except OSError as e:
            results[alias] = f"failed: {e}"

    return results


def _extract_readme_description(root: Path) -> str:
    """Extract project description from README."""
    readme_files = ["README.md", "README.rst", "README.txt"]

    for readme in readme_files:
        readme_path = root / readme
        if readme_path.exists():
            try:
                content = readme_path.read_text(encoding="utf-8")
                paragraphs = content.split("\n\n")

                for para in paragraphs:
                    stripped = para.strip()
                    if not stripped:
                        continue
                    if stripped.startswith(("#", "![", "<", "[!", "---", "```", "|")):
                        continue
                    if stripped.startswith("[") and stripped.endswith(")"):
                        continue
                    if stripped.startswith("*") and stripped.endswith("*") and "\n" not in stripped:
                        continue
                    result = stripped[:300]
                    result = result.strip("*_")
                    result = result.replace("**", "").replace("__", "")
                    return result
            except Exception:
                pass

    return ""


def gather_project_context(
    root_path: Optional[str] = None,
    max_files: int = 50,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Gather project context for LLM analysis (works with any language)."""
    root = Path(root_path).resolve() if root_path else Path.cwd()

    skip_dirs = {
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
        "target",
        ".next",
        ".nuxt",
        "coverage",
        ".cache",
    }
    if exclude_dirs:
        skip_dirs.update(exclude_dirs)

    project_indicators = {
        "pyproject.toml": "Python (modern)",
        "setup.py": "Python (legacy)",
        "package.json": "JavaScript/TypeScript",
        "Cargo.toml": "Rust",
        "go.mod": "Go",
        "pom.xml": "Java (Maven)",
        "build.gradle": "Java/Kotlin (Gradle)",
        "Gemfile": "Ruby",
        "composer.json": "PHP",
        "mix.exs": "Elixir",
        "CMakeLists.txt": "C/C++ (CMake)",
        "Makefile": "Make-based",
        "pubspec.yaml": "Dart/Flutter",
        "Package.swift": "Swift",
        ".csproj": "C# (.NET)",
    }

    source_extensions = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript React",
        ".jsx": "JavaScript React",
        ".rs": "Rust",
        ".go": "Go",
        ".java": "Java",
        ".kt": "Kotlin",
        ".rb": "Ruby",
        ".php": "PHP",
        ".ex": "Elixir",
        ".exs": "Elixir",
        ".c": "C",
        ".cpp": "C++",
        ".h": "C/C++ Header",
        ".cs": "C#",
        ".swift": "Swift",
        ".dart": "Dart",
        ".vue": "Vue",
        ".svelte": "Svelte",
    }

    context = {
        "project_name": root.name,
        "root_path": str(root),
        "detected_languages": [],
        "config_files": [],
        "directory_structure": [],
        "source_files": [],
        "readme_content": "",
        "main_config_content": "",
    }

    for config_file, lang in project_indicators.items():
        if (root / config_file).exists():
            context["detected_languages"].append(lang)
            context["config_files"].append(config_file)

    for readme in ["README.md", "README.rst", "README.txt", "readme.md"]:
        readme_path = root / readme
        if readme_path.exists():
            try:
                context["readme_content"] = readme_path.read_text(encoding="utf-8")[:2000]
            except Exception:
                pass
            break

    main_configs = ["pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
    for config in main_configs:
        config_path = root / config
        if config_path.exists():
            try:
                context["main_config_content"] = config_path.read_text(encoding="utf-8")[:3000]
            except Exception:
                pass
            break

    def walk_dirs(path: Path, depth: int = 0, max_depth: int = 2) -> List[str]:
        dirs = []
        if depth > max_depth:
            return dirs
        try:
            for item in sorted(path.iterdir()):
                if item.name.startswith(".") or item.name in skip_dirs:
                    continue
                if item.is_dir():
                    rel_path = str(item.relative_to(root))
                    dirs.append(rel_path + "/")
                    dirs.extend(walk_dirs(item, depth + 1, max_depth))
        except PermissionError:
            pass
        return dirs

    if include_dirs:
        all_found_dirs = []
        for d in include_dirs:
            dir_path = root / d
            if dir_path.is_dir():
                all_found_dirs.extend(walk_dirs(dir_path))
        context["directory_structure"] = all_found_dirs[:100]
    else:
        context["directory_structure"] = walk_dirs(root)[:100]

    file_count = 0
    lang_counts: Dict[str, int] = {}

    search_paths = [root / d for d in include_dirs] if include_dirs else [root]

    for search_path in search_paths:
        if not search_path.is_dir():
            continue
        for item in search_path.rglob("*"):
            if file_count >= max_files:
                break
            if any(skip in item.parts for skip in skip_dirs):
                continue
            if item.is_file() and item.suffix in source_extensions:
                rel_path = str(item.relative_to(root))
                context["source_files"].append(rel_path)
                lang = source_extensions[item.suffix]
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                file_count += 1
        if file_count >= max_files:
            break

    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        if lang not in context["detected_languages"]:
            context["detected_languages"].append(f"{lang} ({count} files)")

    context["key_files_content"] = {}

    key_file_candidates = []
    main_files = [
        f
        for f in context["source_files"]
        if "main" in f or "app" in f or "server" in f or "index" in f
    ]
    key_file_candidates.extend(main_files)

    large_files = sorted(
        context["source_files"], key=lambda f: (root / f).stat().st_size, reverse=True
    )
    for f in large_files:
        if f not in key_file_candidates:
            key_file_candidates.append(f)

    key_files_to_read = key_file_candidates[:5]

    for file_path in key_files_to_read:
        try:
            content = (root / file_path).read_text(encoding="utf-8")
            context["key_files_content"][file_path] = content[:8192]
        except Exception:
            pass

    return context


def build_llm_prompt_for_victor_md(context: Dict[str, any]) -> str:
    """Build the prompt for LLM to generate project context file."""

    prompt_header = f"""You are an expert software architect tasked with creating a high-level "user manual" for an AI coding assistant named Victor.
This manual, named {VICTOR_CONTEXT_FILE}, will help Victor understand the project's structure, purpose, and conventions.
Your analysis must be comprehensive, distilling the provided information into a clear and actionable guide.

Analyze the following project data and generate the {VICTOR_CONTEXT_FILE} file.

**Output Rules:**
1.  **Start with the Header**: The response MUST begin with `# {VICTOR_CONTEXT_FILE}`.
2.  **Use Markdown**: Format the entire output in clean, readable Markdown. Use tables for structured data.
3.  **Be Factual**: Base your analysis exclusively on the provided context. Do not infer or add information not present in the data.
4.  **Be Concise**: Provide high-level summaries. Focus on the "what" and "why," not implementation details.
5.  **Follow the Structure**: Generate all of the requested sections.

---
"""

    dynamic_context = f"""
**Project Name**: {context['project_name']}
**Detected Languages**: {', '.join(context['detected_languages']) or 'Unknown'}

**Configuration Files**:
{chr(10).join('- ' + f for f in context['config_files']) or 'None detected'}

**Directory Structure Overview**:
```
{chr(10).join(context['directory_structure'][:50]) or 'Unable to determine'}
```

**Sample of Source Files**:
```
{chr(10).join(context['source_files'][:30]) or 'No source files found'}
```
"""

    if context.get("key_files_content"):
        dynamic_context += "\n**Content of Key Files**:\n"
        for file_path, content in context["key_files_content"].items():
            dynamic_context += f"--- `{file_path}` ---\n```\n{content}\n```\n\n"

    prompt_footer = f"""
---

**Generation Task**:

Generate the full content for the `{VICTOR_CONTEXT_FILE}` file, adhering to all rules above. Create the following sections:

1.  `## Project Overview`
    -   Write a one-paragraph summary of the project's purpose, based on the README and file structure.

2.  `## Package Layout`
    -   Create a Markdown table with columns: `| Path | Status | Description |`.
    -   List the most important top-level directories.
    -   Infer the purpose of each directory (e.g., source code, tests, docs). Mark the main source directory as `**ACTIVE**`.

3.  `## Key Components`
    -   Identify 5-7 key files or classes from the provided context.
    -   Create a Markdown table with columns: `| Component | Path | Description |`.
    -   Provide a one-sentence description for each component.

4.  `## Common Commands`
    -   Based on the configuration files (e.g., `package.json`, `pyproject.toml`), list the essential commands for building, testing, and running the project inside a `bash` code block.

5.  `## Architecture Notes`
    -   From the file names and structure, infer 2-3 high-level architectural patterns. (e.g., "Provider Pattern for multiple LLMs", "REST API with FastAPI", "CLI application using Typer").

6.  `## Important Notes`
    -   Add 2-3 bullet points for an AI assistant to remember, such as "Always use `victor/` for core source code" or "Check `pyproject.toml` for dependencies".

Remember, output ONLY the generated `{VICTOR_CONTEXT_FILE}` content, starting with the `# {VICTOR_CONTEXT_FILE}` header.
"""

    return prompt_header + dynamic_context + prompt_footer


async def generate_victor_md_with_llm(
    provider,
    model: str,
    root_path: Optional[str] = None,
    max_files: int = 50,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Generate project context file using an LLM provider."""
    from victor.providers.base import Message

    context = gather_project_context(
        root_path, max_files, include_dirs=include_dirs, exclude_dirs=exclude_dirs
    )

    prompt = build_llm_prompt_for_victor_md(context)

    messages = [Message(role="user", content=prompt)]

    expected_header = f"# {VICTOR_CONTEXT_FILE}"

    try:
        response = await provider.chat(messages, model=model)
        content = response.content.strip()

        if not content.startswith(expected_header):
            content = f"{expected_header}\\n\\n" + content

        return content
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return generate_smart_victor_md(
            root_path, include_dirs=include_dirs, exclude_dirs=exclude_dirs
        )


def _infer_python_requires(root: Path) -> Optional[str]:
    """Read requires-python from pyproject.toml if present."""
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        content = pyproject.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"requires-python\\s*=\\s*['\\\"]([^'\\\"]+)['\\\"]", content)
        if match:
            return match.group(1)
    except Exception:
        return None
    return None


def _infer_commands(root: Path) -> List[str]:
    """Infer project commands from Makefile, package.json, and common conventions."""
    commands: List[str] = []

    makefile = root / "Makefile"
    if makefile.exists():
        try:
            make_text = makefile.read_text(encoding="utf-8", errors="ignore")
            targets = set(re.findall(r"^(\\w+):", make_text, re.MULTILINE))
            preferred = [
                ("install-dev", "make install-dev"),
                ("install", "make install"),
                ("lint", "make lint"),
                ("format", "make format"),
                ("test", "make test"),
                ("test-all", "make test-all"),
                ("serve", "make serve"),
                ("build", "make build"),
                ("docker", "make docker"),
            ]
            for target, cmd in preferred:
                if target in targets:
                    commands.append(cmd)
        except Exception:
            pass

    pkg = root / "package.json"
    if pkg.exists():
        try:
            import json

            data = json.loads(pkg.read_text(encoding="utf-8", errors="ignore"))
            scripts = data.get("scripts", {}) or {}
            for key in ("dev", "start", "build", "test", "lint"):
                if key in scripts:
                    commands.append(f"npm run {key}")
            commands.append("npm install")
        except Exception:
            pass

    commands.append('pip install -e ".[dev]"')
    commands.append("pytest")

    if (root / "docker-compose.yml").exists():
        commands.append("docker-compose up -d")

    if (root / "web" / "server").exists():
        commands.append("uvicorn web.server.main:app --reload")
    if (root / "web" / "ui").exists():
        commands.append("cd web/ui && npm install && npm run dev")

    seen: Set[str] = set()
    deduped: List[str] = []
    for cmd in commands:
        if cmd not in seen:
            deduped.append(cmd)
            seen.add(cmd)
    return deduped


def _infer_env_vars(root: Path) -> List[str]:
    """Infer env vars from .env/.env.example (uppercase keys)."""
    env_files = [root / ".env", root / ".env.example"]
    vars_found: List[str] = []
    seen: Set[str] = set()
    for env_file in env_files:
        if not env_file.exists():
            continue
        try:
            for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key = line.split("=", 1)[0].strip()
                if key.isupper() and key not in seen:
                    vars_found.append(key)
                    seen.add(key)
        except Exception:
            continue
    return vars_found[:10]


def _build_quick_start(commands: List[str]) -> List[str]:
    """Pick a concise quick-start command set."""
    if not commands:
        return []
    quick: List[str] = []
    priorities = [
        ("install", ("install", "pip install", "npm install", "make install")),
        ("lint", ("lint", "ruff", "eslint")),
        ("test", ("test", "pytest", "npm test", "make test")),
        ("serve", ("serve", "uvicorn", "run dev", "npm run dev")),
    ]
    lower_cmds = [(c.lower(), c) for c in commands]
    for _label, needles in priorities:
        for lc, orig in lower_cmds:
            if any(n in lc for n in needles):
                if orig not in quick:
                    quick.append(orig)
                break
    for cmd in commands:
        if len(quick) >= 3:
            break
        if cmd not in quick:
            quick.append(cmd)
    return quick[:4]


def _find_config_files(root: Path) -> List[str]:
    """Find common config files (json/yaml/toml) while skipping vendor/venv/build artifacts."""
    exts = {".json", ".yaml", ".yml", ".toml"}
    results: List[str] = []
    extra_skip = {
        "venv",
        "env",
        "node_modules",
        "build",
        "dist",
        "out",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
    skip_parts = {"htmlcov", "htmlcov_lang", "coverage", "__pycache__", "egg-info"}

    for path in root.rglob("*"):
        if path.suffix not in exts or not path.is_file():
            continue
        if should_ignore_path(path, skip_dirs=DEFAULT_SKIP_DIRS, extra_skip_dirs=extra_skip):
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        rel = str(path.relative_to(root))
        results.append(rel)
        if len(results) >= 40:
            break

    results.sort()
    return results[:15]


def _find_docs_files(root: Path) -> List[str]:
    """Find markdown/adoc docs (top-N) while skipping vendor/venv/build artifacts."""
    results: List[tuple[int, str]] = []
    extra_skip = {
        "venv",
        "env",
        "node_modules",
        "build",
        "dist",
        "out",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
    skip_parts = {"htmlcov", "htmlcov_lang", "coverage", "__pycache__", "egg-info"}

    for path in root.rglob("*.md"):
        if should_ignore_path(path, skip_dirs=DEFAULT_SKIP_DIRS, extra_skip_dirs=extra_skip):
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        try:
            size = path.stat().st_size
        except Exception:
            size = 0
        rel = str(path.relative_to(root))
        results.append((size, rel))
        if len(results) >= 80:
            break
    results.sort(reverse=True)
    return [rel for _size, rel in results[:12]]


async def generate_victor_md_from_index(
    root_path: Optional[str] = None,
    force: bool = False,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Legacy entry point retained for compatibility."""
    return generate_smart_victor_md(
        root_path=root_path, include_dirs=include_dirs, exclude_dirs=exclude_dirs
    )


async def generate_enhanced_init_md(
    root_path: Optional[str] = None,
    use_llm: bool = False,
    include_conversations: bool = True,
    on_progress: Optional[callable] = None,
    force: bool = False,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    auto_index: bool = True,
) -> str:
    """Generate init.md using symbol index, conversation insights, and optional LLM.

    Pipeline: Index -> Learn (optional) -> LLM enhance (optional)
    """
    import time

    from victor.providers.base import Message
    from .query import (
        extract_conversation_insights,
        extract_graph_insights,
    )

    step_times: dict = {}
    step_start: float = 0

    def progress(stage: str, msg: str, complete: bool = False):
        nonlocal step_start
        if complete and step_start > 0:
            elapsed = time.time() - step_start
            step_times[stage] = elapsed
            if on_progress:
                on_progress(stage, f"\u2713 {msg} ({elapsed:.1f}s)")
        else:
            step_start = time.time()
            if on_progress:
                on_progress(stage, msg)

    # Step 1: Index
    progress("index", "Building symbol index...")
    base_content = await generate_victor_md_from_index(
        root_path, force=force, include_dirs=include_dirs, exclude_dirs=exclude_dirs
    )
    progress("index", "Symbol index built", complete=True)

    # Step 2: Learn
    if include_conversations:
        progress("learn", "Extracting conversation insights...")
        insights = await extract_conversation_insights(root_path)
        sessions = insights.get("session_count", 0)
        progress("learn", f"Insights extracted ({sessions} sessions)", complete=True)

        if sessions > 0:
            enhancements = ["\n## Learned from Conversations\n"]
            enhancements.append(
                f"*Based on {insights['session_count']} sessions, {insights['message_count']} messages*\n"
            )

            if insights.get("hot_files"):
                enhancements.append("### Frequently Referenced Files\n")
                for file_path, count in insights["hot_files"][:8]:
                    enhancements.append(f"- `{file_path}` ({count} references)")
                enhancements.append("")

            if insights.get("common_topics"):
                topics = [t[0] for t in insights["common_topics"][:6]]
                enhancements.append("### Common Topics\n")
                enhancements.append(f"Keywords: {', '.join(topics)}\n")

            if insights.get("faq"):
                enhancements.append("### Frequently Asked Questions\n")
                for faq in insights["faq"][:3]:
                    q = (
                        faq["question"][:100] + "..."
                        if len(faq["question"]) > 100
                        else faq["question"]
                    )
                    enhancements.append(f"- {q}")
                enhancements.append("")

            if "## Important Notes" in base_content:
                parts = base_content.split("## Important Notes")
                base_content = (
                    parts[0] + "\n".join(enhancements) + "\n## Important Notes" + parts[1]
                )
            else:
                base_content += "\n" + "\n".join(enhancements)

    # Step 2.5: Graph
    progress("graph", "Analyzing code graph...")
    graph_insights = await extract_graph_insights(root_path)

    if not graph_insights.get("has_graph") and auto_index:
        progress("graph", "No graph data - building index automatically...")
        try:
            from victor.config.settings import load_settings
            from victor.tools.code_search_tool import _get_or_build_index

            root = Path(root_path).resolve() if root_path else Path.cwd()
            settings = load_settings()
            index, rebuilt = await _get_or_build_index(root, settings, force_reindex=force)
            status_msg = "Index built" if rebuilt else "Index loaded"
            if index.graph_store:
                try:
                    stats = await index.graph_store.stats()
                    node_count = stats.get("nodes", 0)
                    edge_count = stats.get("edges", 0)
                    if node_count or edge_count:
                        status_msg += f" ({node_count} symbols, {edge_count} edges)"
                except Exception:
                    pass
            progress("graph", f"{status_msg}, analyzing...")
            graph_insights = await extract_graph_insights(root_path)
        except Exception as e:
            logger.warning(f"Auto-indexing failed: {e}")
            progress("graph", f"Auto-indexing failed: {e}", complete=True)

    if graph_insights.get("has_graph"):
        progress(
            "graph",
            f"Graph analyzed ({graph_insights['stats'].get('total_nodes', 0)} nodes)",
            complete=True,
        )

        graph_section = ["\n## Code Graph Insights\n"]
        graph_section.append(
            f"*{graph_insights['stats'].get('total_nodes', 0)} symbols, {graph_insights['stats'].get('total_edges', 0)} relationships*\n"
        )

        if graph_insights.get("patterns"):
            graph_section.append("### Detected Design Patterns\n")
            for p in graph_insights["patterns"][:5]:
                details = p.get("details", {})
                if p["pattern"] == "provider_strategy":
                    impls = details.get("implementations", [])
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('base_class', '')}` with {len(impls)} implementations"
                    )
                elif p["pattern"] == "facade":
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('class', '')}` ({details.get('incoming_calls', 0)} callers \u2192 {details.get('outgoing_calls', 0)} delegates)"
                    )
                elif p["pattern"] == "composition":
                    composed = details.get("composed_of", [])
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('class', '')}` composed of {len(composed)} components"
                    )
                elif p["pattern"] == "factory":
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('class', '')}` creates {details.get('creates', 0)} types"
                    )
                else:
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('class', details.get('base_class', ''))}`"
                    )
            graph_section.append("")

        if graph_insights.get("important_symbols"):
            graph_section.append("### Most Important Symbols (PageRank)\n")
            graph_section.append("| Symbol | Type | Connections |")
            graph_section.append("|--------|------|-------------|")
            for sym in graph_insights["important_symbols"][:6]:
                conns = f"\u2193{sym['in_degree']} \u2191{sym['out_degree']}"
                line_ref = f":{sym['line']}" if sym.get("line") else ""
                location = f"({sym['file']}{line_ref})"
                graph_section.append(f"| `{sym['name']}` {location} | {sym['type']} | {conns} |")
            graph_section.append("")

        if graph_insights.get("hub_classes"):
            graph_section.append("### Hub Classes (High Connectivity)\n")
            for hub in graph_insights["hub_classes"]:
                line_ref = f":{hub['line']}" if hub.get("line") else ""
                location = f"({hub['file']}{line_ref})"
                graph_section.append(f"- `{hub['name']}` {location} - {hub['degree']} connections")
            graph_section.append("")

        if graph_insights.get("important_modules"):
            graph_section.append("### Key Modules (Architecture)\n")
            graph_section.append("| Module | Role | Connections |")
            graph_section.append("|--------|------|-------------|")
            for mod in graph_insights["important_modules"][:6]:
                role_emoji = {
                    "service": "\U0001f527",
                    "orchestrator": "\U0001f39b\ufe0f",
                    "intermediary": "\u2194\ufe0f",
                    "leaf": "\U0001f343",
                    "entry": "\U0001f680",
                    "peripheral": "\U0001f4e6",
                }.get(mod["role"], "")
                conns = f"\u2193{mod['in_degree']} \u2191{mod['out_degree']}"
                graph_section.append(
                    f"| `{mod['module']}` | {role_emoji} {mod['role']} | {conns} |"
                )
            graph_section.append("")

        if graph_insights.get("module_coupling"):
            graph_section.append("### Coupling Hotspots\n")
            for coupling in graph_insights["module_coupling"][:3]:
                pattern_desc = {
                    "hub": "\u26a0\ufe0f High fan-in AND fan-out",
                    "high_fan_in": "Many callers",
                    "high_fan_out": "Calls many modules",
                }.get(coupling["pattern"], coupling["pattern"])
                graph_section.append(
                    f"- `{coupling['module']}` - {pattern_desc} "
                    f"(\u2193{coupling['in_degree']} \u2191{coupling['out_degree']})"
                )
            graph_section.append("")

        if "## Important Notes" in base_content:
            parts = base_content.split("## Important Notes")
            base_content = parts[0] + "\n".join(graph_section) + "\n## Important Notes" + parts[1]
        else:
            base_content += "\n" + "\n".join(graph_section)
    else:
        if auto_index:
            progress(
                "graph", "Graph indexing incomplete (retry with 'victor index')", complete=True
            )
        else:
            progress(
                "graph", "No graph data (run 'victor index' or enable auto_index)", complete=True
            )

    # Step 3: Deep - Use LLM to enhance content
    if not use_llm:
        return base_content

    progress("deep", "Enhancing with LLM analysis...")

    try:
        from victor.config.settings import Settings
        from victor.providers.registry import ProviderRegistry

        settings = Settings()
        provider_name = settings.default_provider
        model_name = settings.default_model
        provider_settings = settings.get_provider_settings(provider_name)
        provider = ProviderRegistry.create(provider_name, **provider_settings)

        if not provider:
            logger.warning(f"Could not get provider {provider_name}, skipping LLM")
            return base_content

        enhance_prompt = f"""You are an expert software architect reviewing a project documentation file.

Below is an auto-generated init.md file for a codebase. Your task is to:
1. Improve the descriptions to be more specific and actionable
2. Identify any key architectural patterns that were missed
3. Add meaningful relationships between components
4. Ensure the most important components are highlighted
5. Keep the same markdown structure but enhance the content quality

IMPORTANT RULES:
- Keep all existing sections and their structure
- Do NOT add generic advice - only project-specific insights
- Do NOT remove any existing content, only enhance it
- Keep the file concise - quality over quantity
- Focus on what makes this project unique

Here is the current init.md content:

```markdown
{base_content}
```

Return ONLY the enhanced markdown content, no explanations."""

        messages = [Message(role="user", content=enhance_prompt)]
        response = await provider.chat(messages, model=model_name)
        enhanced = response.content.strip()
        progress("deep", "LLM enhancement complete", complete=True)

        if enhanced.startswith("#") or enhanced.startswith("```"):
            if enhanced.startswith("```"):
                lines = enhanced.split("\n")
                lines = lines[1:] if lines[0].startswith("```") else lines
                lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
                enhanced = "\n".join(lines)
            await provider.close()
            return enhanced

        logger.warning("LLM response doesn't look like valid markdown")
        await provider.close()
        return base_content

    except Exception as e:
        progress("deep", f"LLM failed: {e}", complete=True)
        logger.warning(f"LLM enhancement failed: {e}, using base content")
        return base_content
