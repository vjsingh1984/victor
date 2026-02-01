#!/usr/bin/env python3
"""Split large documentation files into smaller, focused documents."""

import re
from pathlib import Path


def split_component_reference():
    """Split COMPONENT_REFERENCE.md into 3 files."""
    source = Path("docs/architecture/COMPONENT_REFERENCE.md")
    if not source.exists():
        print(f"Source file {source} not found")
        return

    content = source.read_text()

    # Define output files with their line ranges (based on section headers)
    outputs = {
        "docs/reference/internals/components.md": {
            "title": "Component Overview",
            "description": "Brief overview of key Victor AI components",
            "sections": ["Overview", "Core Components"],
        },
        "docs/guides/component-usage.md": {
            "title": "Component Usage Guide",
            "description": "How to use Victor AI components and coordinators",
            "sections": ["Coordinators", "Adapters", "Mixins", "Component Interactions", "Extension Points"],
        },
        "docs/reference/api/internal.md": {
            "title": "Internal API Reference",
            "description": "API documentation for internal systems",
            "sections": ["Provider System", "Tool System", "Event System", "Workflow System", "Framework Components"],
        },
    }

    # Split by major sections
    lines = content.split("\n")
    section_starts = {}
    current_section = None

    for i, line in enumerate(lines):
        if line.startswith("## ") and not line.startswith("### "):
            section_name = line[3:].strip()
            section_starts[section_name] = i

    # Generate output files
    for output_path, config in outputs.items():
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        output_lines = []
        for section in config["sections"]:
            if section in section_starts:
                start = section_starts[section]
                # Find end of this section (start of next section or end of file)
                section_indices = sorted([v for k, v in section_starts.items()])
                idx = section_indices.index(start)
                end = section_indices[idx + 1] if idx + 1 < len(section_indices) else len(lines)

                # Add section content
                section_content = lines[start:end]
                output_lines.extend(section_content)
                output_lines.append("")  # Blank line between sections

        # Write output file with header
        header = f"""# {config['title']}

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: {config['description']}

---

"""
        out.write_text(header + "\n".join(output_lines))
        print(f"Created {output_path}")


def split_best_practices():
    """Split BEST_PRACTICES.md into 3 files."""
    source = Path("docs/architecture/BEST_PRACTICES.md")
    if not source.exists():
        print(f"Source file {source} not found")
        return

    content = source.read_text()

    outputs = {
        "docs/architecture/best-practices/protocols.md": {
            "title": "Protocol Best Practices",
            "description": "Best practices for using protocols in Victor AI",
            "sections": ["Using Protocols"],
        },
        "docs/architecture/best-practices/di-events.md": {
            "title": "Dependency Injection and Events",
            "description": "Best practices for DI and event-driven architecture",
            "sections": ["Using Dependency Injection", "Using Event-Driven Architecture"],
        },
        "docs/architecture/best-practices/anti-patterns.md": {
            "title": "Anti-Patterns to Avoid",
            "description": "Common anti-patterns and how to avoid them",
            "sections": ["Anti-Patterns to Avoid"],
        },
    }

    # Split by major sections
    lines = content.split("\n")
    section_starts = {}
    current_section = None

    for i, line in enumerate(lines):
        if line.startswith("## ") and not line.startswith("### "):
            section_name = line[3:].strip()
            section_starts[section_name] = i

    # Generate output files
    for output_path, config in outputs.items():
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        output_lines = []
        for section in config["sections"]:
            if section in section_starts:
                start = section_starts[section]
                # Find end of this section
                section_indices = sorted([v for k, v in section_starts.items()])
                idx = section_indices.index(start)
                end = section_indices[idx + 1] if idx + 1 < len(section_indices) else len(lines)

                section_content = lines[start:end]
                output_lines.extend(section_content)
                output_lines.append("")

        header = f"""# {config['title']}

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: {config['description']}

---

"""
        out.write_text(header + "\n".join(output_lines))
        print(f"Created {output_path}")


def split_design_patterns():
    """Split DESIGN_PATTERNS.md into 3 files."""
    source = Path("docs/architecture/DESIGN_PATTERNS.md")
    if not source.exists():
        print(f"Source file {source} not found")
        return

    content = source.read_text()

    outputs = {
        "docs/architecture/patterns/creational.md": {
            "title": "Creational Design Patterns",
            "description": "Factory, Builder, and Singleton patterns in Victor AI",
            "sections": ["Overview", "Creational Patterns"],
        },
        "docs/architecture/patterns/structural-behavioral.md": {
            "title": "Structural and Behavioral Patterns",
            "description": "Adapter, Facade, Strategy, and Observer patterns",
            "sections": ["Structural Patterns", "Behavioral Patterns"],
        },
        "docs/architecture/patterns/architecture.md": {
            "title": "Architecture Patterns",
            "description": "Higher-level architecture patterns and selection guides",
            "sections": ["Architecture Patterns", "Pattern Selection Guide"],
        },
    }

    # Split by major sections
    lines = content.split("\n")
    section_starts = {}

    for i, line in enumerate(lines):
        if line.startswith("## ") and not line.startswith("### "):
            section_name = line[3:].strip()
            section_starts[section_name] = i

    # Generate output files
    for output_path, config in outputs.items():
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        output_lines = []
        for section in config["sections"]:
            if section in section_starts:
                start = section_starts[section]
                section_indices = sorted([v for k, v in section_starts.items()])
                idx = section_indices.index(start)
                end = section_indices[idx + 1] if idx + 1 < len(section_indices) else len(lines)

                section_content = lines[start:end]
                output_lines.extend(section_content)
                output_lines.append("")

        header = f"""# {config['title']}

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: {config['description']}

---

"""
        out.write_text(header + "\n".join(output_lines))
        print(f"Created {output_path}")


if __name__ == "__main__":
    print("Splitting COMPONENT_REFERENCE.md...")
    split_component_reference()

    print("\nSplitting BEST_PRACTICES.md...")
    split_best_practices()

    print("\nSplitting DESIGN_PATTERNS.md...")
    split_design_patterns()

    print("\nDone!")
