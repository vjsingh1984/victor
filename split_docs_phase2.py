#!/usr/bin/env python3
"""Further split large documentation files to meet 800-line limit."""

import re
from pathlib import Path


def split_component_usage_further():
    """Split component-usage.md - separate coordinators into its own file."""
    source = Path("docs/guides/component-usage.md")
    if not source.exists():
        print(f"Source file {source} not found")
        return

    content = source.read_text()
    lines = content.split("\n")

    # Find section starts
    section_starts = {}
    for i, line in enumerate(lines):
        if line.startswith("## ") and not line.startswith("### "):
            section_name = line[3:].strip()
            section_starts[section_name] = i

    # Create coordinators guide
    if "Coordinators" in section_starts and "Adapters" in section_starts:
        start = section_starts["Coordinators"]
        end = section_starts["Adapters"]

        coordinator_lines = lines[:10] + lines[start:end]  # Include header
        out = Path("docs/guides/coordinators.md")
        header = f"""# Coordinator Usage Guide

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: How to use Victor AI coordinators

This guide covers the 20 specialized coordinators in Victor AI's two-layer architecture.

---

"""
        out.write_text(header + "\n".join(coordinator_lines))
        print(f"Created {out}")

        # Update component-usage.md to remove coordinators and add link
        new_lines = (
            lines[:10] +
            [
                "",
                "## Coordinator Reference",
                "",
                "For detailed information about coordinators, see [Coordinator Usage Guide](coordinators.md).",
                "",
                "Victor AI uses a two-layer coordinator architecture:",
                "",
                "- **Application Layer**: Victor-specific business logic (ChatCoordinator, ToolCoordinator, etc.)",
                "- **Framework Layer**: Domain-agnostic infrastructure (YAMLWorkflowCoordinator, GraphExecutionCoordinator, etc.)",
                "",
                "For coordinator development patterns, see [Architecture Patterns](../architecture/patterns/architecture.md).",
                ""
            ] +
            lines[end:]
        )

        source.write_text("\n".join(new_lines))
        print(f"Updated {source}")


def split_structural_behavioral():
    """Split structural-behavioral.md into separate files."""
    source = Path("docs/architecture/patterns/structural-behavioral.md")
    if not source.exists():
        print(f"Source file {source} not found")
        return

    content = source.read_text()
    lines = content.split("\n")

    # Find section starts
    section_starts = {}
    for i, line in enumerate(lines):
        if line.startswith("## ") and not line.startswith("### "):
            section_name = line[3:].strip()
            section_name = re.sub(r'\{#.*?\}', '', section_name).strip()  # Remove anchors
            section_starts[section_name] = i

    # Create structural patterns file
    if "Structural Patterns" in section_starts and "Behavioral Patterns" in section_starts:
        start = section_starts["Structural Patterns"]
        end = section_starts["Behavioral Patterns"]

        structural_lines = lines[:10] + lines[start:end]
        out = Path("docs/architecture/patterns/structural.md")
        header = f"""# Structural Design Patterns

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: Structural patterns in Victor AI (Adapter, Facade, Proxy, etc.)

---

"""
        out.write_text(header + "\n".join(structural_lines))
        print(f"Created {out}")

        # Create behavioral patterns file
        behavioral_lines = lines[:10] + lines[end:]
        out2 = Path("docs/architecture/patterns/behavioral.md")
        header2 = f"""# Behavioral Design Patterns

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: Behavioral patterns in Victor AI (Strategy, Observer, Command, etc.)

---

"""
        out2.write_text(header2 + "\n".join(behavioral_lines))
        print(f"Created {out2}")

        # Remove the old combined file
        source.unlink()
        print(f"Removed {source}")


if __name__ == "__main__":
    print("Further splitting component-usage.md...")
    split_component_usage_further()

    print("\nSplitting structural-behavioral.md...")
    split_structural_behavioral()

    print("\nDone!")
