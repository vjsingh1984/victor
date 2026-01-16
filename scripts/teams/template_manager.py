#!/usr/bin/env python
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

"""Team template manager CLI.

Command-line interface for managing team templates. Provides commands
for listing, showing, validating, creating, and applying templates.

Example:
    # List all templates
    python -m scripts.teams.template_manager list

    # Show template details
    python -m scripts.teams.template_manager show code_review_parallel

    # Validate template
    python -m scripts.teams.template_manager validate my_template.yaml

    # Search templates
    python -m scripts.teams.template_manager search "code review" --vertical coding

    # Apply template to workflow
    python -m scripts.teams.template_manager apply code_review_parallel --goal "Review PR #123"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from victor.workflows.team_templates import (
    TeamTemplate,
    TeamTemplateRegistry,
    get_registry,
    TaskComplexity,
    VerticalType,
)


def format_template_table(templates: list, show_details: bool = False) -> str:
    """Format templates as a table.

    Args:
        templates: List of TeamTemplate objects
        show_details: Whether to show detailed information

    Returns:
        Formatted table string
    """
    if not templates:
        return "No templates found."

    lines = []
    lines.append("=" * 120)

    for template in templates:
        lines.append(f"\nName: {template.name}")
        lines.append(f"Display: {template.display_name}")
        lines.append(f"Description: {template.description}")

        if show_details:
            lines.append(f"Formation: {template.formation}")
            lines.append(f"Vertical: {template.vertical}")
            lines.append(f"Complexity: {template.complexity}")
            lines.append(f"Members: {len(template.members)}")
            lines.append(f"Tool Budget: {template.total_tool_budget}")
            lines.append(f"Timeout: {template.timeout_seconds}s")
            lines.append(f"Tags: {', '.join(template.tags)}")
            lines.append(f"Use Cases: {len(template.use_cases)}")
            if template.use_cases:
                for uc in template.use_cases[:3]:
                    lines.append(f"  - {uc}")

        lines.append("-" * 120)

    return "\n".join(lines)


def cmd_list(args: argparse.Namespace) -> int:
    """List templates with optional filters.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    registry = get_registry()
    registry.load_templates()

    # Build filters
    tags = set(args.tags) if args.tags else None
    template_names = registry.list_templates(
        vertical=args.vertical,
        formation=args.formation,
        complexity=args.complexity,
        tags=tags,
    )

    if not template_names:
        print("No templates found matching filters.")
        return 0

    # Get template objects
    templates = []
    for name in template_names:
        template = registry.get_template(name)
        if template:
            templates.append(template)

    # Sort by name
    templates.sort(key=lambda t: t.name)

    # Output
    if args.json:
        data = [t.to_dict() for t in templates]
        print(json.dumps(data, indent=2))
    else:
        print(format_template_table(templates, show_details=args.verbose))

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show template details.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    registry = get_registry()
    registry.load_templates()

    template = registry.get_template(args.template)
    if not template:
        print(f"Error: Template '{args.template}' not found.")
        return 1

    if args.json:
        print(json.dumps(template.to_dict(), indent=2))
    else:
        # Detailed output
        lines = [
            f"Name: {template.name}",
            f"Display Name: {template.display_name}",
            f"Description: {template.description}",
            f"\nLong Description:",
            template.long_description or "N/A",
            f"\nVersion: {template.version}",
            f"Author: {template.author}",
            f"Vertical: {template.vertical}",
            f"Formation: {template.formation}",
            f"Complexity: {template.complexity}",
            f"\nConfiguration:",
            f"  Max Iterations: {template.max_iterations}",
            f"  Total Tool Budget: {template.total_tool_budget}",
            f"  Timeout: {template.timeout_seconds}s",
            f"\nTags: {', '.join(template.tags)}",
            f"\nUse Cases:",
        ]

        for uc in template.use_cases:
            lines.append(f"  - {uc}")

        lines.append(f"\nMembers ({len(template.members)}):")
        for member in template.members:
            lines.append(f"\n  {member.name} ({member.role})")
            lines.append(f"    Goal: {member.goal[:100]}...")
            lines.append(f"    Tool Budget: {member.tool_budget}")
            if member.expertise:
                lines.append(f"    Expertise: {', '.join(member.expertise[:5])}")

        if template.examples:
            lines.append(f"\nExamples ({len(template.examples)}):")
            for example in template.examples[:3]:
                lines.append(f"  - {example.get('name', 'Example')}: {example.get('description', '')}")

        if template.metadata:
            lines.append(f"\nMetadata:")
            for key, value in template.metadata.items():
                lines.append(f"  {key}: {value}")

        print("\n".join(lines))

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate template file.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    try:
        template = TeamTemplate.from_yaml(args.file)
    except Exception as e:
        print(f"Error loading template: {e}")
        return 1

    # Validate
    registry = get_registry()
    errors = registry.validate_template(template)

    if errors:
        print(f"Validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("✓ Template is valid")
        print(f"  Name: {template.name}")
        print(f"  Formation: {template.formation}")
        print(f"  Members: {len(template.members)}")
        return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search templates.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    registry = get_registry()
    registry.load_templates()

    results = registry.search(
        query=args.query,
        vertical=args.vertical,
        formation=args.formation,
    )

    if not results:
        print(f"No templates found matching '{args.query}'")
        return 0

    # Sort by relevance (name match first, then description match)
    def relevance(t):
        score = 0
        if args.query.lower() in t.name.lower():
            score += 10
        if args.query.lower() in t.description.lower():
            score += 5
        for tag in t.tags:
            if args.query.lower() in tag.lower():
                score += 3
        return score

    results.sort(key=relevance, reverse=True)

    if args.json:
        data = [t.to_dict() for t in results]
        print(json.dumps(data, indent=2))
    else:
        print(f"Found {len(results)} template(s) matching '{args.query}':")
        print()
        print(format_template_table(results, show_details=args.verbose))

    return 0


def cmd_create(args: argparse.Namespace) -> int:
    """Create new template from wizard.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    # Interactive wizard
    print("Team Template Creation Wizard")
    print("=" * 50)
    print()

    # Basic info
    name = input("Template name (snake_case): ").strip()
    if not name:
        print("Error: Name is required")
        return 1

    display_name = input("Display name: ").strip() or name
    description = input("Description: ").strip()
    vertical = input(f"Vertical [{', '.join([v.value for v in VerticalType])}]: ").strip() or "general"
    formation = input("Formation (sequential/parallel/hierarchical/consensus/pipeline): ").strip() or "sequential"
    complexity = input(f"Complexity [{', '.join([c.value for c in TaskComplexity])}]: ").strip() or "standard"

    # Members
    print("\nAdd team members (press Enter with empty name to finish):")
    members = []
    while True:
        member_id = input(f"\nMember {len(members) + 1} ID: ").strip()
        if not member_id:
            break

        member_role = input("Role (researcher/planner/executor/reviewer/writer): ").strip() or "researcher"
        member_name = input("Display name: ").strip() or member_id
        member_goal = input("Goal: ").strip()
        member_budget = input("Tool budget [15]: ").strip() or "15"

        members.append({
            "id": member_id,
            "role": member_role,
            "name": member_name,
            "goal": member_goal,
            "tool_budget": int(member_budget),
        })

    if not members:
        print("Error: At least one member is required")
        return 1

    # Create template
    template_data = {
        "name": name,
        "display_name": display_name,
        "description": description,
        "vertical": vertical,
        "formation": formation,
        "complexity": complexity,
        "members": members,
    }

    try:
        template = TeamTemplate.from_dict(template_data)
    except Exception as e:
        print(f"Error creating template: {e}")
        return 1

    # Validate
    registry = get_registry()
    errors = registry.validate_template(template)
    if errors:
        print(f"\nValidation errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    # Save
    output_path = args.output or f"{name}.yaml"
    template.to_yaml(output_path)
    print(f"\n✓ Template saved to {output_path}")

    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    """Apply template to create team config or workflow node.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    registry = get_registry()
    registry.load_templates()

    template = registry.get_template(args.template)
    if not template:
        print(f"Error: Template '{args.template}' not found.")
        return 1

    # Build context
    context = {}
    if args.context:
        for item in args.context:
            if "=" in item:
                key, value = item.split("=", 1)
                context[key] = value

    # Apply
    if args.workflow_node:
        # Create workflow node
        node_id = args.node_id or f"{template.name}_node"
        node = template.to_team_node(
            node_id=node_id,
            goal=args.goal,
            output_key=args.output_key,
        )

        if args.json:
            import yaml
            # Convert to dict for YAML serialization
            print(yaml.dump(node.to_dict(), default_flow_style=False))
        else:
            print(f"Created team node: {node_id}")
            print(f"Formation: {template.formation}")
            print(f"Members: {len(template.members)}")
    else:
        # Create team config
        config = template.to_team_config(
            goal=args.goal,
            context=context,
        )

        if args.json:
            print(json.dumps(config.to_dict(), indent=2))
        else:
            print(f"Team Configuration: {config.name}")
            print(f"Goal: {config.goal}")
            print(f"Formation: {config.formation}")
            print(f"Members: {len(config.members)}")
            print(f"Max Iterations: {config.max_iterations}")
            print(f"Total Tool Budget: {config.total_tool_budget}")

    return 0


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Team Template Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all templates
  %(prog)s list

  # List coding templates
  %(prog)s list --vertical coding

  # Show template details
  %(prog)s show code_review_parallel

  # Validate template
  %(prog)s validate my_template.yaml

  # Search templates
  %(prog)s search "code review" --vertical coding

  # Create new template (interactive)
  %(prog)s create --output my_template.yaml

  # Apply template
  %(prog)s apply code_review_parallel --goal "Review PR #123"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List templates")
    list_parser.add_argument("--vertical", help="Filter by vertical")
    list_parser.add_argument("--formation", help="Filter by formation")
    list_parser.add_argument("--complexity", help="Filter by complexity")
    list_parser.add_argument("--tags", nargs="+", help="Filter by tags")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show details")
    list_parser.add_argument("--json", action="store_true", help="JSON output")
    list_parser.set_defaults(func=cmd_list)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show template details")
    show_parser.add_argument("template", help="Template name")
    show_parser.add_argument("--json", action="store_true", help="JSON output")
    show_parser.set_defaults(func=cmd_show)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate template")
    validate_parser.add_argument("file", help="Template YAML file")
    validate_parser.set_defaults(func=cmd_validate)

    # Search command
    search_parser = subparsers.add_parser("search", help="Search templates")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--vertical", help="Filter by vertical")
    search_parser.add_argument("--formation", help="Filter by formation")
    search_parser.add_argument("-v", "--verbose", action="store_true", help="Show details")
    search_parser.add_argument("--json", action="store_true", help="JSON output")
    search_parser.set_defaults(func=cmd_search)

    # Create command
    create_parser = subparsers.add_parser("create", help="Create new template (wizard)")
    create_parser.add_argument("-o", "--output", help="Output file path")
    create_parser.set_defaults(func=cmd_create)

    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply template")
    apply_parser.add_argument("template", help="Template name")
    apply_parser.add_argument("--goal", help="Override team goal")
    apply_parser.add_argument("--context", nargs="+", help="Context (key=value pairs)")
    apply_parser.add_argument("--workflow-node", action="store_true", help="Create workflow node")
    apply_parser.add_argument("--node-id", help="Workflow node ID")
    apply_parser.add_argument("--output-key", help="Output key")
    apply_parser.add_argument("--json", action="store_true", help="JSON output")
    apply_parser.set_defaults(func=cmd_apply)

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
