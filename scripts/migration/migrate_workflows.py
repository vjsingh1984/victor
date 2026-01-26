#!/usr/bin/env python3
"""
Victor Workflow Migration Script (0.5.x to 0.5.0)

This script automatically migrates workflows from Victor 0.5.x to 0.5.0 format.

Usage:
    python scripts/migration/migrate_workflows.py --source ./old_workflows --dest ./victor/workflows
    python scripts/migration/migrate_workflows.py --validate ./victor/workflows
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
import json
import yaml

# Add victor to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class WorkflowMigrator:
    """Migrates Victor workflows from 0.5.x to 0.5.0."""

    def __init__(self, source: Path, dest: Optional[Path] = None):
        self.source = source
        self.dest = dest
        self.changes_made = []

    def migrate_python_workflow(self, source_file: Path) -> Dict[str, Any]:
        """Migrate Python workflow definition to YAML format."""
        workflow = {"workflows": {}}

        # Try to parse Python file
        content = source_file.read_text()

        # Basic extraction of workflow structure
        # This is simplified - real implementation would parse AST
        if "StateGraph" in content or "graph = " in content:
            workflow_name = source_file.stem
            workflow["workflows"][workflow_name] = {
                "metadata": {"version": "0.5.0", "migrated_from": "python"},
                "nodes": [],
            }

            self.changes_made.append(f"Migrated Python workflow from {source_file}")

        return workflow

    def migrate_yaml_workflow(self, source_file: Path) -> Dict[str, Any]:
        """Migrate YAML workflow to 0.5.0 format."""
        with open(source_file) as f:
            old_workflow = yaml.safe_load(f)

        new_workflow = {"workflows": {}}

        # Check if already migrated
        if "workflows" in old_workflow:
            return old_workflow  # Already in 0.5.0 format

        # Migrate to new format
        workflow_name = old_workflow.get("name", source_file.stem)
        new_workflow["workflows"][workflow_name] = {
            "metadata": {"version": "0.5.0", "migrated_from": "yaml_0.5.x"},
            "nodes": [],
        }

        # Migrate nodes
        old_nodes = old_workflow.get("nodes", [])
        for node in old_nodes:
            new_node = self.migrate_node(node)
            new_workflow["workflows"][workflow_name]["nodes"].append(new_node)

        self.changes_made.append(f"Migrated YAML workflow from {source_file}")

        return new_workflow

    def migrate_node(self, old_node: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate a single node to 0.5.0 format."""
        new_node = {
            "id": old_node.get("id", "unknown"),
            "type": old_node.get("type", "agent"),
        }

        # Migrate agent nodes
        if old_node.get("type") == "agent":
            if "agent" in old_node:
                new_node["role"] = old_node["agent"]
                self.changes_made.append(f"Renamed 'agent' to 'role' in node {new_node['id']}")
            if "prompt" in old_node:
                new_node["goal"] = old_node["prompt"]
                self.changes_made.append(f"Renamed 'prompt' to 'goal' in node {new_node['id']}")

        # Add required 'next' field
        if "next" not in old_node:
            # Try to infer next from edges
            if "edges" in old_node and old_node["edges"]:
                new_node["next"] = old_node["edges"][0].get("to", "END")
            else:
                new_node["next"] = ["END"] if old_node.get("terminal", False) else []
        else:
            new_node["next"] = old_node["next"]

        # Copy other fields
        for key, value in old_node.items():
            if key not in ["id", "type", "agent", "prompt", "next", "edges"]:
                new_node[key] = value

        return new_node

    def save_workflow(self, workflow: Dict[str, Any], dest_file: Path) -> None:
        """Save migrated workflow to YAML file."""
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_file, "w") as f:
            yaml.dump(workflow, f, default_flow_style=False, sort_keys=False)

        self.changes_made.append(f"Created {dest_file}")

    def migrate(self) -> bool:
        """Perform migration."""
        print(f"Migrating workflows from {self.source} to {self.dest}")

        if not self.source.exists():
            print(f"Source not found: {self.source}")
            return False

        # Migrate single file
        if self.source.is_file():
            workflow = self.migrate_file(self.source)
            if workflow and self.dest:
                self.save_workflow(workflow, self.dest)

        # Migrate directory
        elif self.source.is_dir():
            for source_file in self.source.rglob("*.yaml"):
                workflow = self.migrate_file(source_file)
                if workflow and self.dest:
                    # Preserve directory structure
                    rel_path = source_file.relative_to(self.source)
                    dest_file = self.dest / rel_path
                    self.save_workflow(workflow, dest_file)

            for source_file in self.source.rglob("*.py"):
                workflow = self.migrate_file(source_file)
                if workflow and self.dest:
                    # Python files go to workflows/ directory
                    dest_file = self.dest / f"{source_file.stem}.yaml"
                    self.save_workflow(workflow, dest_file)

        # Print summary
        print("\nMigration complete!")
        print(f"\nChanges made ({len(self.changes_made)}):")
        for change in self.changes_made[:10]:  # Show first 10
            print(f"  • {change}")
        if len(self.changes_made) > 10:
            print(f"  ... and {len(self.changes_made) - 10} more")

        return True

    def migrate_file(self, source_file: Path) -> Dict[str, Any]:
        """Migrate a single workflow file."""
        if source_file.suffix == ".yaml":
            return self.migrate_yaml_workflow(source_file)
        elif source_file.suffix == ".py":
            return self.migrate_python_workflow(source_file)
        else:
            print(f"Skipping unsupported file: {source_file}")
            return {}

    def validate_workflow(self, workflow_file: Path) -> bool:
        """Validate migrated workflow."""
        print(f"\nValidating workflow: {workflow_file}")

        issues = []

        # Check if file exists
        if not workflow_file.exists():
            issues.append(f"Workflow file not found: {workflow_file}")
            return False

        # Validate YAML syntax
        try:
            with open(workflow_file) as f:
                workflow = yaml.safe_load(f)
        except Exception as e:
            issues.append(f"Invalid YAML: {e}")
            return False

        # Validate structure
        if "workflows" not in workflow:
            issues.append("Missing 'workflows' root key")
            return False

        # Validate each workflow
        for workflow_name, workflow_def in workflow["workflows"].items():
            if "nodes" not in workflow_def:
                issues.append(f"Workflow '{workflow_name}' missing 'nodes'")

            # Validate nodes
            for node in workflow_def.get("nodes", []):
                if "id" not in node:
                    issues.append(f"Node missing 'id'")
                if "type" not in node:
                    issues.append(f"Node {node.get('id', 'unknown')} missing 'type'")
                if "next" not in node:
                    issues.append(f"Node {node.get('id', 'unknown')} missing 'next'")

        if issues:
            print("\nValidation issues found:")
            for issue in issues:
                print(f"  ✗ {issue}")
            return False
        else:
            print(f"✓ {workflow_file} is valid")
            return True

    def validate_workflows(self, workflow_dir: Path) -> bool:
        """Validate all workflows in directory."""
        print(f"\nValidating workflows in {workflow_dir}...")

        all_valid = True
        count = 0

        for workflow_file in workflow_dir.rglob("*.yaml"):
            if not self.validate_workflow(workflow_file):
                all_valid = False
            count += 1

        if count == 0:
            print("No workflow files found")
            return False

        if all_valid:
            print(f"\n✓ All {count} workflow(s) are valid")
        else:
            print(f"\n✗ Some workflows have validation issues")

        return all_valid


def main():
    parser = argparse.ArgumentParser(description="Migrate Victor workflows from 0.5.x to 0.5.0")
    parser.add_argument(
        "--source", type=Path, required=True, help="Source workflow file or directory"
    )
    parser.add_argument("--dest", type=Path, help="Destination directory for migrated workflows")
    parser.add_argument("--validate", type=Path, help="Validate migrated workflows")

    args = parser.parse_args()

    if args.validate:
        migrator = WorkflowMigrator(args.validate)
        success = migrator.validate_workflows(args.validate)
        sys.exit(0 if success else 1)

    if not args.dest:
        print("Error: --dest is required unless using --validate")
        sys.exit(1)

    migrator = WorkflowMigrator(args.source, args.dest)
    success = migrator.migrate()

    # Validate after migration
    if success and args.dest:
        print("\nValidating migrated workflows...")
        migrator.validate_workflows(args.dest)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
