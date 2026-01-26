#!/usr/bin/env python3
"""
Victor Configuration Migration Script (0.5.x to 0.5.0)

This script automatically migrates configuration from Victor 0.5.x to 0.5.0 format.

Usage:
    python scripts/migration/migrate_config.py --source ./old_config.py --dest ./victor/config
    python scripts/migration/migrate_config.py --validate ./victor/config
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

# Add victor to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ConfigMigrator:
    """Migrates Victor configuration from 0.5.x to 0.5.0."""

    def __init__(self, source: Path, dest: Optional[Path] = None):
        self.source = source
        self.dest = dest
        self.changes_made = []

    def migrate_python_config(self, source_file: Path) -> Dict[str, Any]:
        """Migrate Python configuration file to 0.5.0 format."""
        config = {}

        # Read old config
        if source_file.exists():
            with open(source_file) as f:
                content = f.read()

            # Parse old config patterns
            if "Config()" in content:
                print(f"Found old Config() usage in {source_file}")
                self.changes_made.append(f"Parsed Config() from {source_file}")

            # Extract configuration values
            if "max_tokens" in content:
                config["max_tokens"] = self._extract_value(content, "max_tokens")
            if "temperature" in content:
                config["temperature"] = self._extract_value(content, "temperature")
            if "tool_budget" in content:
                config["tool_budget"] = self._extract_value(content, "tool_budget")
            if "use_semantic_tool_selection" in content:
                config["tool_selection_strategy"] = "hybrid"
                self.changes_made.append(
                    "Migrated use_semantic_tool_selection to tool_selection_strategy=hybrid"
                )

        return config

    def migrate_env_file(self, env_file: Path) -> Dict[str, str]:
        """Migrate .env file to 0.5.0 format."""
        env_vars = {}

        if not env_file.exists():
            return env_vars

        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Migrate old variable names
                    if key == "VICTOR_TOOL_SELECTION":
                        new_key = "VICTOR_TOOL_SELECTION_STRATEGY"
                        env_vars[new_key] = value
                        self.changes_made.append(f"Renamed {key} to {new_key}")
                    elif key == "VICTOR_USE_SEMANTIC":
                        new_key = "VICTOR_TOOL_SELECTION_STRATEGY"
                        env_vars[new_key] = "semantic" if value.lower() == "true" else "keyword"
                        self.changes_made.append(f"Converted {key} to {new_key}")
                    elif key == "VICTOR_CACHE_ENABLED":
                        new_key = "VICTOR_TOOL_CACHE_ENABLED"
                        env_vars[new_key] = value
                        self.changes_made.append(f"Renamed {key} to {new_key}")
                    else:
                        env_vars[key] = value

        return env_vars

    def create_yaml_config(self, config: Dict[str, Any], dest_dir: Path) -> None:
        """Create YAML configuration files."""
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Create settings.yaml
        settings_file = dest_dir / "settings.yaml"
        with open(settings_file, "w") as f:
            f.write("# Victor 0.5.0 Configuration\n")
            f.write("# Migrated from 0.5.x\n\n")

            f.write("agent:\n")
            if "max_tokens" in config:
                f.write(f"  max_tokens: {config['max_tokens']}\n")
            if "temperature" in config:
                f.write(f"  temperature: {config['temperature']}\n")
            if "tool_budget" in config:
                f.write(f"  tool_budget: {config['tool_budget']}\n")

            f.write("\ntool_selection:\n")
            if "tool_selection_strategy" in config:
                f.write(f"  strategy: {config['tool_selection_strategy']}\n")
            else:
                f.write("  strategy: hybrid  # Default\n")

            f.write("  cache_enabled: true\n")
            f.write("  cache_size: 500\n")

        self.changes_made.append(f"Created {settings_file}")

    def update_env_file(self, env_vars: Dict[str, str], env_file: Path) -> None:
        """Update .env file with new variable names."""
        with open(env_file, "w") as f:
            f.write("# Victor 0.5.0 Environment Variables\n")
            f.write("# Migrated from 0.5.x\n\n")

            for key, value in sorted(env_vars.items()):
                f.write(f"{key}={value}\n")

        self.changes_made.append(f"Updated {env_file}")

    def _extract_value(self, content: str, key: str) -> Any:
        """Extract value from Python config content."""
        import re

        patterns = [
            rf"{key}\s*=\s*(.+)",
            rf"{key}\s*=\s*(.+)\n",
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                value = match.group(1).strip()
                # Try to evaluate as Python literal
                try:
                    return eval(value)
                except:
                    # Remove quotes if string
                    if value.startswith('"') and value.endswith('"'):
                        return value[1:-1]
                    if value.startswith("'") and value.endswith("'"):
                        return value[1:-1]
                    return value

        return None

    def validate_config(self, config_dir: Path) -> bool:
        """Validate migrated configuration."""
        print(f"\nValidating configuration in {config_dir}...")

        issues = []

        # Check if directory exists
        if not config_dir.exists():
            issues.append(f"Configuration directory does not exist: {config_dir}")
            return False

        # Validate settings.yaml
        settings_file = config_dir / "settings.yaml"
        if not settings_file.exists():
            issues.append(f"settings.yaml not found in {config_dir}")
        else:
            try:
                import yaml

                with open(settings_file) as f:
                    yaml.safe_load(f)
                print(f"✓ {settings_file} is valid YAML")
            except Exception as e:
                issues.append(f"Invalid YAML in {settings_file}: {e}")

        # Validate .env file
        env_file = config_dir.parent / ".env"
        if env_file.exists():
            print(f"✓ {env_file} exists")
            # Check for deprecated variables
            with open(env_file) as f:
                content = f.read()
                deprecated = [
                    "VICTOR_API_KEY",
                    "VICTOR_USE_SEMANTIC",
                    "VICTOR_CACHE_ENABLED",
                ]
                for var in deprecated:
                    if var in content:
                        issues.append(f"Deprecated variable found: {var}")

        if issues:
            print("\nValidation issues found:")
            for issue in issues:
                print(f"  ✗ {issue}")
            return False
        else:
            print("✓ Configuration is valid")
            return True

    def migrate(self) -> bool:
        """Perform migration."""
        print(f"Migrating configuration from {self.source} to {self.dest}")

        # Determine source type
        if self.source.is_file():
            if self.source.suffix == ".py":
                config = self.migrate_python_config(self.source)
                if self.dest:
                    self.create_yaml_config(config, self.dest)
            elif self.source.name == ".env":
                env_vars = self.migrate_env_file(self.source)
                if self.dest:
                    self.update_env_file(env_vars, self.dest / ".env")
            else:
                print(f"Unsupported source file type: {self.source.suffix}")
                return False

        elif self.source.is_dir():
            # Migrate all config files in directory
            env_file = self.source / ".env"
            if env_file.exists():
                env_vars = self.migrate_env_file(env_file)
                if self.dest:
                    self.update_env_file(env_vars, self.dest / ".env")

            # Look for Python config files
            for py_file in self.source.glob("*.py"):
                if "config" in py_file.name.lower():
                    config = self.migrate_python_config(py_file)
                    if self.dest:
                        self.create_yaml_config(config, self.dest)
        else:
            print(f"Source not found: {self.source}")
            return False

        # Print summary
        print("\nMigration complete!")
        print(f"\nChanges made ({len(self.changes_made)}):")
        for change in self.changes_made:
            print(f"  • {change}")

        return True


def main():
    parser = argparse.ArgumentParser(description="Migrate Victor configuration from 0.5.x to 0.5.0")
    parser.add_argument(
        "--source", type=Path, required=True, help="Source configuration file or directory"
    )
    parser.add_argument(
        "--dest", type=Path, help="Destination directory for migrated configuration"
    )
    parser.add_argument("--validate", type=Path, help="Validate migrated configuration")

    args = parser.parse_args()

    if args.validate:
        migrator = ConfigMigrator(args.validate)
        success = migrator.validate_config(args.validate)
        sys.exit(0 if success else 1)

    if not args.dest:
        print("Error: --dest is required unless using --validate")
        sys.exit(1)

    migrator = ConfigMigrator(args.source, args.dest)
    success = migrator.migrate()

    # Validate after migration
    if success and args.dest:
        print("\nValidating migrated configuration...")
        migrator.validate_config(args.dest)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
