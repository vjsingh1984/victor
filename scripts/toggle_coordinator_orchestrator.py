#!/usr/bin/env python3
"""Coordinator Orchestrator Feature Flag Toggle Script

This script enables or disables the coordinator-based orchestrator feature flag
in a safe, controlled manner. It backs up settings, logs changes, and validates
the configuration before and after toggling.

Usage:
    python scripts/toggle_coordinator_orchestrator.py enable [--backup]
    python scripts/toggle_coordinator_orchestrator.py disable [--backup]
    python scripts/toggle_coordinator_orchestrator.py status
    python scripts/toggle_coordinator_orchestrator.py validate
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from victor.config.settings import GLOBAL_VICTOR_DIR


class FeatureFlagToggle:
    """Manager for toggling coordinator orchestrator feature flag."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.profiles_path = GLOBAL_VICTOR_DIR / "profiles.yaml"
        self.backup_dir = GLOBAL_VICTOR_DIR / "backups"
        self.log_path = GLOBAL_VICTOR_DIR / "logs" / "feature_flag_history.jsonl"

        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        if self.verbose:
            prefix = {
                "INFO": "✓",
                "WARN": "⚠",
                "ERROR": "✗",
                "SUCCESS": "★",
            }.get(level, "•")
            print(f"{prefix} {message}")

    def log_action(self, action: str, previous_value: Optional[bool], new_value: bool, success: bool, notes: str = ""):
        """Log an action to history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "previous_value": previous_value,
            "new_value": new_value,
            "success": success,
            "notes": notes,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def get_current_status(self) -> dict:
        """Get current feature flag status."""
        try:
            from victor.config.settings import Settings

            settings = Settings()
            return {
                "enabled": settings.use_coordinator_orchestrator,
                "source": "environment_override" if "VICTOR_USE_COORDINATOR_ORCHESTRATOR" in __import__('os').environ else "settings",
                "profiles_path": str(self.profiles_path),
                "profiles_exists": self.profiles_path.exists(),
            }
        except Exception as e:
            return {
                "enabled": None,
                "error": str(e),
                "profiles_path": str(self.profiles_path),
                "profiles_exists": self.profiles_path.exists(),
            }

    def backup_settings(self) -> Optional[Path]:
        """Backup current settings file."""
        if not self.profiles_path.exists():
            self.log("No settings file to backup", "WARN")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"profiles.yaml.backup_{timestamp}"

        import shutil
        shutil.copy2(self.profiles_path, backup_path)

        self.log(f"Settings backed up to: {backup_path}", "SUCCESS")
        return backup_path

    def restore_settings(self, backup_path: Path) -> bool:
        """Restore settings from backup."""
        if not backup_path.exists():
            self.log(f"Backup file not found: {backup_path}", "ERROR")
            return False

        import shutil
        shutil.copy2(backup_path, self.profiles_path)

        self.log(f"Settings restored from: {backup_path}", "SUCCESS")
        return True

    def load_profiles(self) -> dict:
        """Load profiles.yaml."""
        if not self.profiles_path.exists():
            return {"profiles": {}}

        import yaml
        with open(self.profiles_path, "r") as f:
            return yaml.safe_load(f) or {"profiles": {}}

    def save_profiles(self, data: dict) -> bool:
        """Save profiles.yaml."""
        import yaml
        try:
            with open(self.profiles_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            self.log(f"Failed to save settings: {e}", "ERROR")
            return False

    def enable(self, backup: bool = True) -> bool:
        """Enable the coordinator orchestrator feature flag."""
        status = self.get_current_status()
        previous_value = status.get("enabled")

        if previous_value is True:
            self.log("Feature flag is already enabled", "INFO")
            return True

        self.log("Enabling coordinator orchestrator...", "INFO")

        # Backup if requested
        backup_path = None
        if backup:
            backup_path = self.backup_settings()

        try:
            # Load current settings
            data = self.load_profiles()

            # Set feature flag at top level
            data["use_coordinator_orchestrator"] = True

            # Save settings
            if not self.save_profiles(data):
                self.log_action("enable", previous_value, True, False, "Failed to save settings")
                return False

            # Validate the change (note: Settings may be cached in current process)
            # The file has been updated and will take effect in new Victor instances
            self.log("Settings file updated successfully!", "SUCCESS")
            self.log("Note: Changes will take effect in new Victor instances (restart required)", "INFO")
            self.log_action(
                "enable",
                previous_value,
                True,
                True,
                f"Backup: {backup_path}" if backup_path else "No backup"
            )
            return True

        except Exception as e:
            self.log(f"Error enabling feature flag: {e}", "ERROR")
            self.log_action("enable", previous_value, False, False, str(e))

            # Restore backup if we created one
            if backup and backup_path:
                self.log("Restoring from backup...", "WARN")
                self.restore_settings(backup_path)

            return False

    def disable(self, backup: bool = True) -> bool:
        """Disable the coordinator orchestrator feature flag."""
        status = self.get_current_status()
        previous_value = status.get("enabled")

        if previous_value is False:
            self.log("Feature flag is already disabled", "INFO")
            return True

        self.log("Disabling coordinator orchestrator...", "INFO")

        # Backup if requested
        backup_path = None
        if backup:
            backup_path = self.backup_settings()

        try:
            # Load current settings
            data = self.load_profiles()

            # Set feature flag at top level
            data["use_coordinator_orchestrator"] = False

            # Save settings
            if not self.save_profiles(data):
                self.log_action("disable", previous_value, False, False, "Failed to save settings")
                return False

            # Validate the change (note: Settings may be cached in current process)
            # The file has been updated and will take effect in new Victor instances
            self.log("Settings file updated successfully!", "SUCCESS")
            self.log("Note: Changes will take effect in new Victor instances (restart required)", "INFO")
            self.log_action(
                "disable",
                previous_value,
                False,
                True,
                f"Backup: {backup_path}" if backup_path else "No backup"
            )
            return True

        except Exception as e:
            self.log(f"Error disabling feature flag: {e}", "ERROR")
            self.log_action("disable", previous_value, True, False, str(e))

            # Restore backup if we created one
            if backup and backup_path:
                self.log("Restoring from backup...", "WARN")
                self.restore_settings(backup_path)

            return False

    def validate(self) -> bool:
        """Validate the current configuration."""
        self.log("Validating coordinator orchestrator configuration...", "INFO")

        status = self.get_current_status()

        if status.get("enabled") is None:
            self.log("Unable to determine feature flag status", "ERROR")
            if "error" in status:
                self.log(f"Error: {status['error']}", "ERROR")
            return False

        self.log(f"Feature flag status: {'ENABLED' if status['enabled'] else 'DISABLED'}", "INFO")
        self.log(f"Settings file: {status['profiles_path']}", "INFO")
        self.log(f"Settings exists: {status['profiles_exists']}", "INFO")

        # Run validation script
        import subprocess

        try:
            result = subprocess.run(
                [sys.executable, "scripts/validate_coordinator_orchestrator.py", "--quick"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                self.log("Configuration validation passed", "SUCCESS")
                return True
            else:
                self.log("Configuration validation failed", "ERROR")
                if result.stderr:
                    self.log(f"Validation errors:\n{result.stderr}", "ERROR")
                return False

        except subprocess.TimeoutExpired:
            self.log("Validation timed out", "WARN")
            return False
        except Exception as e:
            self.log(f"Validation error: {e}", "ERROR")
            return False

    def print_status(self):
        """Print current status."""
        status = self.get_current_status()

        print("\n" + "=" * 70)
        print("COORDINATOR ORCHESTRATOR FEATURE FLAG STATUS")
        print("=" * 70)

        if status.get("enabled") is None:
            print(f"Status: ERROR")
            if "error" in status:
                print(f"Error: {status['error']}")
        else:
            print(f"Status: {'ENABLED ✓' if status['enabled'] else 'DISABLED ✗'}")
            print(f"Source: {status.get('source', 'unknown')}")
            print(f"Settings File: {status['profiles_path']}")
            print(f"Settings Exists: {status['profiles_exists']}")

        print("=" * 70)

        # Show recent history
        if self.log_path.exists():
            print("\nRecent Actions (Last 5):")
            print("-" * 70)

            try:
                with open(self.log_path, "r") as f:
                    lines = f.readlines()[-5:]  # Last 5 entries

                for line in reversed(lines):
                    entry = json.loads(line.strip())
                    status_icon = "✓" if entry['success'] else "✗"
                    print(f"{status_icon} {entry['timestamp']} - {entry['action'].upper()}")
                    print(f"  {entry['previous_value']} → {entry['new_value']}")
                    if entry.get('notes'):
                        print(f"  Note: {entry['notes']}")
                    print()
            except Exception as e:
                print(f"Unable to read history: {e}")

        print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Toggle coordinator orchestrator feature flag"
    )
    parser.add_argument(
        "action",
        choices=["enable", "disable", "status", "validate"],
        help="Action to perform",
    )
    parser.add_argument(
        "--backup", "-b",
        action="store_true",
        help="Backup settings before making changes (recommended)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    toggle = FeatureFlagToggle(verbose=args.verbose)

    try:
        if args.action == "enable":
            success = toggle.enable(backup=args.backup)
            if success:
                print("\n★ Coordinator orchestrator enabled successfully!")
                print("  Run 'victor chat' to test the new architecture.")
                print("  Use 'python scripts/toggle_coordinator_orchestrator.py validate' to verify.")
                sys.exit(0)
            else:
                print("\n✗ Failed to enable coordinator orchestrator")
                sys.exit(1)

        elif args.action == "disable":
            success = toggle.disable(backup=args.backup)
            if success:
                print("\n★ Coordinator orchestrator disabled successfully!")
                print("  Victor will use the legacy orchestrator.")
                sys.exit(0)
            else:
                print("\n✗ Failed to disable coordinator orchestrator")
                sys.exit(1)

        elif args.action == "status":
            toggle.print_status()
            sys.exit(0)

        elif args.action == "validate":
            success = toggle.validate()
            if success:
                print("\n★ Configuration is valid!")
                sys.exit(0)
            else:
                print("\n✗ Configuration validation failed")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
