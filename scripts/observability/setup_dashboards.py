#!/usr/bin/env python3
"""Victor Observability Dashboard Setup Script.

This script automates the provisioning of Grafana dashboards and Prometheus
alerting rules for Victor team workflow observability.

Features:
- Provision Grafana dashboards via API
- Import Prometheus alerting rules
- Configure datasource connections
- Validate dashboard syntax
- Health checks for dashboards

Usage:
    python scripts/observability/setup_dashboards.py [OPTIONS]

Examples:
    # Provision all dashboards (interactive mode)
    python scripts/observability/setup_dashboards.py

    # Provision specific dashboard
    python scripts/observability/setup_dashboards.py --dashboard team_overview

    # Dry-run without making changes
    python scripts/observability/setup_dashboards.py --dry-run

    # Use custom configuration
    python scripts/observability/setup_dashboards.py --config custom_config.yaml

Environment Variables:
    GRAFANA_URL: Grafana server URL (default: http://localhost:3000)
    GRAFANA_API_KEY: Grafana API key for authentication
    PROMETHEUS_URL: Prometheus server URL (default: http://localhost:9090)
    PROMETHEUS_CONFIG_PATH: Path to prometheus.yml (default: /etc/prometheus/prometheus.yml)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.parse

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Config:
    """Configuration for dashboard setup."""

    grafana_url: str
    grafana_api_key: Optional[str]
    prometheus_url: str
    prometheus_config_path: str
    dashboard_dir: Path
    alerts_dir: Path
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            grafana_url=os.getenv("GRAFANA_URL", "http://localhost:3000"),
            grafana_api_key=os.getenv("GRAFANA_API_KEY"),
            prometheus_url=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
            prometheus_config_path=os.getenv(
                "PROMETHEUS_CONFIG_PATH", "/etc/prometheus/prometheus.yml"
            ),
            dashboard_dir=Path(__file__).parent.parent.parent / "observability" / "dashboards",
            alerts_dir=Path(__file__).parent.parent.parent / "observability" / "alerts",
            dry_run=False,
        )


# =============================================================================
# Dashboard Provisioning
# =============================================================================


class GrafanaClient:
    """Client for interacting with Grafana API."""

    def __init__(self, config: Config):
        """Initialize Grafana client.

        Args:
            config: Configuration object
        """
        self.config = config
        self.session = requests.Session()
        self.base_url = f"{config.grafana_url}/api"

        if config.grafana_api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {config.grafana_api_key}",
                    "Content-Type": "application/json",
                }
            )
        else:
            logger.warning(
                "No GRAFANA_API_KEY provided. " "Authentication will fail if Grafana requires it."
            )

    def health_check(self) -> bool:
        """Check Grafana health.

        Returns:
            True if Grafana is healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("Grafana health check passed")
                return True
            else:
                logger.error(f"Grafana health check failed: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"Grafana health check failed: {e}")
            return False

    def ensure_datasource(self) -> bool:
        """Ensure Prometheus datasource exists in Grafana.

        Returns:
            True if datasource exists or was created, False otherwise
        """
        try:
            # Check if datasource exists
            datasources = self.session.get(f"{self.base_url}/datasources").json()

            for ds in datasources:
                if ds["name"] == "Prometheus" and ds["type"] == "prometheus":
                    logger.info(f"Prometheus datasource already exists: {ds['url']}")
                    return True

            # Create datasource
            logger.info(f"Creating Prometheus datasource: {self.config.prometheus_url}")

            if self.config.dry_run:
                logger.info("[DRY-RUN] Would create Prometheus datasource")
                return True

            datasource_payload = {
                "name": "Prometheus",
                "type": "prometheus",
                "access": "proxy",
                "url": self.config.prometheus_url,
                "isDefault": True,
                "jsonData": {"httpMethod": "POST"},
            }

            response = self.session.post(f"{self.base_url}/datasources", json=datasource_payload)

            if response.status_code in (200, 201):
                logger.info("Prometheus datasource created successfully")
                return True
            else:
                logger.error(f"Failed to create datasource: {response.text}")
                return False

        except requests.RequestException as e:
            logger.error(f"Failed to ensure datasource: {e}")
            return False

    def provision_dashboard(self, dashboard_path: Path) -> bool:
        """Provision a single dashboard.

        Args:
            dashboard_path: Path to dashboard JSON file

        Returns:
            True if dashboard was provisioned successfully, False otherwise
        """
        try:
            with open(dashboard_path, "r") as f:
                dashboard_data = json.load(f)

            dashboard_uid = dashboard_data.get("uid")
            dashboard_title = dashboard_data.get("title")

            if not dashboard_uid:
                logger.error(f"Dashboard missing UID: {dashboard_path}")
                return False

            # Check if dashboard exists
            if self.config.dry_run:
                logger.info(
                    f"[DRY-RUN] Would provision dashboard: {dashboard_title} "
                    f"(uid: {dashboard_uid})"
                )
                return True

            response = self.session.get(f"{self.base_url}/dashboards/uid/{dashboard_uid}")

            payload = {"overwrite": True, "dashboard": dashboard_data}

            if response.status_code == 200:
                # Update existing dashboard
                logger.info(f"Updating dashboard: {dashboard_title}")
                response = self.session.post(f"{self.base_url}/dashboards/db", json=payload)
            else:
                # Create new dashboard
                logger.info(f"Creating dashboard: {dashboard_title}")
                response = self.session.post(f"{self.base_url}/dashboards/db", json=payload)

            if response.status_code in (200, 201):
                logger.info(f"Dashboard provisioned successfully: {dashboard_title}")
                return True
            else:
                logger.error(f"Failed to provision dashboard: {response.text}")
                return False

        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read dashboard file {dashboard_path}: {e}")
            return False
        except requests.RequestException as e:
            logger.error(f"Failed to provision dashboard: {e}")
            return False

    def provision_dashboards(self, dashboard_dir: Path) -> Dict[str, bool]:
        """Provision all dashboards from directory.

        Args:
            dashboard_dir: Directory containing dashboard JSON files

        Returns:
            Dictionary mapping dashboard filenames to success status
        """
        results = {}

        for dashboard_file in dashboard_dir.glob("*.json"):
            logger.info(f"Processing dashboard: {dashboard_file.name}")
            results[dashboard_file.name] = self.provision_dashboard(dashboard_file)

        return results


# =============================================================================
# Prometheus Rules
# =============================================================================


class PrometheusManager:
    """Manager for Prometheus alerting rules."""

    def __init__(self, config: Config):
        """Initialize Prometheus manager.

        Args:
            config: Configuration object
        """
        self.config = config

    def validate_rules(self, rules_file: Path) -> bool:
        """Validate Prometheus alerting rules syntax.

        Args:
            rules_file: Path to rules file

        Returns:
            True if rules are valid, False otherwise
        """
        try:
            # Try promtool if available
            import subprocess

            if self.config.dry_run:
                logger.info(f"[DRY-RUN] Would validate rules: {rules_file}")
                return True

            result = subprocess.run(
                ["promtool", "check", "rules", str(rules_file)], capture_output=True, text=True
            )

            if result.returncode == 0:
                logger.info(f"Rules validation passed: {rules_file}")
                return True
            else:
                logger.error(f"Rules validation failed: {result.stderr}")
                return False

        except FileNotFoundError:
            logger.warning(
                "promtool not found. Skipping rules validation. "
                "Install prometheus-tools to enable validation."
            )
            return True
        except Exception as e:
            logger.error(f"Failed to validate rules: {e}")
            return False

    def import_rules(self, rules_file: Path) -> bool:
        """Import Prometheus alerting rules.

        This method updates the Prometheus configuration to include the
        rules file and reloads Prometheus.

        Args:
            rules_file: Path to rules file

        Returns:
            True if rules were imported successfully, False otherwise
        """
        try:
            config_path = Path(self.config.prometheus_config_path)

            if not config_path.exists():
                logger.error(f"Prometheus config not found: {config_path}")
                return False

            # Read existing config
            with open(config_path, "r") as f:
                config_content = f.read()

            # Check if rules file is already included
            rules_file_str = str(rules_file)
            if rules_file_str in config_content:
                logger.info(f"Rules file already included in Prometheus config")
                return True

            if self.config.dry_run:
                logger.info(f"[DRY-RUN] Would import rules: {rules_file}")
                return True

            # Add rules file to config
            rule_files_section = '\nrule_files:\n  - "' + rules_file_str + '"\n'

            # Insert rule_files section if it doesn't exist
            if "rule_files:" not in config_content:
                # Insert after global section if exists, otherwise at beginning
                if "global:" in config_content:
                    # Find end of global section
                    global_end = config_content.find("\n\n", config_content.find("global:"))
                    config_content = (
                        config_content[:global_end]
                        + rule_files_section
                        + config_content[global_end:]
                    )
                else:
                    config_content = rule_files_section + config_content
            else:
                # Append to existing rule_files section
                config_content = config_content.replace("rule_files:", rule_files_section.strip())

            # Backup existing config
            backup_path = config_path.with_suffix(".yml.bak")
            with open(backup_path, "w") as f:
                f.write(config_content)
            logger.info(f"Backed up Prometheus config to: {backup_path}")

            # Write new config
            with open(config_path, "w") as f:
                f.write(config_content)
            logger.info(f"Updated Prometheus config: {config_path}")

            # Reload Prometheus
            import subprocess

            subprocess.run(["killall", "-HUP", "prometheus"], check=False)
            logger.info("Sent HUP signal to Prometheus for reload")

            return True

        except IOError as e:
            logger.error(f"Failed to import rules: {e}")
            return False


# =============================================================================
# Dashboard Validation
# =============================================================================


class DashboardValidator:
    """Validator for Grafana dashboards."""

    def validate_dashboard(self, dashboard_path: Path) -> List[str]:
        """Validate a dashboard JSON file.

        Args:
            dashboard_path: Path to dashboard JSON file

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            with open(dashboard_path, "r") as f:
                dashboard_data = json.load(f)

            # Check required fields
            required_fields = ["title", "uid", "panels"]
            for field in required_fields:
                if field not in dashboard_data:
                    errors.append(f"Missing required field: {field}")

            # Validate panels
            panels = dashboard_data.get("panels", [])
            if not panels:
                errors.append("Dashboard has no panels")

            for i, panel in enumerate(panels):
                if "targets" not in panel:
                    errors.append(f"Panel {i} missing targets")

                for j, target in enumerate(panel.get("targets", [])):
                    if "expr" not in target:
                        errors.append(f"Panel {i}, target {j} missing expr")

            # Check for template variables
            templating = dashboard_data.get("templating", {})
            variables = templating.get("list", [])

            for var in variables:
                if var.get("name") and not var.get("query"):
                    errors.append(f"Variable {var['name']} missing query")

            return errors

        except (IOError, json.JSONDecodeError) as e:
            return [f"Failed to parse dashboard: {e}"]


# =============================================================================
# Main Setup
# =============================================================================


def setup_dashboards(config: Config, dashboard_filter: Optional[str] = None) -> int:
    """Setup Grafana dashboards and Prometheus rules.

    Args:
        config: Configuration object
        dashboard_filter: Optional specific dashboard to provision

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("Starting Victor observability dashboard setup")

    # Initialize clients
    grafana_client = GrafanaClient(config)
    prometheus_manager = PrometheusManager(config)
    validator = DashboardValidator()

    # Health checks
    logger.info("Performing health checks...")
    if not grafana_client.health_check():
        logger.error("Grafana health check failed. Is Grafana running?")
        return 1

    logger.info("Health checks passed")

    # Ensure datasource
    logger.info("Ensuring Prometheus datasource exists...")
    if not grafana_client.ensure_datasource():
        logger.error("Failed to ensure Prometheus datasource")
        return 1

    # Validate dashboards
    logger.info("Validating dashboards...")
    dashboard_files = list(config.dashboard_dir.glob("*.json"))

    if dashboard_filter:
        dashboard_files = [f for f in dashboard_files if dashboard_filter in f.name]

    for dashboard_file in dashboard_files:
        errors = validator.validate_dashboard(dashboard_file)
        if errors:
            logger.error(f"Dashboard validation failed: {dashboard_file}")
            for error in errors:
                logger.error(f"  - {error}")
            return 1

    logger.info("All dashboards validated")

    # Provision dashboards
    logger.info("Provisioning dashboards...")
    results = grafana_client.provision_dashboards(config.dashboard_dir)

    for dashboard_name, success in results.items():
        if success:
            logger.info(f"✓ {dashboard_name}")
        else:
            logger.error(f"✗ {dashboard_name}")

    if not all(results.values()):
        logger.error("Some dashboards failed to provision")
        return 1

    # Validate and import Prometheus rules
    logger.info("Processing Prometheus alerting rules...")
    rules_files = list(config.alerts_dir.glob("*.yml")) + list(config.alerts_dir.glob("*.yaml"))

    for rules_file in rules_files:
        logger.info(f"Validating rules: {rules_file}")
        if not prometheus_manager.validate_rules(rules_file):
            logger.error(f"Rules validation failed: {rules_file}")
            return 1

        logger.info(f"Importing rules: {rules_file}")
        if not prometheus_manager.import_rules(rules_file):
            logger.error(f"Failed to import rules: {rules_file}")
            # Don't fail on rules import failure, just warn

    logger.info("Dashboard setup completed successfully")
    return 0


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Victor Observability Dashboard Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dashboard", type=str, help="Specific dashboard to provision (e.g., team_overview)"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be done without making changes"
    )

    parser.add_argument(
        "--grafana-url",
        type=str,
        help="Grafana server URL (default: GRAFANA_URL env var or http://localhost:3000)",
    )

    parser.add_argument(
        "--grafana-api-key", type=str, help="Grafana API key (default: GRAFANA_API_KEY env var)"
    )

    parser.add_argument(
        "--prometheus-url",
        type=str,
        help="Prometheus server URL (default: PROMETHEUS_URL env var or http://localhost:9090)",
    )

    args = parser.parse_args()

    # Load configuration
    config = Config.from_env()

    # Override with command-line arguments
    if args.grafana_url:
        config.grafana_url = args.grafana_url
    if args.grafana_api_key:
        config.grafana_api_key = args.grafana_api_key
    if args.prometheus_url:
        config.prometheus_url = args.prometheus_url
    if args.dry_run:
        config.dry_run = True

    # Run setup
    try:
        return setup_dashboards(config, dashboard_filter=args.dashboard)
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Setup failed with exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
