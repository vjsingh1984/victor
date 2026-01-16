#!/usr/bin/env python3
"""Example script demonstrating complete dashboard setup workflow.

This example shows how to:
1. Configure environment variables
2. Validate Prometheus and Grafana connectivity
3. Provision dashboards via API
4. Configure alerting rules
5. Perform health checks

Usage:
    cd /path/to/victor
    python examples/observability/dashboard_setup.py

Prerequisites:
    - Grafana running on http://localhost:3000
    - Prometheus running on http://localhost:9090
    - Victor metrics enabled (http://localhost:8000/metrics)
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional
import requests
import json

# Add victor to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class DashboardConfig:
    """Configuration for dashboard setup."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        self.grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
        self.grafana_api_key = os.getenv("GRAFANA_API_KEY")
        self.prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
        self.victor_metrics_url = os.getenv("VICTOR_METRICS_URL", "http://localhost:8000")

        # Paths
        self.repo_root = Path(__file__).parent.parent.parent
        self.dashboards_dir = self.repo_root / "observability" / "dashboards"
        self.alerts_dir = self.repo_root / "observability" / "alerts"

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if configuration is valid
        """
        if not self.grafana_api_key:
            logger.warning(
                "GRAFANA_API_KEY not set. Dashboard creation may fail if "
                "Grafana requires authentication."
            )
            logger.info(
                "To create an API key:\n"
                "1. Open Grafana: %s\n"
                "2. Go to Configuration → API Keys\n"
                "3. Create new key with Admin role",
                self.grafana_url
            )

        if not self.dashboards_dir.exists():
            logger.error(f"Dashboards directory not found: {self.dashboards_dir}")
            return False

        if not self.alerts_dir.exists():
            logger.error(f"Alerts directory not found: {self.alerts_dir}")
            return False

        return True


# =============================================================================
# Health Checks
# =============================================================================

class HealthChecker:
    """Health checker for monitoring stack."""

    def __init__(self, config: DashboardConfig):
        """Initialize health checker.

        Args:
            config: Dashboard configuration
        """
        self.config = config

    def check_prometheus(self) -> bool:
        """Check Prometheus health.

        Returns:
            True if Prometheus is healthy
        """
        try:
            response = requests.get(
                f"{self.config.prometheus_url}/-/healthy",
                timeout=5
            )
            if response.status_code == 200:
                logger.info("✓ Prometheus is healthy")
                return True
            else:
                logger.error(f"✗ Prometheus health check failed: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"✗ Prometheus connection failed: {e}")
            logger.error(f"  Ensure Prometheus is running on {self.config.prometheus_url}")
            return False

    def check_grafana(self) -> bool:
        """Check Grafana health.

        Returns:
            True if Grafana is healthy
        """
        try:
            response = requests.get(
                f"{self.config.grafana_url}/api/health",
                timeout=5
            )
            if response.status_code == 200:
                logger.info("✓ Grafana is healthy")
                return True
            else:
                logger.error(f"✗ Grafana health check failed: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"✗ Grafana connection failed: {e}")
            logger.error(f"  Ensure Grafana is running on {self.config.grafana_url}")
            return False

    def check_victor_metrics(self) -> bool:
        """Check Victor metrics endpoint.

        Returns:
            True if metrics are available
        """
        try:
            response = requests.get(
                f"{self.config.victor_metrics_url}/metrics",
                timeout=5
            )
            if response.status_code == 200:
                # Check for team metrics
                if "victor_teams_executed_total" in response.text:
                    logger.info("✓ Victor metrics are available")
                    return True
                else:
                    logger.warning(
                        "✗ Victor metrics endpoint exists but team metrics not found. "
                        "Enable team metrics collection."
                    )
                    return False
            else:
                logger.error(f"✗ Victor metrics endpoint returned: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"✗ Victor metrics connection failed: {e}")
            logger.error(f"  Ensure Victor is running on {self.config.victor_metrics_url}")
            return False

    def check_prometheus_scrape_config(self) -> bool:
        """Check if Prometheus is configured to scrape Victor metrics.

        Returns:
            True if scrape target exists
        """
        try:
            response = requests.get(
                f"{self.config.prometheus_url}/api/v1/targets",
                timeout=5
            )
            if response.status_code == 200:
                targets = response.json().get("data", {}).get("activeTargets", [])
                for target in targets:
                    if self.config.victor_metrics_url in target.get("scrapeUrl", ""):
                        health = target.get("health", "unknown")
                        if health == "up":
                            logger.info(f"✓ Prometheus is scraping Victor metrics (health: {health})")
                            return True
                        else:
                            logger.warning(
                                f"✗ Prometheus scrape target exists but health is '{health}'"
                            )
                            return False

                logger.warning(
                    "✗ Prometheus is not configured to scrape Victor metrics. "
                    f"Add scrape target for {self.config.victor_metrics_url}"
                )
                return False
            else:
                logger.error("✗ Failed to query Prometheus targets")
                return False
        except requests.RequestException as e:
            logger.error(f"✗ Failed to check Prometheus scrape config: {e}")
            return False

    def run_all(self) -> bool:
        """Run all health checks.

        Returns:
            True if all checks pass
        """
        logger.info("=" * 60)
        logger.info("Running health checks...")
        logger.info("=" * 60)

        checks = [
            self.check_prometheus(),
            self.check_grafana(),
            self.check_victor_metrics(),
            self.check_prometheus_scrape_config(),
        ]

        all_passed = all(checks)

        if all_passed:
            logger.info("=" * 60)
            logger.info("All health checks passed ✓")
            logger.info("=" * 60)
        else:
            logger.warning("=" * 60)
            logger.warning("Some health checks failed. Please fix issues before proceeding.")
            logger.warning("=" * 60)

        return all_passed


# =============================================================================
# Dashboard Provisioning
# =============================================================================

class DashboardProvisioner:
    """Provision Grafana dashboards."""

    def __init__(self, config: DashboardConfig):
        """Initialize provisioner.

        Args:
            config: Dashboard configuration
        """
        self.config = config
        self.session = requests.Session()
        if config.grafana_api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {config.grafana_api_key}",
                "Content-Type": "application/json"
            })

    def ensure_datasource(self) -> bool:
        """Ensure Prometheus datasource exists in Grafana.

        Returns:
            True if datasource exists or was created
        """
        logger.info("Ensuring Prometheus datasource...")

        try:
            # Check existing datasources
            response = self.session.get(f"{self.config.grafana_url}/api/datasources")
            response.raise_for_status()

            datasources = response.json()
            for ds in datasources:
                if ds.get("name") == "Prometheus" and ds.get("type") == "prometheus":
                    logger.info(f"✓ Prometheus datasource already exists: {ds.get('url')}")
                    return True

            # Create datasource
            logger.info(f"Creating Prometheus datasource: {self.config.prometheus_url}")

            payload = {
                "name": "Prometheus",
                "type": "prometheus",
                "access": "proxy",
                "url": self.config.prometheus_url,
                "isDefault": True,
                "jsonData": {
                    "httpMethod": "POST"
                }
            }

            response = self.session.post(
                f"{self.config.grafana_url}/api/datasources",
                json=payload
            )
            response.raise_for_status()

            logger.info("✓ Prometheus datasource created")
            return True

        except requests.RequestException as e:
            logger.error(f"✗ Failed to ensure datasource: {e}")
            return False

    def provision_dashboard(self, dashboard_path: Path) -> bool:
        """Provision a single dashboard.

        Args:
            dashboard_path: Path to dashboard JSON file

        Returns:
            True if dashboard was provisioned
        """
        try:
            with open(dashboard_path, "r") as f:
                dashboard_data = json.load(f)

            dashboard_uid = dashboard_data.get("uid")
            dashboard_title = dashboard_data.get("title")

            if not dashboard_uid:
                logger.error(f"✗ Dashboard missing UID: {dashboard_path}")
                return False

            logger.info(f"Provisioning dashboard: {dashboard_title} (uid: {dashboard_uid})")

            payload = {
                "overwrite": True,
                "dashboard": dashboard_data
            }

            response = self.session.post(
                f"{self.config.grafana_url}/api/dashboards/db",
                json=payload
            )
            response.raise_for_status()

            logger.info(f"✓ Dashboard provisioned: {dashboard_title}")
            return True

        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"✗ Failed to read dashboard {dashboard_path}: {e}")
            return False
        except requests.RequestException as e:
            logger.error(f"✗ Failed to provision dashboard: {e}")
            return False

    def provision_all(self) -> bool:
        """Provision all dashboards.

        Returns:
            True if all dashboards were provisioned
        """
        logger.info("=" * 60)
        logger.info("Provisioning dashboards...")
        logger.info("=" * 60)

        # Ensure datasource
        if not self.ensure_datasource():
            return False

        # Provision dashboards
        dashboard_files = list(self.config.dashboards_dir.glob("*.json"))
        if not dashboard_files:
            logger.error(f"✗ No dashboard files found in {self.config.dashboards_dir}")
            return False

        results = {}
        for dashboard_file in dashboard_files:
            results[dashboard_file.name] = self.provision_dashboard(dashboard_file)

        all_success = all(results.values())

        if all_success:
            logger.info("=" * 60)
            logger.info(f"✓ All {len(results)} dashboards provisioned successfully")
            logger.info("=" * 60)
            logger.info(f"Access dashboards at: {self.config.grafana_url}")
        else:
            failed = [name for name, success in results.items() if not success]
            logger.error(f"✗ {len(failed)} dashboards failed to provision: {failed}")

        return all_success


# =============================================================================
# Main Setup
# =============================================================================

def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Victor Observability Dashboard Setup Example")
    logger.info("=" * 60)
    logger.info("")

    # Load configuration
    config = DashboardConfig()

    logger.info("Configuration:")
    logger.info(f"  Grafana URL: {config.grafana_url}")
    logger.info(f"  Prometheus URL: {config.prometheus_url}")
    logger.info(f"  Victor Metrics URL: {config.victor_metrics_url}")
    logger.info(f"  Dashboards Directory: {config.dashboards_dir}")
    logger.info(f"  Alerts Directory: {config.alerts_dir}")
    logger.info("")

    # Validate configuration
    if not config.validate():
        logger.error("Configuration validation failed")
        return 1

    # Run health checks
    health_checker = HealthChecker(config)
    if not health_checker.run_all():
        logger.error("")
        logger.error("Health checks failed. Please fix issues before proceeding.")
        logger.error("")
        logger.error("Common fixes:")
        logger.error("  1. Start Grafana: sudo systemctl start grafana-server")
        logger.error("  2. Start Prometheus: sudo systemctl start prometheus")
        logger.error("  3. Start Victor: victor chat --no-tui")
        logger.error("  4. Configure Prometheus scrape target:")
        logger.error("     # Add to /etc/prometheus/prometheus.yml:")
        logger.error(f"     scrape_configs:")
        logger.error(f"       - job_name: 'victor'")
        logger.error(f"         static_configs:")
        logger.error(f"           - targets: ['{config.victor_metrics_url}']")
        logger.error("     # Reload: sudo killall -HUP prometheus")
        return 1

    logger.info("")

    # Provision dashboards
    provisioner = DashboardProvisioner(config)
    if not provisioner.provision_all():
        logger.error("")
        logger.error("Dashboard provisioning failed")
        return 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("Setup completed successfully!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Open Grafana: {config.grafana_url}")
    logger.info("  2. Navigate to Dashboards → Victor")
    logger.info("  3. Explore the dashboards:")
    logger.info("     - Team Overview (victor-team-overview)")
    logger.info("     - Team Performance (victor-team-performance)")
    logger.info("     - Team Recursion (victor-team-recursion)")
    logger.info("     - Team Members (victor-team-members)")
    logger.info("")
    logger.info("To configure Prometheus alerting rules:")
    logger.info("  1. Copy alerting rules:")
    logger.info(f"     sudo cp {config.alerts_dir}/team_alerts.yml /etc/prometheus/rules/")
    logger.info("  2. Add to prometheus.yml:")
    logger.info("     rule_files:")
    logger.info("       - '/etc/prometheus/rules/team_alerts.yml'")
    logger.info("  3. Reload Prometheus:")
    logger.info("     sudo killall -HUP prometheus")
    logger.info("")
    logger.info("For more information, see: docs/observability/dashboards.md")
    logger.info("")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nSetup interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Setup failed with exception: {e}")
        sys.exit(1)
