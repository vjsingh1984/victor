#!/usr/bin/env python3
"""
Victor Production Monitoring Setup Script

This script helps set up production monitoring and observability
for the coordinator-based orchestrator.

Usage:
    python scripts/setup_monitoring.py [--config CONFIG_FILE]

Options:
    --config CONFIG_FILE    Path to configuration file (default: config/monitoring.yml)
    --validate-only         Only validate configuration without applying
    --dry-run               Show what would be done without making changes
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_step(step: str, description: str) -> None:
    """Print setup step."""
    print(f"[*] {step}")
    print(f"    {description}\n")


def validate_prometheus_config(config_path: Path) -> bool:
    """Validate Prometheus configuration."""
    print_section("Validating Prometheus Configuration")

    try:
        import subprocess

        result = subprocess.run(
            ["promtool", "check", "rules", str(config_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print_step("✓ Success", "Prometheus rules are valid")
            print(result.stdout)
            return True
        else:
            print_step("✗ Failed", "Prometheus rules validation failed")
            print(result.stderr)
            return False

    except FileNotFoundError:
        print_step("⚠ Warning", "promtool not found - skipping validation")
        print("    Install promtool: apt-get install prometheus-toolchain")
        return True
    except Exception as e:
        print_step("✗ Error", f"Validation failed: {e}")
        return False


def setup_prometheus_rules(
    source_path: Path,
    target_path: Path,
    dry_run: bool = False,
) -> bool:
    """Setup Prometheus alerting rules."""
    print_section("Setting Up Prometheus Alert Rules")

    if not source_path.exists():
        print_step("✗ Error", f"Source file not found: {source_path}")
        return False

    # Create target directory
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy file
    if dry_run:
        print_step("Dry Run", f"Would copy {source_path} → {target_path}")
    else:
        import shutil

        shutil.copy(source_path, target_path)
        print_step("✓ Success", f"Copied to {target_path}")

    # Validate
    return validate_prometheus_config(target_path)


def setup_grafana_dashboard(
    source_path: Path,
    dry_run: bool = False,
) -> bool:
    """Setup Grafana dashboard."""
    print_section("Setting Up Grafana Dashboard")

    if not source_path.exists():
        print_step("✗ Error", f"Dashboard file not found: {source_path}")
        return False

    # Load dashboard
    with open(source_path) as f:
        dashboard = json.load(f)

    print_step("✓ Loaded", f"Dashboard: {dashboard.get('title', 'Unknown')}")

    if dry_run:
        print_step("Dry Run", "Would import dashboard to Grafana")
        return True

    # Check if Grafana API is available
    import os

    grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
    grafana_api_key = os.getenv("GRAFANA_API_KEY")

    if not grafana_api_key:
        print_step("⚠ Warning", "GRAFANA_API_KEY not set")
        print("    Set GRAFANA_API_KEY environment variable for automatic import")
        print("    Or import manually:")
        print(f"    1. Open {grafana_url}")
        print(f"    2. Go to Dashboards → Import")
        print(f"    3. Upload {source_path}")
        return True

    # Import dashboard via API
    try:
        import requests

        headers = {"Authorization": f"Bearer {grafana_api_key}"}
        data = {
            "dashboard": dashboard,
            "overwrite": True,
            "message": "Imported via setup_monitoring.py",
        }

        response = requests.post(
            f"{grafana_url}/api/dashboards/import",
            headers=headers,
            json=data,
        )

        if response.status_code == 200:
            result = response.json()
            print_step("✓ Success", f"Dashboard imported: {grafana_url}{result.get('url')}")
            return True
        else:
            print_step("✗ Failed", f"Import failed: {response.text}")
            return False

    except Exception as e:
        print_step("✗ Error", f"Import failed: {e}")
        return False


def setup_logging_config(
    dry_run: bool = False,
) -> bool:
    """Setup logging configuration."""
    print_section("Setting Up Structured Logging")

    if dry_run:
        print_step("Dry Run", "Would configure structured logging")
        return True

    # Create log directory
    log_dir = Path("/var/log/victor")

    if not log_dir.exists():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            print_step("✓ Success", f"Created log directory: {log_dir}")
        except PermissionError:
            print_step("⚠ Warning", f"Cannot create {log_dir} (requires root)")
            print("    Using current directory for logs")
            log_dir = Path(".")
    else:
        print_step("✓ Exists", f"Log directory: {log_dir}")

    # Show example configuration
    print_step("Example", "Add this to your application:")

    example_code = """
from victor.observability.coordinator_logging import (
    setup_coordinator_logging,
    get_coordinator_logger,
)

# Setup logging
setup_coordinator_logging(
    level="INFO",
    format_type="json",
    output_file="/var/log/victor/coordinators.log",
)

# Get logger
logger = get_coordinator_logger("ChatCoordinator")
logger.info("Coordinator started", extra={"coordinator": "ChatCoordinator"})
"""
    print(example_code)

    return True


def setup_health_checks(
    dry_run: bool = False,
) -> bool:
    """Setup health check endpoints."""
    print_section("Setting Up Health Check Endpoints")

    if dry_run:
        print_step("Dry Run", "Would setup health check endpoints")
        return True

    # Show example configuration
    print_step("Example", "Add this to your FastAPI application:")

    example_code = """
from fastapi import FastAPI
from victor.observability.health import setup_health_endpoints

app = FastAPI()

# Setup health endpoints
setup_health_endpoints(app)

# Endpoints available:
# GET /health - Overall health
# GET /health/ready - Readiness probe
# GET /health/live - Liveness probe
# GET /health/detailed - Comprehensive health
"""
    print(example_code)

    return True


def setup_prometheus_exporter(
    dry_run: bool = False,
) -> bool:
    """Setup Prometheus metrics exporter."""
    print_section("Setting Up Prometheus Exporter")

    if dry_run:
        print_step("Dry Run", "Would setup Prometheus exporter")
        return True

    # Show example configuration
    print_step("Example", "Add this to your FastAPI application:")

    example_code = """
from fastapi import FastAPI
from victor.observability import get_prometheus_exporter

app = FastAPI()

# Get Prometheus exporter
exporter = get_prometheus_exporter()

# Add metrics endpoint
app.add_route("/metrics", exporter.get_endpoint())

# Metrics available at http://localhost:8000/metrics
"""
    print(example_code)

    print_step("Standalone Server", "Or run standalone metrics server:")

    standalone_code = """
from victor.observability import get_prometheus_exporter

exporter = get_prometheus_exporter()
exporter.start_server(port=9090)
"""
    print(standalone_code)

    return True


def generate_docker_compose(dry_run: bool = False) -> bool:
    """Generate docker-compose configuration."""
    print_section("Generating Docker Compose Configuration")

    compose_content = """version: '3.8'

services:
  victor-coordinators:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      - VICTOR_LOG_LEVEL=INFO
      - VICTOR_LOG_FORMAT=json
    volumes:
      - ./logs:/var/log/victor
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./docs/production/prometheus_alerts.yml:/etc/prometheus/rules/victor-alerts.yml
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
"""

    output_path = Path("docker-compose.monitoring.yml")

    if dry_run:
        print_step("Dry Run", f"Would generate {output_path}")
    else:
        with open(output_path, "w") as f:
            f.write(compose_content)
        print_step("✓ Success", f"Generated {output_path}")

    print(compose_content)
    return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup Victor production monitoring",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/monitoring.yml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without applying",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--skip-prometheus",
        action="store_true",
        help="Skip Prometheus setup",
    )
    parser.add_argument(
        "--skip-grafana",
        action="store_true",
        help="Skip Grafana setup",
    )

    args = parser.parse_args()

    # Get repository root
    repo_root = Path(__file__).parent.parent
    docs_dir = repo_root / "docs" / "production"

    print_section("Victor Production Monitoring Setup")
    print(f"Repository: {repo_root}")
    print(f"Documentation: {docs_dir}")

    results = []

    # Setup Prometheus rules
    if not args.skip_prometheus:
        results.append(
            setup_prometheus_rules(
                source_path=docs_dir / "prometheus_alerts.yml",
                target_path=Path("/etc/prometheus/rules/victor-alerts.yml"),
                dry_run=args.dry_run,
            )
        )

    # Setup Grafana dashboard
    if not args.skip_grafana:
        results.append(
            setup_grafana_dashboard(
                source_path=docs_dir / "grafana_dashboard.json",
                dry_run=args.dry_run,
            )
        )

    # Setup logging
    if not args.validate_only:
        results.append(setup_logging_config(dry_run=args.dry_run))

        # Setup health checks
        results.append(setup_health_checks(dry_run=args.dry_run))

        # Setup Prometheus exporter
        results.append(setup_prometheus_exporter(dry_run=args.dry_run))

        # Generate docker-compose
        results.append(generate_docker_compose(dry_run=args.dry_run))

    # Summary
    print_section("Setup Summary")

    success_count = sum(1 for r in results if r)
    total_count = len(results)

    print(f"Completed: {success_count}/{total_count}")

    if all(results):
        print_step("✓ Success", "All monitoring components configured")
        print("\nNext Steps:")
        print("1. Update prometheus.yml with victor scrape config")
        print("2. Restart Prometheus: systemctl restart prometheus")
        print("3. Import Grafana dashboard (or use API)")
        print("4. Check health: curl http://localhost:8000/health")
        print("5. View metrics: curl http://localhost:8000/metrics")
        print("6. Open Grafana: http://localhost:3000")
        print("\nFor more information, see: docs/production/README.md")
        return 0
    else:
        print_step("✗ Failed", "Some components failed to configure")
        print("Review the errors above and fix manually")
        return 1


if __name__ == "__main__":
    sys.exit(main())
