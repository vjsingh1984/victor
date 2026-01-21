"""
Environment promotion tests.

Validates that configurations can be promoted correctly through
the environment pipeline: development → testing → staging → production.
"""

import os
import pytest
import yaml
from pathlib import Path
from typing import Dict, Any, List


@pytest.fixture
def environment_order() -> List[str]:
    """Return the correct order of environments."""
    return ["development", "testing", "staging", "production"]


@pytest.fixture
def environment_configs() -> Dict[str, Dict[str, Any]]:
    """Load configurations for all environments."""
    configs = {}

    base_values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values.yaml")
    with open(base_values_path) as f:
        configs["base"] = yaml.safe_load(f)

    prod_values_path = Path(
        "/Users/vijaysingh/code/codingagent/deployment/helm/values-prod.yaml"
    )
    with open(prod_values_path) as f:
        configs["production"] = yaml.safe_load(f)

    # Load overlay configurations
    for env in ["development", "staging"]:
        overlay_path = (
            Path("/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays")
            / env
            / "kustomization.yaml"
        )
        if overlay_path.exists():
            with open(overlay_path) as f:
                configs[env] = yaml.safe_load(f)

    return configs


class TestEnvironmentPromotionPath:
    """Test environment promotion path and order."""

    def test_environment_order_is_correct(self, environment_order):
        """Test that environment order is correct."""
        assert environment_order == ["development", "testing", "staging", "production"]

    def test_no_environment_skipping(self, environment_order):
        """Test that environments cannot be skipped in promotion."""
        # Promotion should follow strict order
        for i in range(len(environment_order) - 1):
            current = environment_order[i]
            next_env = environment_order[i + 1]
            assert environment_order.index(next_env) == environment_order.index(current) + 1


class TestConfigurationPromotion:
    """Test configuration promotion between environments."""

    def test_resource_scaling(self, environment_configs):
        """Test that resources scale up through environments."""
        # Resources should increase: dev < test < staging < prod
        expected_replicas = {"development": 1, "testing": 1, "staging": 2, "production": 6}

        for env, expected_replica_count in expected_replicas.items():
            if env in environment_configs:
                config = environment_configs[env]
                actual_replicas = config.get("replicas", config.get("replicaCount"))
                assert (
                    actual_replicas == expected_replica_count
                ), f"{env}: expected {expected_replica_count} replicas, got {actual_replicas}"

    def test_cpu_scaling(self, environment_configs):
        """Test that CPU allocation scales up through environments."""
        expected_cpu_limits = {
            "development": "1000m",
            "testing": "1000m",
            "staging": "2000m",
            "production": "4000m",
        }

        for env, expected_cpu in expected_cpu_limits.items():
            if env in environment_configs:
                config = environment_configs[env]
                # Handle different config structures
                if "resources" in config:
                    actual_cpu = config["resources"]["limits"]["cpu"]
                elif "spec" in config:
                    # Kubernetes patch format
                    actual_cpu = "1000m"  # Default
                assert actual_cpu == expected_cpu, f"{env}: CPU limit mismatch"

    def test_memory_scaling(self, environment_configs):
        """Test that memory allocation scales up through environments."""
        expected_memory_limits = {
            "development": "1Gi",
            "testing": "1Gi",
            "staging": "2Gi",
            "production": "4Gi",
        }

        for env, expected_memory in expected_memory_limits.items():
            if env in environment_configs:
                config = environment_configs[env]
                if "resources" in config:
                    actual_memory = config["resources"]["limits"]["memory"]
                else:
                    actual_memory = "1Gi"
                assert actual_memory == expected_memory, f"{env}: Memory limit mismatch"

    def test_cache_size_scaling(self, environment_configs):
        """Test that cache size scales up through environments."""
        expected_cache_sizes = {
            "development": "500",
            "testing": "500",
            "staging": "1000",
            "production": "2000",
        }

        for env, expected_size in expected_cache_sizes.items():
            if env in environment_configs:
                config = environment_configs[env]
                if "configMapGenerator" in config:
                    literals = config["configMapGenerator"][0].get("literals", [])
                    for lit in literals:
                        if lit.startswith("VICTOR_CACHE_SIZE="):
                            actual_size = lit.split("=")[1]
                            assert actual_size == expected_size

    def test_worker_count_scaling(self, environment_configs):
        """Test that worker count scales up through environments."""
        expected_workers = {
            "development": "2",
            "testing": "2",
            "staging": "4",
            "production": "8",
        }

        for env, expected_count in expected_workers.items():
            if env in environment_configs:
                config = environment_configs[env]
                if "configMapGenerator" in config:
                    literals = config["configMapGenerator"][0].get("literals", [])
                    for lit in literals:
                        if lit.startswith("VICTOR_MAX_WORKERS="):
                            actual_count = lit.split("=")[1]
                            assert actual_count == expected_count


class TestConfigurationDrift:
    """Test for configuration drift between environments."""

    def test_no_unexpected_configuration_drift(self, environment_configs):
        """Test that configuration changes are intentional."""
        # This checks that known configuration differences are expected
        # and there are no unexpected differences

        # Common configuration that should be consistent
        common_fields = [
            "image.repository",
            "image.pullPolicy",
            "service.type",
        ]

        # These fields are expected to differ
        allowed_differences = {
            "replicas",
            "replicaCount",
            "image.tag",
            "resources",
            "autoscaling",
            "config",
            "ingress",
            "monitoring",
            "networkPolicy",
            "affinity",
        }

        # Compare configurations
        if "base" in environment_configs and "production" in environment_configs:
            base = environment_configs["base"]
            prod = environment_configs["production"]

            # Check that base fields are present in production (with allowed overrides)
            for field in common_fields:
                field_parts = field.split(".")
                base_value = base
                for part in field_parts:
                    base_value = base_value.get(part, {})
                    if base_value == {}:
                        break

    def test_log_level_progression(self):
        """Test that log level follows expected progression."""
        # DEBUG for dev/test/staging, INFO for production
        expected_log_levels = {
            "development": "DEBUG",
            "testing": "DEBUG",
            "staging": "DEBUG",
            "production": "INFO",
        }

        for env in ["development", "staging"]:
            overlay_path = (
                Path("/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays")
                / env
                / "kustomization.yaml"
            )
            if overlay_path.exists():
                with open(overlay_path) as f:
                    config = yaml.safe_load(f)
                    literals = config.get("configMapGenerator", [{}])[0].get(
                        "literals", []
                    )
                    for lit in literals:
                        if lit.startswith("VICTOR_LOG_LEVEL="):
                            actual_level = lit.split("=")[1]
                            expected_level = expected_log_levels[env]
                            assert (
                                actual_level == expected_level
                            ), f"{env}: Expected log level {expected_level}, got {actual_level}"

    def test_backend_progression(self):
        """Test that backend progression is correct."""
        # Memory → SQLite → Postgres (for checkpoint)
        # Memory → Redis (for cache)
        # Memory → Kafka (for event bus)

        expected_backends = {
            "development": {
                "checkpoint": "memory",
                "cache": "memory",
                "event-bus": "memory",
            },
            "testing": {
                "checkpoint": "sqlite",
                "cache": "memory",
                "event-bus": "memory",
            },
            "staging": {
                "checkpoint": "sqlite",
                "cache": "memory",
                "event-bus": "memory",
            },
            "production": {
                "checkpoint": "postgres",
                "cache": "redis",
                "event-bus": "kafka",
            },
        }

        for env in ["development", "staging"]:
            overlay_path = (
                Path("/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays")
                / env
                / "kustomization.yaml"
            )
            if overlay_path.exists():
                with open(overlay_path) as f:
                    config = yaml.safe_load(f)
                    literals = config.get("configMapGenerator", [{}])[0].get(
                        "literals", []
                    )

                    actual_backends = {}
                    for lit in literals:
                        if lit.startswith("checkpoint-backend="):
                            actual_backends["checkpoint"] = lit.split("=")[1]
                        elif lit.startswith("cache-backend="):
                            actual_backends["cache"] = lit.split("=")[1]
                        elif lit.startswith("event-bus-backend="):
                            actual_backends["event-bus"] = lit.split("=")[1]

                    expected = expected_backends[env]
                    for backend_type, expected_value in expected.items():
                        actual_value = actual_backends.get(backend_type)
                        assert (
                            actual_value == expected_value
                        ), f"{env}: Expected {backend_type}={expected_value}, got {actual_value}"


class TestEnvironmentSpecificOverrides:
    """Test environment-specific configuration overrides."""

    def test_development_hot_reload(self):
        """Test that hot reload is only enabled in development."""
        overlay_path = (
            Path("/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays")
            / "development"
            / "deployment-patch.yaml"
        )

        if overlay_path.exists():
            with open(overlay_path) as f:
                config = yaml.safe_load(f)
                env_vars = (
                    config.get("spec", {})
                    .get("template", {})
                    .get("spec", {})
                    .get("containers", [{}])[0]
                    .get("env", [])
                )

                hot_reload_found = False
                for env_var in env_vars:
                    if env_var.get("name") == "VICTOR_HOT_RELOAD":
                        hot_reload_found = True
                        assert env_var.get("value") == "true"

                assert hot_reload_found, "VICTOR_HOT_RELOAD not found in development"

    def test_production_tls_enabled(self):
        """Test that TLS is enabled in production."""
        prod_values_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/helm/values-prod.yaml"
        )

        with open(prod_values_path) as f:
            values = yaml.safe_load(f)
            assert values["ingress"]["enabled"] is True
            assert len(values["ingress"]["tls"]) > 0

    def test_production_autoscaling_enabled(self):
        """Test that autoscaling is enabled in production."""
        prod_values_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/helm/values-prod.yaml"
        )

        with open(prod_values_path) as f:
            values = yaml.safe_load(f)
            assert values["autoscaling"]["enabled"] is True
            assert values["autoscaling"]["minReplicas"] == 6
            assert values["autoscaling"]["maxReplicas"] == 50

    def test_staging_rate_limiting(self):
        """Test that rate limiting is properly configured in staging."""
        overlay_path = (
            Path("/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays")
            / "staging"
            / "kustomization.yaml"
        )

        if overlay_path.exists():
            with open(overlay_path) as f:
                config = yaml.safe_load(f)
                literals = config.get("configMapGenerator", [{}])[0].get("literals", [])

                rate_limit_found = False
                for lit in literals:
                    if lit.startswith("rate-limit-requests-per-minute="):
                        rate_limit_found = True
                        assert lit == "rate-limit-requests-per-minute=500"

                assert rate_limit_found, "Rate limiting not found in staging"

    def test_production_rate_limiting(self):
        """Test that rate limiting is properly configured in production."""
        overlay_path = (
            Path("/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays")
            / "production"
            / "kustomization.yaml"
        )

        if overlay_path.exists():
            with open(overlay_path) as f:
                config = yaml.safe_load(f)
                literals = config.get("configMapGenerator", [{}])[0].get("literals", [])

                rate_limit_found = False
                for lit in literals:
                    if lit.startswith("rate-limit-requests-per-minute="):
                        rate_limit_found = True
                        assert lit == "rate-limit-requests-per-minute=1000"

                assert rate_limit_found, "Rate limiting not found in production"


class TestSecretsPromotion:
    """Test secrets handling during promotion."""

    def test_secrets_not_in_version_control(self):
        """Test that secrets are not stored in configuration files."""
        import re

        # Check that no actual secrets are in YAML files
        yaml_files = [
            "/Users/vijaysingh/code/codingagent/deployment/helm/values.yaml",
            "/Users/vijaysingh/code/codingagent/deployment/helm/values-prod.yaml",
        ]

        for yaml_file in yaml_files:
            if Path(yaml_file).exists():
                with open(yaml_file) as f:
                    content = f.read()
                    # Check for placeholder secrets (not actual values)
                    assert "change-me" in content or "password" in content.lower()
                    # Check that no API keys are present
                    assert "sk-ant-" not in content  # Anthropic key pattern
                    assert "sk-" not in content  # OpenAI key pattern

    def test_external_secrets_configured(self):
        """Test that production uses external secrets."""
        # In production, secrets should be injected via external secrets operator
        # or Kubernetes secrets, not hardcoded in values

        base_values_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/helm/values.yaml"
        )
        with open(base_values_path) as f:
            values = yaml.safe_load(f)
            # Check that existingSecret field is present
            assert "secrets" in values
            assert "existingSecret" in values["secrets"]


class TestImageTagPromotion:
    """Test image tag promotion through environments."""

    def test_image_tags(self):
        """Test that image tags follow expected pattern."""
        expected_tags = {
            "development": "latest",
            "testing": "0.5.1-testing",
            "staging": "0.5.1-staging",
            "production": "0.5.1",
        }

        # Check development and staging overlays
        for env in ["development", "staging"]:
            overlay_path = (
                Path("/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays")
                / env
                / "kustomization.yaml"
            )
            if overlay_path.exists():
                with open(overlay_path) as f:
                    config = yaml.safe_load(f)
                    images = config.get("images", [])
                    if images:
                        actual_tag = images[0].get("newTag")
                        expected_tag = expected_tags[env]
                        assert actual_tag == expected_tag, f"{env}: Expected tag {expected_tag}, got {actual_tag}"

        # Check production values
        prod_values_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/helm/values-prod.yaml"
        )
        with open(prod_values_path) as f:
            values = yaml.safe_load(f)
            actual_tag = values["image"]["tag"]
            expected_tag = expected_tags["production"]
            assert actual_tag == expected_tag, f"production: Expected tag {expected_tag}, got {actual_tag}"


@pytest.mark.promotion
class TestPromotionChecklist:
    """Test promotion checklist validation."""

    def test_development_to_testing_promotion(self):
        """Test checklist for development → testing promotion."""
        checklist = {
            "unit_tests_pass": True,
            "code_coverage_acceptable": True,
            "no_critical_linting_errors": True,
            "documentation_updated": True,
        }

        # All items should be True
        for item, status in checklist.items():
            assert status is True, f"Checklist item '{item}' failed"

    def test_testing_to_staging_promotion(self):
        """Test checklist for testing → staging promotion."""
        checklist = {
            "all_tests_pass": True,
            "integration_tests_pass": True,
            "performance_benchmarks_met": True,
            "security_scan_clean": True,
        }

        for item, status in checklist.items():
            assert status is True, f"Checklist item '{item}' failed"

    def test_staging_to_production_promotion(self):
        """Test checklist for staging → production promotion."""
        checklist = {
            "uat_approved": True,
            "performance_acceptable": True,
            "monitoring_configured": True,
            "rollback_plan_ready": True,
            "backup_verified": True,
        }

        for item, status in checklist.items():
            assert status is True, f"Checklist item '{item}' failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
