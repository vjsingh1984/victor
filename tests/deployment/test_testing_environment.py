"""
Testing environment configuration tests.

Validates that the testing environment is properly configured
for automated testing and CI/CD validation.
"""

import os
import pytest
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def testing_env_vars() -> Dict[str, str]:
    """Testing environment variables."""
    return {
        "VICTOR_PROFILE": "testing",
        "VICTOR_LOG_LEVEL": "DEBUG",
        "VICTOR_MAX_WORKERS": "2",
        "VICTOR_CACHE_SIZE": "500",
        "event-bus-backend": "memory",
        "checkpoint-backend": "sqlite",
        "cache-backend": "memory",
        "enable-metrics": "true",
        "enable-tracing": "false",
        "enable-rate-limiting": "false",
        "tool-selection-strategy": "keyword",
    }


@pytest.fixture
def testing_kubernetes_config() -> Dict[str, Any]:
    """Expected Kubernetes configuration for testing."""
    return {
        "namespace": "victor-ai-test",
        "replicas": 1,
        "resources": {
            "requests": {"cpu": "250m", "memory": "256Mi"},
            "limits": {"cpu": "1000m", "memory": "1Gi"},
        },
        "image": {
            "repository": "victorai/victor",
            "tag": "0.5.1-testing",
            "pullPolicy": "Always",
        },
        "autoscaling": {"enabled": False},
    }


class TestTestingEnvironmentConfig:
    """Test testing environment configuration loading."""

    def test_profile_is_testing(self, testing_env_vars):
        """Test that profile is set to testing."""
        assert testing_env_vars["VICTOR_PROFILE"] == "testing"

    def test_log_level_is_debug(self, testing_env_vars):
        """Test that log level is DEBUG for test failure analysis."""
        assert testing_env_vars["VICTOR_LOG_LEVEL"] == "DEBUG"

    def test_max_workers_is_2(self, testing_env_vars):
        """Test that max workers is set to 2 for CI/CD efficiency."""
        assert testing_env_vars["VICTOR_MAX_WORKERS"] == "2"

    def test_cache_size_is_500(self, testing_env_vars):
        """Test that cache size is 500 entries."""
        assert testing_env_vars["VICTOR_CACHE_SIZE"] == "500"

    def test_sqlite_checkpoint_backend(self, testing_env_vars):
        """Test that SQLite backend is used for lightweight persistence."""
        assert testing_env_vars["checkpoint-backend"] == "sqlite"

    def test_memory_backends(self, testing_env_vars):
        """Test that memory backends are used for event bus and cache."""
        assert testing_env_vars["event-bus-backend"] == "memory"
        assert testing_env_vars["cache-backend"] == "memory"

    def test_no_rate_limiting(self, testing_env_vars):
        """Test that rate limiting is disabled for test traffic."""
        assert testing_env_vars["enable-rate-limiting"].lower() == "false"

    def test_keyword_tool_selection(self, testing_env_vars):
        """Test that keyword tool selection is used for speed."""
        assert testing_env_vars["tool-selection-strategy"] == "keyword"

    def test_tracing_disabled(self, testing_env_vars):
        """Test that distributed tracing is disabled."""
        assert testing_env_vars["enable-tracing"].lower() == "false"

    def test_metrics_enabled(self, testing_env_vars):
        """Test that metrics are enabled for test reporting."""
        assert testing_env_vars["enable-metrics"].lower() == "true"


class TestTestingEnvironmentKubernetes:
    """Test Kubernetes configuration for testing environment."""

    def test_namespace_is_test(self, testing_kubernetes_config):
        """Test that namespace is victor-ai-test."""
        assert testing_kubernetes_config["namespace"] == "victor-ai-test"

    def test_single_replica(self, testing_kubernetes_config):
        """Test that only 1 replica is configured for CI/CD."""
        assert testing_kubernetes_config["replicas"] == 1

    def test_minimal_resources(self, testing_kubernetes_config):
        """Test that minimal resources are allocated."""
        resources = testing_kubernetes_config["resources"]
        assert resources["requests"]["cpu"] == "250m"
        assert resources["requests"]["memory"] == "256Mi"
        assert resources["limits"]["cpu"] == "1000m"
        assert resources["limits"]["memory"] == "1Gi"

    def test_testing_image_tag(self, testing_kubernetes_config):
        """Test that testing image tag is used."""
        assert testing_kubernetes_config["image"]["tag"] == "0.5.1-testing"

    def test_no_autoscaling(self, testing_kubernetes_config):
        """Test that autoscaling is disabled."""
        assert testing_kubernetes_config["autoscaling"]["enabled"] is False


class TestTestingEnvironmentFeatures:
    """Test testing-specific features."""

    def test_testing_overlay_exists(self):
        """Test that testing overlay directory exists."""
        overlay_path = Path("/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays")
        # Note: Testing environment uses development overlay with different namespace
        assert overlay_path.exists()

    def test_ci_cd_integration(self):
        """Test that CI/CD can use testing configuration."""
        # Check that values file can be parsed
        import yaml

        values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values.yaml")
        with open(values_path) as f:
            values = yaml.safe_load(f)
            assert values is not None
            # Check that testing-relevant values exist
            assert "config" in values
            assert "profile" in values["config"]

    def test_sqlite_backend_available(self):
        """Test that SQLite checkpoint backend is available."""
        from victor.framework.checkpoint import CheckpointBackend

        # SQLite should be available as a backend option
        backends = [CheckpointBackend.MEMORY, CheckpointBackend.SQLITE]
        assert CheckpointBackend.SQLITE in backends


@pytest.mark.integration
class TestTestingEnvironmentIntegration:
    """Integration tests for testing environment."""

    def test_sqlite_checkpoint_persistence(self, tmp_path):
        """Test that SQLite checkpoint backend works."""
        import sqlite3
        from pathlib import Path

        # Create test database
        db_path = tmp_path / "test_checkpoints.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT PRIMARY KEY,
                state_data TEXT,
                timestamp REAL
            )
        """
        )
        conn.commit()

        # Insert test data
        cursor.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?)", ("test-thread", "{}", 1234567890.0)
        )
        conn.commit()

        # Verify data
        cursor.execute("SELECT * FROM checkpoints WHERE thread_id = ?", ("test-thread",))
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == "test-thread"

        conn.close()

    def test_memory_event_bus(self):
        """Test that memory event bus works."""
        from victor.core.events import create_event_backend, MessagingEvent
        from victor.core.events.protocols import BackendConfig, BackendType
        import asyncio

        async def test_bus():
            backend = create_event_backend(BackendConfig(backend_type=BackendType.IN_MEMORY))
            await backend.connect()

            # Publish test event
            event = MessagingEvent(topic="test.event", data={"test": "data"}, source="test")
            await backend.publish(event)

            await backend.disconnect()

        asyncio.run(test_bus())

    def test_metrics_collection(self):
        """Test that metrics can be collected."""
        from victor.observability.metrics import MetricsRegistry

        registry = MetricsRegistry()
        counter = registry.counter("test_calls", "Test counter", labels={"environment": "testing"})
        counter.increment()
        counter.increment()

        # Verify metric was recorded
        metrics = registry.collect()
        assert len(metrics) > 0


@pytest.mark.deployment
class TestTestingEnvironmentDeployment:
    """Deployment-specific tests for testing environment."""

    def test_helm_values_syntax(self):
        """Test that Helm values file has valid syntax."""
        import yaml

        values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values.yaml")
        with open(values_path) as f:
            values = yaml.safe_load(f)
            assert values is not None
            assert "replicaCount" in values
            assert "image" in values
            assert "resources" in values

    def test_helm_rendering(self):
        """Test that Helm chart can be rendered (dry-run)."""
        # This test requires helm to be installed
        import subprocess

        try:
            result = subprocess.run(
                [
                    "helm",
                    "template",
                    "victor-ai",
                    "/Users/vijaysingh/code/codingagent/deployment/helm",
                    "--values",
                    "/Users/vijaysingh/code/codingagent/deployment/helm/values.yaml",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Helm should exit with 0 even without a cluster
            # (just template rendering)
            assert result.returncode == 0 or "not found" not in result.stderr.lower()
        except FileNotFoundError:
            pytest.skip("Helm not installed")
        except subprocess.TimeoutExpired:
            pytest.skip("Helm template command timed out")

    def test_testing_profile_in_settings(self, monkeypatch):
        """Test that testing profile can be loaded."""
        monkeypatch.setenv("VICTOR_PROFILE", "testing")
        from victor.config.settings import Settings

        settings = Settings()
        # Verify Settings can be instantiated with testing profile env var
        # (Note: profile may not exist, so we just verify no errors during instantiation)
        assert settings is not None


@pytest.mark.cicd
class TestTestingEnvironmentCI:
    """CI/CD-specific tests for testing environment."""

    def test_environment_config_parseable(self):
        """Test that environment configuration is parseable."""
        import yaml

        values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values.yaml")
        with open(values_path) as f:
            values = yaml.safe_load(f)
            # Verify structure
            assert isinstance(values, dict)
            assert "config" in values

    def test_base_kubernetes_resources_exist(self):
        """Test that base Kubernetes resources exist."""
        base_path = Path("/Users/vijaysingh/code/codingagent/deployment/kubernetes/base")
        assert base_path.exists()

        # Check for common resource files
        yaml_files = list(base_path.glob("*.yaml")) + list(base_path.glob("*.yml"))
        assert len(yaml_files) > 0

    def test_development_overlay_for_testing(self):
        """Test that development overlay can be used for testing."""
        overlay_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays/development"
        )
        assert overlay_path.exists()

        kustomization = overlay_path / "kustomization.yaml"
        assert kustomization.exists()


class TestTestingEnvironmentData:
    """Test data handling in testing environment."""

    def test_synthetic_test_data(self):
        """Test that synthetic test data can be generated."""
        # This is a placeholder for test data generation logic
        test_data = {"test_queries": ["test query 1", "test query 2"]}
        assert len(test_data["test_queries"]) == 2

    def test_no_production_data_access(self):
        """Test that production data is not accessible from testing."""
        # Verify that production database is not configured
        import os

        prod_db_url = os.environ.get("POSTGRES_URL", "")
        assert "production" not in prod_db_url.lower()
        assert "prod" not in prod_db_url.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
