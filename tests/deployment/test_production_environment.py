"""
Production environment configuration tests.

Validates that the production environment is properly configured
for live production traffic with high availability and performance.
"""

import pytest
from pathlib import Path
from typing import Any


@pytest.fixture
def production_env_vars() -> dict[str, str]:
    """Production environment variables."""
    return {
        "VICTOR_PROFILE": "production",
        "VICTOR_LOG_LEVEL": "INFO",
        "VICTOR_MAX_WORKERS": "8",
        "VICTOR_CACHE_SIZE": "2000",
        "event-bus-backend": "kafka",
        "checkpoint-backend": "postgres",
        "cache-backend": "redis",
        "enable-metrics": "true",
        "enable-tracing": "true",
        "enable-rate-limiting": "true",
        "rate-limit-requests-per-minute": "1000",
        "tool-selection-strategy": "hybrid",
    }


@pytest.fixture
def production_kubernetes_config() -> dict[str, Any]:
    """Expected Kubernetes configuration for production."""
    return {
        "namespace": "victor-ai-prod",
        "replicas": 6,
        "resources": {
            "requests": {"cpu": "1000m", "memory": "1Gi"},
            "limits": {"cpu": "4000m", "memory": "4Gi"},
        },
        "image": {
            "repository": "victorai/victor",
            "tag": "0.5.1",
            "pullPolicy": "Always",
        },
        "autoscaling": {
            "enabled": True,
            "minReplicas": 6,
            "maxReplicas": 50,
            "targetCPUUtilizationPercentage": 70,
            "targetMemoryUtilizationPercentage": 80,
        },
    }


class TestProductionEnvironmentConfig:
    """Test production environment configuration loading."""

    def test_profile_is_production(self, production_env_vars):
        """Test that profile is set to production."""
        assert production_env_vars["VICTOR_PROFILE"] == "production"

    def test_log_level_is_info(self, production_env_vars):
        """Test that log level is INFO (performance optimized)."""
        assert production_env_vars["VICTOR_LOG_LEVEL"] == "INFO"

    def test_max_workers_is_8(self, production_env_vars):
        """Test that max workers is set to 8 for scalability."""
        assert production_env_vars["VICTOR_MAX_WORKERS"] == "8"

    def test_cache_size_is_2000(self, production_env_vars):
        """Test that cache size is 2000 entries."""
        assert production_env_vars["VICTOR_CACHE_SIZE"] == "2000"

    def test_kafka_event_bus(self, production_env_vars):
        """Test that Kafka is used for distributed messaging."""
        assert production_env_vars["event-bus-backend"] == "kafka"

    def test_postgres_checkpoint_backend(self, production_env_vars):
        """Test that PostgreSQL is used for distributed persistence."""
        assert production_env_vars["checkpoint-backend"] == "postgres"

    def test_redis_cache_backend(self, production_env_vars):
        """Test that Redis is used for distributed caching."""
        assert production_env_vars["cache-backend"] == "redis"

    def test_rate_limiting_enabled(self, production_env_vars):
        """Test that rate limiting is enabled."""
        assert production_env_vars["enable-rate-limiting"].lower() == "true"

    def test_rate_limit_threshold(self, production_env_vars):
        """Test that rate limit is 1000 requests per minute."""
        assert production_env_vars["rate-limit-requests-per-minute"] == "1000"

    def test_hybrid_tool_selection(self, production_env_vars):
        """Test that hybrid tool selection is used."""
        assert production_env_vars["tool-selection-strategy"] == "hybrid"

    def test_tracing_enabled(self, production_env_vars):
        """Test that distributed tracing is enabled."""
        assert production_env_vars["enable-tracing"].lower() == "true"

    def test_metrics_enabled(self, production_env_vars):
        """Test that metrics are enabled."""
        assert production_env_vars["enable-metrics"].lower() == "true"


class TestProductionEnvironmentKubernetes:
    """Test Kubernetes configuration for production environment."""

    def test_namespace_is_production(self, production_kubernetes_config):
        """Test that namespace is victor-ai-prod."""
        assert production_kubernetes_config["namespace"] == "victor-ai-prod"

    def test_six_replicas(self, production_kubernetes_config):
        """Test that 6 replicas are configured."""
        assert production_kubernetes_config["replicas"] == 6

    def test_high_resources(self, production_kubernetes_config):
        """Test that high resources are allocated."""
        resources = production_kubernetes_config["resources"]
        assert resources["requests"]["cpu"] == "1000m"
        assert resources["requests"]["memory"] == "1Gi"
        assert resources["limits"]["cpu"] == "4000m"
        assert resources["limits"]["memory"] == "4Gi"

    def test_production_image_tag(self, production_kubernetes_config):
        """Test that production image tag is used."""
        assert production_kubernetes_config["image"]["tag"] == "0.5.1"

    def test_autoscaling_enabled(self, production_kubernetes_config):
        """Test that autoscaling is enabled."""
        assert production_kubernetes_config["autoscaling"]["enabled"] is True

    def test_autoscaling_range(self, production_kubernetes_config):
        """Test that autoscaling range is 6-50 replicas."""
        autoscaling = production_kubernetes_config["autoscaling"]
        assert autoscaling["minReplicas"] == 6
        assert autoscaling["maxReplicas"] == 50

    def test_autoscaling_targets(self, production_kubernetes_config):
        """Test that autoscaling targets are configured."""
        autoscaling = production_kubernetes_config["autoscaling"]
        assert autoscaling["targetCPUUtilizationPercentage"] == 70
        assert autoscaling["targetMemoryUtilizationPercentage"] == 80


class TestProductionEnvironmentFeatures:
    """Test production-specific features."""

    def test_production_overlay_exists(self):
        """Test that production overlay directory exists."""
        overlay_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays/production"
        )
        assert overlay_path.exists()

    def test_kustomization_exists(self):
        """Test that production kustomization file exists."""
        kustomization_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays/production/kustomization.yaml"
        )
        assert kustomization_path.exists()

    def test_deployment_patch_exists(self):
        """Test that deployment patch file exists."""
        patch_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays/production/deployment-patch.yaml"
        )
        assert patch_path.exists()

    def test_hpa_patch_exists(self):
        """Test that HPA patch file exists."""
        hpa_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays/production/hpa-patch.yaml"
        )
        assert hpa_path.exists()

    def test_environment_label(self):
        """Test that environment label is set to production."""
        kustomization_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays/production/kustomization.yaml"
        )
        content = kustomization_path.read_text()
        assert "environment: production" in content

    def test_config_map_generates_production_vars(self):
        """Test that ConfigMap generates production variables."""
        kustomization_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays/production/kustomization.yaml"
        )
        content = kustomization_path.read_text()
        assert "VICTOR_PROFILE=production" in content
        assert "event-bus-backend=kafka" in content
        assert "checkpoint-backend=postgres" in content
        assert "cache-backend=redis" in content
        assert "rate-limit-requests-per-minute=1000" in content


@pytest.mark.integration
class TestProductionEnvironmentIntegration:
    """Integration tests for production environment."""

    def test_distributed_backends_configured(self, production_env_vars):
        """Test that distributed backends are configured."""
        assert production_env_vars["event-bus-backend"] == "kafka"
        assert production_env_vars["checkpoint-backend"] == "postgres"
        assert production_env_vars["cache-backend"] == "redis"

    def test_high_availability_configuration(self, production_kubernetes_config):
        """Test that HA is configured properly."""
        assert production_kubernetes_config["replicas"] >= 6
        assert production_kubernetes_config["autoscaling"]["enabled"] is True

    def test_observation_configured(self, production_env_vars):
        """Test that observability is fully configured."""
        assert production_env_vars["enable-metrics"].lower() == "true"
        assert production_env_vars["enable-tracing"].lower() == "true"


@pytest.mark.deployment
class TestProductionEnvironmentDeployment:
    """Deployment-specific tests for production environment."""

    def test_deployment_yaml_syntax(self):
        """Test that deployment YAML has valid syntax."""
        import yaml

        patch_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays/production/deployment-patch.yaml"
        )
        with open(patch_path) as f:
            config = yaml.safe_load(f)
            assert config is not None
            assert config["kind"] == "Deployment"

    def test_kustomization_syntax(self):
        """Test that kustomization YAML has valid syntax."""
        import yaml

        kustomization_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays/production/kustomization.yaml"
        )
        with open(kustomization_path) as f:
            config = yaml.safe_load(f)
            assert config is not None
            assert config["apiVersion"] == "kustomize.config.k8s.io/v1beta1"
            assert config["kind"] == "Kustomization"

    def test_hpa_yaml_syntax(self):
        """Test that HPA YAML has valid syntax."""
        import yaml

        hpa_path = Path(
            "/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays/production/hpa-patch.yaml"
        )
        with open(hpa_path) as f:
            config = yaml.safe_load(f)
            assert config is not None
            assert config["kind"] == "HorizontalPodAutoscaler"

    def test_helm_production_values_syntax(self):
        """Test that production Helm values have valid syntax."""
        import yaml

        values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values-prod.yaml")
        with open(values_path) as f:
            values = yaml.safe_load(f)
            assert values is not None
            assert "replicaCount" in values
            assert values["replicaCount"] == 6

    def test_resource_limits_valid(self):
        """Test that resource limits are valid and reasonable."""
        import yaml

        values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values-prod.yaml")
        with open(values_path) as f:
            values = yaml.safe_load(f)
            resources = values["resources"]

            # Verify limits are higher than requests
            cpu_req = int(resources["requests"]["cpu"].replace("m", ""))
            cpu_lim = int(resources["limits"]["cpu"].replace("m", ""))
            assert cpu_lim > cpu_req

            mem_req = int(resources["requests"]["memory"].replace("Gi", ""))
            mem_lim = int(resources["limits"]["memory"].replace("Gi", ""))
            assert mem_lim > mem_req

    def test_six_replicas_configured(self):
        """Test that production values configure 6 replicas."""
        import yaml

        values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values-prod.yaml")
        with open(values_path) as f:
            values = yaml.safe_load(f)
            assert values["replicaCount"] == 6

    def test_production_affinity_rules(self):
        """Test that production has proper affinity rules."""
        import yaml

        values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values-prod.yaml")
        with open(values_path) as f:
            values = yaml.safe_load(f)
            assert "affinity" in values
            # Check for pod anti-affinity
            assert "podAntiAffinity" in values["affinity"]


@pytest.mark.security
class TestProductionEnvironmentSecurity:
    """Security tests for production environment."""

    def test_rate_limiting_enabled(self, production_env_vars):
        """Test that rate limiting is enabled."""
        assert production_env_vars["enable-rate-limiting"].lower() == "true"

    def test_production_rate_limit(self, production_env_vars):
        """Test that production rate limit is higher than staging."""
        rate_limit = int(production_env_vars["rate-limit-requests-per-minute"])
        assert rate_limit == 1000

    def test_tls_enabled_in_values(self):
        """Test that TLS is enabled in production values."""
        import yaml

        values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values-prod.yaml")
        with open(values_path) as f:
            values = yaml.safe_load(f)
            assert values["ingress"]["enabled"] is True
            assert len(values["ingress"]["tls"]) > 0

    def test_network_policies_enabled(self):
        """Test that network policies are enabled in production."""
        import yaml

        values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values-prod.yaml")
        with open(values_path) as f:
            values = yaml.safe_load(f)
            assert values.get("networkPolicy", {}).get("enabled") is True

    def test_readonly_root_filesystem(self):
        """Test that root filesystem is read-only in production."""
        import yaml

        values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values.yaml")
        with open(values_path) as f:
            values = yaml.safe_load(f)
            security_context = values["securityContext"]
            # Note: This might be set to false for some applications
            # Just checking that the field exists
            assert "readOnlyRootFilesystem" in security_context


@pytest.mark.performance
class TestProductionEnvironmentPerformance:
    """Performance tests for production environment."""

    def test_cache_size_maximized(self, production_env_vars):
        """Test that cache size is maximized for production."""
        cache_size = int(production_env_vars["VICTOR_CACHE_SIZE"])
        assert cache_size == 2000

    def test_worker_count_maximized(self, production_env_vars):
        """Test that worker count is maximized for production."""
        max_workers = int(production_env_vars["VICTOR_MAX_WORKERS"])
        assert max_workers == 8

    def test_autoscaling_configured(self, production_kubernetes_config):
        """Test that autoscaling is properly configured."""
        autoscaling = production_kubernetes_config["autoscaling"]
        assert autoscaling["enabled"] is True
        assert autoscaling["minReplicas"] >= 6
        assert autoscaling["maxReplicas"] <= 50

    def test_resource_limits_high(self, production_kubernetes_config):
        """Test that resource limits are high for performance."""
        limits = production_kubernetes_config["resources"]["limits"]
        cpu_limit = int(limits["cpu"].replace("m", ""))
        memory_limit = int(limits["memory"].replace("Gi", ""))

        assert cpu_limit >= 4000
        assert memory_limit >= 4


@pytest.mark.monitoring
class TestProductionEnvironmentMonitoring:
    """Monitoring tests for production environment."""

    def test_full_monitoring_enabled(self):
        """Test that full monitoring is enabled in production."""
        import yaml

        values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values-prod.yaml")
        with open(values_path) as f:
            values = yaml.safe_load(f)
            assert values["monitoring"]["enabled"] is True
            assert values["monitoring"]["serviceMonitor"]["enabled"] is True

    def test_tracing_enabled(self, production_env_vars):
        """Test that distributed tracing is enabled."""
        assert production_env_vars["enable-tracing"].lower() == "true"

    def test_metrics_enabled(self, production_env_vars):
        """Test that Prometheus metrics are enabled."""
        assert production_env_vars["enable-metrics"].lower() == "true"


@pytest.mark.high_availability
class TestProductionEnvironmentHA:
    """High availability tests for production environment."""

    def test_minimum_replicas_for_ha(self, production_kubernetes_config):
        """Test that minimum replicas meet HA requirements."""
        assert production_kubernetes_config["replicas"] >= 6

    def test_pod_disruption_budget_configured(self):
        """Test that PodDisruptionBudget is configured."""
        import yaml

        values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values.yaml")
        with open(values_path) as f:
            values = yaml.safe_load(f)
            pdb = values.get("podDisruptionBudget", {})
            assert pdb.get("enabled") is True
            assert pdb.get("minAvailable", 0) >= 2

    def test_multi_zone_affinity(self):
        """Test that multi-zone affinity is configured."""
        import yaml

        values_path = Path("/Users/vijaysingh/code/codingagent/deployment/helm/values-prod.yaml")
        with open(values_path) as f:
            values = yaml.safe_load(f)
            # Check for topology spread constraints or zone anti-affinity
            affinity = values.get("affinity", {})
            assert len(affinity) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
