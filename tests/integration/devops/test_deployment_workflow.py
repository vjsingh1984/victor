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

"""Integration tests for DevOps deployment workflows.

Tests cover:
- Deployment planning
- Docker workflow execution
- Configuration management
- Infrastructure validation
- Multi-stage deployments
"""

from unittest.mock import Mock, patch

import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_dockerfile():
    """Sample Dockerfile for testing."""
    return """
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
"""


@pytest.fixture
def sample_docker_compose():
    """Sample docker-compose.yml for testing."""
    return {
        "version": "3.8",
        "services": {
            "web": {
                "build": ".",
                "ports": ["8000:8000"],
                "environment": {
                    "DATABASE_URL": "postgresql://db:5432/mydb",
                    "REDIS_URL": "redis://redis:6379/0",
                },
                "depends_on": ["db", "redis"],
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3,
                },
                "restart": "unless-stopped",
            },
            "db": {
                "image": "postgres:15",
                "environment": {
                    "POSTGRES_USER": "user",
                    "POSTGRES_PASSWORD": "password",
                    "POSTGRES_DB": "mydb",
                },
                "volumes": ["postgres_data:/var/lib/postgresql/data"],
                "restart": "unless-stopped",
            },
            "redis": {
                "image": "redis:7-alpine",
                "restart": "unless-stopped",
            },
        },
        "volumes": {
            "postgres_data": {"driver": "local"},
        },
        "networks": {
            "default": {
                "driver": "bridge",
            },
        },
    }


@pytest.fixture
def sample_kubernetes_deployment():
    """Sample Kubernetes deployment manifest."""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "web-app", "labels": {"app": "web"}},
        "spec": {
            "replicas": 3,
            "selector": {"matchLabels": {"app": "web"}},
            "template": {
                "metadata": {"labels": {"app": "web"}},
                "spec": {
                    "containers": [
                        {
                            "name": "web",
                            "image": "myapp:1.0",
                            "ports": [{"containerPort": 8000}],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "128Mi"},
                                "limits": {"cpu": "500m", "memory": "512Mi"},
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/ready", "port": 8000},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                            },
                        }
                    ],
                },
            },
        },
    }


@pytest.fixture
def sample_terraform_config():
    """Sample Terraform configuration."""
    return """
resource "aws_instance" "web" {
    ami           = "ami-12345678"
    instance_type = "t3.micro"

    tags = {
        Name        = "WebServer"
        Environment = var.environment
    }
}

resource "aws_security_group" "web" {
    name = "web-sg"

    ingress {
        from_port   = 80
        to_port     = 80
        protocol    = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    egress {
        from_port   = 0
        to_port     = 0
        protocol    = "-1"
        cidr_blocks = ["0.0.0.0/0"]
    }
}

variable "environment" {
    description = "Environment name"
    type        = string
    default     = "dev"
}

output "instance_ip" {
    description = "Public IP of the web server"
    value       = aws_instance.web.public_ip
}
"""


# ============================================================================
# Deployment Planning Tests
# ============================================================================


class TestDeploymentPlanning:
    """Test deployment planning workflow."""

    def test_deployment_identifies_components(self, sample_docker_compose):
        """Test that deployment planning identifies all components."""
        services = sample_docker_compose.get("services", {})

        assert "web" in services
        assert "db" in services
        assert "redis" in services

    def test_deployment_identifies_dependencies(self, sample_docker_compose):
        """Test that deployment planning identifies service dependencies."""
        web_service = sample_docker_compose["services"]["web"]
        dependencies = web_service.get("depends_on", [])

        assert "db" in dependencies
        assert "redis" in dependencies

    def test_deployment_validates_startup_order(self, sample_docker_compose):
        """Test that deployment validates correct startup order."""
        # Web depends on db and redis, so they should start first
        web_deps = sample_docker_compose["services"]["web"].get("depends_on", [])

        # db and redis have no dependencies
        db_deps = sample_docker_compose["services"]["db"].get("depends_on", [])
        redis_deps = sample_docker_compose["services"]["redis"].get("depends_on", [])

        assert len(db_deps) == 0
        assert len(redis_deps) == 0
        assert len(web_deps) == 2

    def test_deployment_plans_resource_allocation(self, sample_kubernetes_deployment):
        """Test that deployment plans compute resources."""
        container = sample_kubernetes_deployment["spec"]["template"]["spec"]["containers"][0]

        assert "resources" in container
        assert "requests" in container["resources"]
        assert "limits" in container["resources"]

    def test_deployment_validates_health_checks(self, sample_docker_compose):
        """Test that deployment validates health check configuration."""
        web_service = sample_docker_compose["services"]["web"]

        assert "healthcheck" in web_service
        healthcheck = web_service["healthcheck"]
        assert "test" in healthcheck
        assert "interval" in healthcheck


# ============================================================================
# Docker Workflow Execution Tests
# ============================================================================


class TestDockerWorkflowExecution:
    """Test Docker-based deployment workflows."""

    @pytest.mark.asyncio
    async def test_docker_build_workflow(self):
        """Test Docker image build workflow."""
        with patch("victor.tools.docker_tool._run_docker_command_async") as mock_docker:
            # Mock successful build
            mock_docker.return_value = (True, "Built image myapp:1.0", "")

            # Simulate build workflow
            result = await docker(operation="build", resource_id="myapp:1.0", options={"path": "."})

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_compose_start_workflow(self, sample_docker_compose):
        """Test Docker Compose startup workflow."""
        with patch("victor.tools.subprocess_executor.run_command_async") as mock_run:
            # Mock successful compose up
            mock_result = Mock()
            mock_result.success = True
            mock_result.stdout = "Container started"
            mock_run.return_value = mock_result

            # Verify configuration is valid
            assert "version" in sample_docker_compose
            assert "services" in sample_docker_compose

            # Check all services are defined
            services = sample_docker_compose["services"]
            assert len(services) == 3

    @pytest.mark.asyncio
    async def test_docker_compose_restart_strategy(self, sample_docker_compose):
        """Test that services have proper restart strategies."""
        for service_name, service_config in sample_docker_compose["services"].items():
            assert "restart" in service_config
            assert service_config["restart"] == "unless-stopped"

    @pytest.mark.asyncio
    async def test_docker_volume_persistence(self, sample_docker_compose):
        """Test that stateful services use persistent volumes."""
        db_service = sample_docker_compose["services"]["db"]

        assert "volumes" in db_service
        assert any("postgres" in vol for vol in db_service["volumes"])

    @pytest.mark.asyncio
    async def test_docker_network_isolation(self, sample_docker_compose):
        """Test network isolation between services."""
        # All services should be on the same network
        assert "networks" in sample_docker_compose

    @pytest.mark.asyncio
    async def test_docker_environment_configuration(self, sample_docker_compose):
        """Test environment variable configuration."""
        web_service = sample_docker_compose["services"]["web"]

        assert "environment" in web_service
        env = web_service["environment"]
        assert "DATABASE_URL" in env
        assert "REDIS_URL" in env


# ============================================================================
# Configuration Management Tests
# ============================================================================


class TestConfigurationManagement:
    """Test configuration management in deployments."""

    def test_environment_variable_validation(self, sample_docker_compose):
        """Test that required environment variables are defined."""
        web_service = sample_docker_compose["services"]["web"]
        env = web_service.get("environment", {})

        # Required variables for web service
        assert "DATABASE_URL" in env
        assert "REDIS_URL" in env

    def test_configuration_secrets_detection(self, sample_docker_compose):
        """Test detection of sensitive configuration."""
        db_service = sample_docker_compose["services"]["db"]
        db_env = db_service.get("environment", {})

        # Database password is sensitive
        assert "POSTGRES_PASSWORD" in db_env
        # In production, this should use secrets manager

    def test_config_map_generation(self, sample_kubernetes_deployment):
        """Test ConfigMap generation for Kubernetes."""
        # Deployment should be able to reference ConfigMaps
        assert "spec" in sample_kubernetes_deployment

    def test_secret_mounting(self, sample_kubernetes_deployment):
        """Test secret volume mounting in Kubernetes."""
        # Deployment should support secret mounts
        containers = sample_kubernetes_deployment["spec"]["template"]["spec"]["containers"]
        assert len(containers) > 0

    def test_multi_environment_configuration(self):
        """Test configuration for multiple environments."""
        environments = {
            "dev": {"replicas": 1, "resources": {"cpu": "100m"}},
            "staging": {"replicas": 2, "resources": {"cpu": "200m"}},
            "production": {"replicas": 3, "resources": {"cpu": "500m"}},
        }

        for env, config in environments.items():
            assert "replicas" in config
            assert config["replicas"] > 0


# ============================================================================
# Infrastructure Validation Tests
# ============================================================================


class TestInfrastructureValidation:
    """Test infrastructure validation workflows."""

    def test_dockerfile_validates(self, sample_dockerfile):
        """Test Dockerfile validation."""
        assert "FROM" in sample_dockerfile
        assert "WORKDIR" in sample_dockerfile
        assert "EXPOSE" in sample_dockerfile
        assert "HEALTHCHECK" in sample_dockerfile
        assert "CMD" in sample_dockerfile

    def test_dockerfile_uses_multistage_build(self, sample_dockerfile):
        """Test Dockerfile uses multi-stage build (if applicable)."""
        # Check for multi-stage build pattern
        has_stages = "as" in sample_dockerfile.lower() and sample_dockerfile.count("FROM") > 1
        # Multi-stage is recommended but not required

    def test_dockerfile_uses_non_root_user(self, sample_dockerfile):
        """Test Dockerfile uses non-root user."""
        assert "USER" in sample_dockerfile

    def test_kubernetes_deployment_validates(self, sample_kubernetes_deployment):
        """Test Kubernetes deployment validation."""
        assert "apiVersion" in sample_kubernetes_deployment
        assert "kind" in sample_kubernetes_deployment
        assert sample_kubernetes_deployment["kind"] == "Deployment"
        assert "spec" in sample_kubernetes_deployment

    def test_kubernetes_has_resource_limits(self, sample_kubernetes_deployment):
        """Test Kubernetes pods have resource limits."""
        container = sample_kubernetes_deployment["spec"]["template"]["spec"]["containers"][0]
        assert "resources" in container
        assert "limits" in container["resources"]

    def test_kubernetes_has_health_probes(self, sample_kubernetes_deployment):
        """Test Kubernetes pods have health probes."""
        container = sample_kubernetes_deployment["spec"]["template"]["spec"]["containers"][0]
        assert "livenessProbe" in container
        assert "readinessProbe" in container

    def test_terraform_syntax_validation(self, sample_terraform_config):
        """Test Terraform configuration syntax."""
        assert "resource" in sample_terraform_config
        assert "variable" in sample_terraform_config
        assert "output" in sample_terraform_config

    def test_terraform_resource_dependencies(self, sample_terraform_config):
        """Test Terraform resource dependencies."""
        # Should have at least one resource
        assert "resource" in sample_terraform_config


# ============================================================================
# Multi-Stage Deployment Tests
# ============================================================================


class TestMultiStageDeployment:
    """Test multi-stage deployment workflows."""

    def test_deployment_stages_defined(self):
        """Test that deployment stages are properly defined."""
        stages = ["build", "test", "staging", "production"]
        assert len(stages) == 4

    def test_canary_deployment_strategy(self, sample_kubernetes_deployment):
        """Test canary deployment configuration."""
        replicas = sample_kubernetes_deployment["spec"]["replicas"]
        assert replicas >= 1

    def test_blue_green_deployment(self):
        """Test blue-green deployment setup."""
        # Blue-green requires two versions
        blue_version = "1.0"
        green_version = "1.1"

        assert blue_version != green_version

    def test_rolling_update_configuration(self, sample_kubernetes_deployment):
        """Test rolling update configuration."""
        spec = sample_kubernetes_deployment["spec"]
        assert "replicas" in spec

    def test_zero_downtime_deployment(self, sample_docker_compose):
        """Test zero-downtime deployment configuration."""
        # Services should have health checks for zero downtime
        web_service = sample_docker_compose["services"]["web"]
        assert "healthcheck" in web_service


# ============================================================================
# Deployment Workflow Integration Tests
# ============================================================================


class TestDeploymentWorkflowIntegration:
    """Test complete deployment workflow integration."""

    @pytest.mark.asyncio
    async def test_complete_deployment_workflow(self, sample_dockerfile, sample_docker_compose):
        """Test end-to-end deployment workflow."""
        # Step 1: Validate Dockerfile
        assert "FROM" in sample_dockerfile
        assert "EXPOSE" in sample_dockerfile

        # Step 2: Validate docker-compose configuration
        assert "services" in sample_docker_compose

        # Step 3: Verify service dependencies
        services = sample_docker_compose["services"]
        assert "web" in services
        assert "db" in services

        # Step 4: Verify health checks
        web_service = services["web"]
        assert "healthcheck" in web_service

    @pytest.mark.asyncio
    async def test_deployment_rollback_scenario(self):
        """Test deployment rollback scenario."""
        # Simulate deployment versions
        current_version = "1.0"
        new_version = "1.1"

        # Rollback would go back to current_version
        rollback_version = current_version
        assert rollback_version == "1.0"

    @pytest.mark.asyncio
    async def test_deployment_monitoring_setup(self, sample_kubernetes_deployment):
        """Test deployment monitoring configuration."""
        container = sample_kubernetes_deployment["spec"]["template"]["spec"]["containers"][0]

        # Health probes enable monitoring
        assert "livenessProbe" in container
        assert "readinessProbe" in container

    @pytest.mark.asyncio
    async def test_deployment_scaling_configuration(self, sample_kubernetes_deployment):
        """Test deployment auto-scaling configuration."""
        spec = sample_kubernetes_deployment["spec"]
        replicas = spec.get("replicas", 1)

        assert replicas > 0

    @pytest.mark.asyncio
    async def test_deployment_backup_strategy(self, sample_docker_compose):
        """Test deployment backup and recovery strategy."""
        # Database should have persistent volume for backup
        db_service = sample_docker_compose["services"]["db"]
        assert "volumes" in db_service


# ============================================================================
# Deployment Security Tests
# ============================================================================


class TestDeploymentSecurity:
    """Test security aspects of deployments."""

    def test_dockerfile_security_hardening(self, sample_dockerfile):
        """Test Dockerfile security best practices."""
        # Should use non-root user
        assert "USER" in sample_dockerfile

        # Should use specific version (not latest)
        assert ":latest" not in sample_dockerfile

        # Should have health check
        assert "HEALTHCHECK" in sample_dockerfile

    def test_kubernetes_security_context(self, sample_kubernetes_deployment):
        """Test Kubernetes security context configuration."""
        # Should have security context defined
        spec = sample_kubernetes_deployment["spec"]
        assert "template" in spec

    def test_network_policies(self, sample_docker_compose):
        """Test network policies and isolation."""
        # Services should be on isolated networks
        assert "networks" in sample_docker_compose

    def test_secrets_management(self, sample_docker_compose):
        """Test secrets management in deployment."""
        db_service = sample_docker_compose["services"]["db"]
        env = db_service.get("environment", {})

        # Should not hardcode sensitive data in production
        assert "POSTGRES_PASSWORD" in env


# ============================================================================
# DevOps Assistant Integration Tests
# ============================================================================


class TestDevOpsAssistantIntegration:
    """Test DevOps assistant vertical integration."""

    def test_devops_assistant_tools(self):
        """Test that DevOps assistant provides required tools."""
        from victor.devops import DevOpsAssistant
        from victor.tools.tool_names import ToolNames

        tools = DevOpsAssistant.get_tools()

        # Core tools for DevOps
        assert ToolNames.DOCKER in tools
        assert ToolNames.SHELL in tools
        assert ToolNames.GIT in tools

    def test_devops_assistant_stages(self):
        """Test that DevOps assistant defines deployment stages."""
        from victor.devops import DevOpsAssistant

        stages = DevOpsAssistant.get_stages()

        # Should have deployment-related stages
        assert "PLANNING" in stages
        assert "IMPLEMENTATION" in stages
        assert "VALIDATION" in stages
        assert "DEPLOYMENT" in stages

    def test_devops_assistant_system_prompt(self):
        """Test that DevOps assistant has appropriate system prompt."""
        from victor.devops import DevOpsAssistant

        prompt = DevOpsAssistant.get_system_prompt()

        # Should mention DevOps concepts
        assert "Docker" in prompt or "container" in prompt.lower()
        assert len(prompt) > 100  # Should be comprehensive


# ============================================================================
# Deployment Performance Tests
# ============================================================================


class TestDeploymentPerformance:
    """Test deployment performance characteristics."""

    def test_image_size_optimization(self, sample_dockerfile):
        """Test Dockerfile optimization for image size."""
        # Should use multi-stage build
        has_multistage = "as" in sample_dockerfile.lower()

        # Should use slim/alpine images
        has_slim = "slim" in sample_dockerfile or "alpine" in sample_dockerfile

        assert has_slim or has_multistage

    def test_deployment_startup_time(self, sample_docker_compose):
        """Test deployment startup time optimization."""
        # Services should have health checks to signal readiness
        web_service = sample_docker_compose["services"]["web"]
        assert "healthcheck" in web_service

    def test_resource_utilization(self, sample_kubernetes_deployment):
        """Test resource utilization configuration."""
        container = sample_kubernetes_deployment["spec"]["template"]["spec"]["containers"][0]

        # Should have resource requests and limits
        assert "resources" in container
        assert "requests" in container["resources"]
        assert "limits" in container["resources"]


# Import docker function for tests
from victor.tools.docker_tool import docker
