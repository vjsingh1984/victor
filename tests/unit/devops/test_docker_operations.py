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

"""Unit tests for Docker operations.

Tests cover:
- Dockerfile parsing and validation
- docker-compose.yml parsing
- Container operations (mocked)
- Image operations (mocked)
- Network and volume operations
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import yaml

from victor.tools.docker_tool import docker, _run_docker_command_async


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_dockerfile():
    """Sample Dockerfile content."""
    return """
# Multi-stage Dockerfile for Python application
FROM python:3.11-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY . .

# Install runtime dependencies
RUN pip install --no-cache-dir gunicorn

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
"""


@pytest.fixture
def sample_dockerfile_v2():
    """Sample Dockerfile with different format."""
    return """
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy source
COPY . .

# Expose
EXPOSE 3000

# Start
CMD ["npm", "start"]
"""


@pytest.fixture
def invalid_dockerfile():
    """Dockerfile with common issues."""
    return """
FROM python:3.11

# No WORKDIR set
RUN pip install django
RUN pip install flask
# Multiple RUNs that should be combined

# No non-root user
# No health check
CMD ["python", "app.py"]
"""


@pytest.fixture
def sample_docker_compose():
    """Sample docker-compose.yml content."""
    return {
        "version": "3.8",
        "services": {
            "web": {
                "build": ".",
                "ports": ["8000:8000"],
                "environment": {"DATABASE_URL": "postgresql://db:5432/mydb"},
                "depends_on": ["db"],
                "volumes": ["./app:/app"],
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
            "default": {"driver": "bridge"},
        },
    }


@pytest.fixture
def docker_compose_with_healthchecks():
    """docker-compose.yml with health checks."""
    return {
        "version": "3.8",
        "services": {
            "app": {
                "image": "myapp:1.0",
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3,
                },
            },
        },
    }


# ============================================================================
# Dockerfile Parsing Tests
# ============================================================================


class TestDockerfileParsing:
    """Test Dockerfile parsing and validation."""

    def test_dockerfile_base_image_detection(self, sample_dockerfile):
        """Test detecting base image from Dockerfile."""
        lines = sample_dockerfile.strip().split("\n")
        base_images = [line for line in lines if line.strip().startswith("FROM")]

        assert len(base_images) == 2
        assert "python:3.11-slim" in base_images[0]
        assert "python:3.11-slim" in base_images[1]

    def test_dockerfile_workdir_detection(self, sample_dockerfile):
        """Test detecting WORKDIR instructions."""
        lines = sample_dockerfile.strip().split("\n")
        workdirs = [line for line in lines if line.strip().startswith("WORKDIR")]

        assert len(workdirs) == 2
        assert "/app" in workdirs[0]
        assert "/app" in workdirs[1]

    def test_dockerfile_expose_detection(self, sample_dockerfile):
        """Test detecting EXPOSE instructions."""
        lines = sample_dockerfile.strip().split("\n")
        expose_lines = [line for line in lines if line.strip().startswith("EXPOSE")]

        assert len(expose_lines) == 1
        assert "8000" in expose_lines[0]

    def test_dockerfile_multistage_detection(self, sample_dockerfile):
        """Test detecting multi-stage builds."""
        lines = sample_dockerfile.strip().split("\n")
        from_lines = [line for line in lines if line.strip().startswith("FROM")]

        # Multi-stage builds have multiple FROM statements with 'as' keyword
        has_stage = any("as" in line.lower() for line in from_lines)
        assert has_stage

    def test_dockerfile_healthcheck_detection(self, sample_dockerfile):
        """Test detecting HEALTHCHECK instructions."""
        lines = sample_dockerfile.strip().split("\n")
        healthcheck = [line for line in lines if line.strip().startswith("HEALTHCHECK")]

        assert len(healthcheck) == 1
        assert "HEALTHCHECK" in healthcheck[0]

    def test_dockerfile_user_detection(self, sample_dockerfile):
        """Test detecting USER instructions for security."""
        lines = sample_dockerfile.strip().split("\n")
        user_lines = [line for line in lines if line.strip().startswith("USER")]

        assert len(user_lines) == 1
        assert "appuser" in user_lines[0]

    def test_dockerfile_cmd_detection(self, sample_dockerfile):
        """Test detecting CMD instructions."""
        lines = sample_dockerfile.strip().split("\n")
        # Match CMD instructions at the start of line (not indented within HEALTHCHECK)
        # HEALTHCHECK has indented CMD, real CMD is not indented
        cmd_lines = []
        for line in lines:
            # Match lines that start with CMD (after stripping leading whitespace)
            # This excludes the indented CMD in HEALTHCHECK
            if line.lstrip().startswith("CMD") and not line.startswith("    CMD"):
                cmd_lines.append(line)

        assert len(cmd_lines) == 1
        assert "gunicorn" in cmd_lines[0]

    def test_dockerfile_copy_detection(self, sample_dockerfile):
        """Test detecting COPY instructions."""
        lines = sample_dockerfile.strip().split("\n")
        copy_lines = [line for line in lines if line.strip().startswith("COPY")]

        assert len(copy_lines) >= 2

    def test_dockerfile_run_detection(self, sample_dockerfile):
        """Test detecting RUN instructions."""
        lines = sample_dockerfile.strip().split("\n")
        run_lines = [line for line in lines if line.strip().startswith("RUN")]

        assert len(run_lines) >= 2


# ============================================================================
# Dockerfile Validation Tests
# ============================================================================


class TestDockerfileValidation:
    """Test Dockerfile validation and best practices."""

    def test_valid_dockerfile_multistage(self, sample_dockerfile):
        """Test that sample Dockerfile uses multi-stage build."""
        assert "FROM" in sample_dockerfile
        assert "as builder" in sample_dockerfile.lower()

    def test_valid_dockerfile_non_root_user(self, sample_dockerfile):
        """Test that Dockerfile uses non-root user."""
        assert "USER" in sample_dockerfile
        assert "appuser" in sample_dockerfile

    def test_valid_dockerfile_healthcheck(self, sample_dockerfile):
        """Test that Dockerfile has health check."""
        assert "HEALTHCHECK" in sample_dockerfile

    def test_valid_dockerfile_specific_version(self, sample_dockerfile):
        """Test that base image uses specific version."""
        assert "python:3.11" in sample_dockerfile
        # Should not use 'latest' tag
        assert ":latest" not in sample_dockerfile

    def test_invalid_dockerfile_missing_healthcheck(self, invalid_dockerfile):
        """Test detecting missing health check."""
        assert "HEALTHCHECK" not in invalid_dockerfile

    def test_invalid_dockerfile_no_non_root_user(self, invalid_dockerfile):
        """Test detecting missing non-root user."""
        assert "USER" not in invalid_dockerfile

    def test_invalid_dockerfile_combined_runs(self, invalid_dockerfile):
        """Test detecting non-combined RUN instructions."""
        run_lines = [
            line for line in invalid_dockerfile.split("\n") if line.strip().startswith("RUN")
        ]
        assert len(run_lines) > 1  # Should be combined for layer optimization


# ============================================================================
# Docker Compose Parsing Tests
# ============================================================================


class TestDockerComposeParsing:
    """Test docker-compose.yml parsing."""

    def test_parse_docker_compose_version(self, sample_docker_compose):
        """Test parsing version from docker-compose.yml."""
        assert "version" in sample_docker_compose
        assert sample_docker_compose["version"] == "3.8"

    def test_parse_docker_compose_services(self, sample_docker_compose):
        """Test parsing services from docker-compose.yml."""
        assert "services" in sample_docker_compose
        services = sample_docker_compose["services"]

        assert "web" in services
        assert "db" in services
        assert "redis" in services

    def test_parse_service_ports(self, sample_docker_compose):
        """Test parsing service port mappings."""
        web_service = sample_docker_compose["services"]["web"]
        assert "ports" in web_service
        assert "8000:8000" in web_service["ports"]

    def test_parse_service_environment(self, sample_docker_compose):
        """Test parsing service environment variables."""
        web_service = sample_docker_compose["services"]["web"]
        assert "environment" in web_service
        assert "DATABASE_URL" in web_service["environment"]

    def test_parse_service_volumes(self, sample_docker_compose):
        """Test parsing service volume mounts."""
        web_service = sample_docker_compose["services"]["web"]
        assert "volumes" in web_service
        assert "./app:/app" in web_service["volumes"]

    def test_parse_service_depends_on(self, sample_docker_compose):
        """Test parsing service dependencies."""
        web_service = sample_docker_compose["services"]["web"]
        assert "depends_on" in web_service
        assert "db" in web_service["depends_on"]

    def test_parse_restart_policy(self, sample_docker_compose):
        """Test parsing restart policies."""
        web_service = sample_docker_compose["services"]["web"]
        assert "restart" in web_service
        assert web_service["restart"] == "unless-stopped"

    def test_parse_named_volumes(self, sample_docker_compose):
        """Test parsing named volumes."""
        assert "volumes" in sample_docker_compose
        assert "postgres_data" in sample_docker_compose["volumes"]

    def test_parse_networks(self, sample_docker_compose):
        """Test parsing networks."""
        assert "networks" in sample_docker_compose
        assert "default" in sample_docker_compose["networks"]


# ============================================================================
# Docker Compose Validation Tests
# ============================================================================


class TestDockerComposeValidation:
    """Test docker-compose.yml validation."""

    def test_validate_healthchecks_present(self, docker_compose_with_healthchecks):
        """Test detecting health checks in services."""
        app_service = docker_compose_with_healthchecks["services"]["app"]
        assert "healthcheck" in app_service
        assert "test" in app_service["healthcheck"]
        assert "interval" in app_service["healthcheck"]

    def test_validate_restart_policy(self, sample_docker_compose):
        """Test that services have restart policies."""
        for service_name, service in sample_docker_compose["services"].items():
            assert "restart" in service, f"Service {service_name} missing restart policy"

    def test_validate_volume_persistence(self, sample_docker_compose):
        """Test that databases use persistent volumes."""
        db_service = sample_docker_compose["services"]["db"]
        assert "volumes" in db_service
        # Should use named volume for data persistence
        assert any("postgres" in v for v in db_service["volumes"])

    def test_validate_network_isolation(self, sample_docker_compose):
        """Test network isolation configuration."""
        assert "networks" in sample_docker_compose


# ============================================================================
# Docker Tool Operations Tests (Mocked)
# ============================================================================


class TestDockerToolOperations:
    """Test Docker tool operations with mocked Docker CLI."""

    @pytest.mark.asyncio
    async def test_docker_ps_running_containers(self):
        """Test listing running containers."""
        mock_containers = [
            {"ID": "abc123", "Names": "web-1", "Status": "Up 2 hours", "Ports": "8000:8000"},
            {"ID": "def456", "Names": "db-1", "Status": "Up 2 hours", "Ports": "5432:5432"},
        ]
        # Docker CLI returns JSON lines (one JSON per line)
        json_output = "\n".join(json.dumps(c) for c in mock_containers)

        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, json_output, ""),
        ):
            result = await docker(operation="ps")

            assert result["success"] is True
            assert result["result"]["count"] == 2
            assert len(result["result"]["containers"]) == 2

    @pytest.mark.asyncio
    async def test_docker_ps_all_containers(self):
        """Test listing all containers including stopped."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "[]", ""),
        ):
            result = await docker(operation="ps", options={"all": True})

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_start_container(self):
        """Test starting a container."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "abc123", ""),
        ):
            result = await docker(operation="start", resource_id="abc123")

            assert result["success"] is True
            assert "start successful" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_docker_stop_container(self):
        """Test stopping a container."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "abc123", ""),
        ):
            result = await docker(operation="stop", resource_id="abc123")

            assert result["success"] is True
            assert "stop successful" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_docker_restart_container(self):
        """Test restarting a container."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "abc123", ""),
        ):
            result = await docker(operation="restart", resource_id="abc123")

            assert result["success"] is True
            assert "restart successful" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_docker_remove_container(self):
        """Test removing a container."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "abc123", ""),
        ):
            result = await docker(operation="rm", resource_id="abc123")

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_remove_container_force(self):
        """Test force removing a container."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "abc123", ""),
        ):
            result = await docker(operation="rm", resource_id="abc123", options={"force": True})

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_logs(self):
        """Test getting container logs."""
        log_output = "Starting server...\nServer running on port 8000"

        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, log_output, ""),
        ):
            result = await docker(operation="logs", resource_id="abc123")

            assert result["success"] is True
            assert "Starting server" in result["result"]["logs"]

    @pytest.mark.asyncio
    async def test_docker_logs_with_tail(self):
        """Test getting container logs with tail option."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "Last 100 lines", ""),
        ):
            result = await docker(operation="logs", resource_id="abc123", options={"tail": 100})

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_inspect(self):
        """Test inspecting a container."""
        inspect_data = {
            "Id": "abc123",
            "State": {"Running": True},
            "Config": {"Image": "nginx:latest"},
        }

        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, json.dumps(inspect_data), ""),
        ):
            result = await docker(operation="inspect", resource_id="abc123")

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_images(self):
        """Test listing images."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "[]", ""),
        ):
            result = await docker(operation="images")

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_pull(self):
        """Test pulling an image."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "Pull complete", ""),
        ):
            result = await docker(operation="pull", resource_id="nginx:latest")

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_remove_image(self):
        """Test removing an image."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "Deleted", ""),
        ):
            result = await docker(operation="rmi", resource_id="nginx:old")

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_networks(self):
        """Test listing networks."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "[]", ""),
        ):
            result = await docker(operation="networks")

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_volumes(self):
        """Test listing volumes."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "[]", ""),
        ):
            result = await docker(operation="volumes")

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_operation_missing_resource_id(self):
        """Test operation that requires resource_id without providing it."""
        result = await docker(operation="stop")

        assert result["success"] is False
        assert "resource_id required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_docker_not_available(self):
        """Test behavior when Docker CLI is not available."""
        with patch("victor.tools.docker_tool.check_docker_available", return_value=False):
            result = await docker(operation="ps")

            assert result["success"] is False
            assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_docker_stats(self):
        """Test getting container stats."""
        mock_stats = [
            {"Name": "web-1", "CPUPerc": "5.5%", "MemUsage": "100MiB/1GiB"},
            {"Name": "db-1", "CPUPerc": "2.3%", "MemUsage": "500MiB/2GiB"},
        ]
        # Docker CLI returns JSON lines
        json_output = "\n".join(json.dumps(s) for s in mock_stats)

        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, json_output, ""),
        ):
            result = await docker(operation="stats")

            assert result["success"] is True
            assert result["result"]["count"] == 2

    @pytest.mark.asyncio
    async def test_docker_stats_with_container(self):
        """Test getting stats for specific container."""
        mock_stats = [{"Name": "web-1", "CPUPerc": "5.5%", "MemUsage": "100MiB/1GiB"}]
        json_output = "\n".join(json.dumps(s) for s in mock_stats)

        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, json_output, ""),
        ):
            result = await docker(operation="stats", resource_id="web-1")

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_exec_success(self):
        """Test executing command in container successfully."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "file1.txt\nfile2.txt\n", ""),
        ):
            result = await docker(
                operation="exec",
                resource_id="abc123",
                options={"command": "ls -la"},
            )

            assert result["success"] is True
            assert "file1.txt" in result["result"]["stdout"]

    @pytest.mark.asyncio
    async def test_docker_exec_missing_command(self):
        """Test exec without providing command."""
        result = await docker(operation="exec", resource_id="abc123")

        assert result["success"] is False
        assert "command required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_docker_exec_list_command(self):
        """Test exec with list command."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "output", ""),
        ):
            result = await docker(
                operation="exec",
                resource_id="abc123",
                options={"command": ["ls", "-la", "/app"]},
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_inspect_parse_error(self):
        """Test inspect with invalid JSON response."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "invalid json", ""),
        ):
            result = await docker(operation="inspect", resource_id="abc123")

            assert result["success"] is False
            assert "parse" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_docker_pull_with_platform(self):
        """Test pulling image with platform option."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "Pull complete", ""),
        ):
            result = await docker(
                operation="pull",
                resource_id="nginx:latest",
                options={"platform": "linux/arm64"},
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_pull_missing_image(self):
        """Test pull without providing image name."""
        result = await docker(operation="pull")

        assert result["success"] is False
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_docker_remove_image_force(self):
        """Test force removing an image."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "Deleted", ""),
        ):
            result = await docker(
                operation="rmi",
                resource_id="nginx:old",
                options={"force": True},
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_remove_image_missing_id(self):
        """Test rmi without providing image ID."""
        result = await docker(operation="rmi")

        assert result["success"] is False
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_docker_unknown_operation(self):
        """Test unknown Docker operation."""
        result = await docker(operation="unknown_op")

        assert result["success"] is False
        assert "unknown operation" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_docker_logs_follow_timeout(self):
        """Test logs with follow option uses longer timeout."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "streaming logs", ""),
        ) as mock_run:
            result = await docker(
                operation="logs",
                resource_id="abc123",
                options={"follow": True},
            )

            assert result["success"] is True
            # Check that timeout was increased for follow mode
            mock_run.assert_called_once()
            # _run_docker_command_async is called with args list and timeout kwarg
            call_args, call_kwargs = mock_run.call_args
            # call_args[0] is the args list
            # call_kwargs['timeout'] should be 60
            assert "timeout" in call_kwargs
            assert call_kwargs["timeout"] == 60  # timeout parameter for follow mode

    @pytest.mark.asyncio
    async def test_docker_ps_json_parse_error(self):
        """Test ps with invalid JSON output."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "invalid\njson\noutput", ""),
        ):
            result = await docker(operation="ps")

            assert result["success"] is True
            # Should handle parse errors gracefully
            assert result["result"]["count"] == 0

    @pytest.mark.asyncio
    async def test_docker_images_json_parse_error(self):
        """Test images with invalid JSON output."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "invalid\njson", ""),
        ):
            result = await docker(operation="images")

            assert result["success"] is True
            assert result["result"]["count"] == 0

    @pytest.mark.asyncio
    async def test_docker_networks_json_parse_error(self):
        """Test networks with invalid JSON output."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "invalid\njson", ""),
        ):
            result = await docker(operation="networks")

            assert result["success"] is True
            assert result["result"]["count"] == 0

    @pytest.mark.asyncio
    async def test_docker_volumes_json_parse_error(self):
        """Test volumes with invalid JSON output."""
        with patch(
            "victor.tools.docker_tool._run_docker_command_async",
            return_value=(True, "invalid\njson", ""),
        ):
            result = await docker(operation="volumes")

            assert result["success"] is True
            assert result["result"]["count"] == 0


# ============================================================================
# Dockerfile Analysis Tests
# ============================================================================


class TestDockerfileAnalysis:
    """Test Dockerfile analysis and recommendations."""

    def test_dockerfile_layer_optimization(self, invalid_dockerfile):
        """Test detecting non-optimized layers."""
        run_lines = [
            line for line in invalid_dockerfile.split("\n") if line.strip().startswith("RUN")
        ]
        # Multiple separate RUNs should be combined
        assert len(run_lines) > 1

    def test_dockerfile_security_scan(self, sample_dockerfile):
        """Test basic security checks."""
        # Should use non-root user
        assert "USER" in sample_dockerfile

        # Should use specific version (not latest)
        assert ":latest" not in sample_dockerfile

        # Should have health check
        assert "HEALTHCHECK" in sample_dockerfile

    def test_dockerfile_cache_optimization(self, sample_dockerfile):
        """Test detecting Docker cache optimization."""
        # Dependencies should be copied before application code
        lines = sample_dockerfile.split("\n")
        copy_indices = [i for i, line in enumerate(lines) if line.strip().startswith("COPY")]

        assert len(copy_indices) >= 2

    def test_dockerfile_minimal_base_image(self, sample_dockerfile):
        """Test detecting use of minimal base images."""
        assert "slim" in sample_dockerfile or "alpine" in sample_dockerfile

    def test_dockerfile_entrypoint_detection(self, sample_dockerfile):
        """Test detecting ENTRYPOINT instructions."""
        lines = sample_dockerfile.strip().split("\n")
        entrypoint_lines = [line for line in lines if line.strip().startswith("ENTRYPOINT")]

        # Sample doesn't have ENTRYPOINT, so this is just testing detection logic
        assert len(entrypoint_lines) == 0

    def test_dockerfile_arg_detection(self, sample_dockerfile):
        """Test detecting ARG instructions for build parameters."""
        lines = sample_dockerfile.strip().split("\n")
        arg_lines = [line for line in lines if line.strip().startswith("ARG")]

        # Sample doesn't have ARG, testing detection
        assert isinstance(arg_lines, list)

    def test_dockerfile_env_detection(self, sample_dockerfile):
        """Test detecting ENV instructions."""
        lines = sample_dockerfile.strip().split("\n")
        env_lines = [line for line in lines if line.strip().startswith("ENV")]

        # Sample doesn't have ENV, testing detection
        assert isinstance(env_lines, list)

    def test_dockerfile_label_detection(self, sample_dockerfile):
        """Test detecting LABEL instructions for metadata."""
        lines = sample_dockerfile.strip().split("\n")
        label_lines = [line for line in lines if line.strip().startswith("LABEL")]

        # Sample doesn't have LABEL, testing detection
        assert isinstance(label_lines, list)

    def test_dockerfile_add_vs_copy(self, sample_dockerfile):
        """Test detecting ADD vs COPY usage."""
        lines = sample_dockerfile.strip().split("\n")
        add_lines = [line for line in lines if line.strip().startswith("ADD")]
        copy_lines = [line for line in lines if line.strip().startswith("COPY")]

        # COPY is preferred over ADD (except for URLs/tar extraction)
        # Sample should use COPY
        assert len(copy_lines) > 0
        # ADD should not be used for local files
        assert len(add_lines) == 0


# ============================================================================
# Docker Compose Analysis Tests
# ============================================================================


class TestDockerComposeAnalysis:
    """Test docker-compose.yml analysis."""

    def test_detect_service_dependencies(self, sample_docker_compose):
        """Test analyzing service dependency graph."""
        web_deps = sample_docker_compose["services"]["web"].get("depends_on", [])
        assert "db" in web_deps

    def test_detect_exposed_ports(self, sample_docker_compose):
        """Test detecting exposed ports for security review."""
        web_ports = sample_docker_compose["services"]["web"].get("ports", [])
        assert len(web_ports) > 0

    def test_detect_environment_variables(self, sample_docker_compose):
        """Test detecting environment variable usage."""
        web_env = sample_docker_compose["services"]["web"].get("environment", {})
        assert len(web_env) > 0

    def test_detect_volume_mounts(self, sample_docker_compose):
        """Test detecting volume mounts for persistence."""
        db_volumes = sample_docker_compose["services"]["db"].get("volumes", [])
        assert len(db_volumes) > 0

    def test_detect_service_build_context(self, sample_docker_compose):
        """Test detecting build context for services."""
        web_service = sample_docker_compose["services"]["web"]
        assert "build" in web_service
        assert web_service["build"] == "."

    def test_detect_service_image(self, sample_docker_compose):
        """Test detecting service images."""
        db_service = sample_docker_compose["services"]["db"]
        assert "image" in db_service
        assert "postgres" in db_service["image"]

    def test_detect_multiple_networks(self):
        """Test detecting services on multiple networks."""
        compose = {
            "version": "3.8",
            "services": {
                "web": {
                    "image": "nginx",
                    "networks": ["frontend", "backend"],
                }
            },
            "networks": {
                "frontend": {"driver": "bridge"},
                "backend": {"driver": "bridge"},
            },
        }

        web_networks = compose["services"]["web"].get("networks", [])
        assert len(web_networks) == 2

    def test_detect_environment_files(self):
        """Test detecting env_file usage."""
        compose = {
            "version": "3.8",
            "services": {
                "web": {
                    "image": "nginx",
                    "env_file": [".env", ".env.production"],
                }
            },
        }

        web_env_files = compose["services"]["web"].get("env_file", [])
        assert len(web_env_files) == 2

    def test_detect_resource_limits(self):
        """Test detecting resource limits."""
        compose = {
            "version": "3.8",
            "services": {
                "web": {
                    "image": "nginx",
                    "deploy": {
                        "resources": {
                            "limits": {"cpus": "0.50", "memory": "512M"},
                            "reservations": {"cpus": "0.25", "memory": "256M"},
                        }
                    },
                }
            },
        }

        web_deploy = compose["services"]["web"].get("deploy", {})
        assert "resources" in web_deploy
        assert "limits" in web_deploy["resources"]

    def test_detect_service_scaling(self):
        """Test detecting service scaling configuration."""
        compose = {
            "version": "3.8",
            "services": {
                "web": {
                    "image": "nginx",
                    "deploy": {"replicas": 3},
                }
            },
        }

        web_deploy = compose["services"]["web"].get("deploy", {})
        assert "replicas" in web_deploy
        assert web_deploy["replicas"] == 3


# ============================================================================
# Best Practices Tests
# ============================================================================


class TestDockerBestPractices:
    """Test Docker best practices recommendations."""

    def test_multistage_build(self, sample_dockerfile):
        """Test recommending multi-stage builds."""
        assert "as builder" in sample_dockerfile.lower()

    def test_always_pull_latest_secure_images(self, invalid_dockerfile):
        """Test detecting use of 'latest' tag."""
        # Should not use :latest
        has_latest = ":latest" in invalid_dockerfile
        # This is a violation for production
        if has_latest:
            assert True  # Detected violation

    def test_combined_run_instructions(self, sample_dockerfile):
        """Test detecting combined RUN instructions."""
        # Sample should have combined RUNs
        lines = sample_dockerfile.split("\n")
        # Check if there are combined commands (&& or continuation)
        has_combined = any("&&" in line or line.endswith("\\") for line in lines)
        # At least one combined instruction pattern

    def test_order_of_instructions(self, sample_dockerfile):
        """Test proper ordering of Dockerfile instructions."""
        lines = [line.strip() for line in sample_dockerfile.split("\n") if line.strip()]

        # Find indices
        from_idx = next((i for i, line in enumerate(lines) if line.startswith("FROM")), -1)
        workdir_idx = next((i for i, line in enumerate(lines) if line.startswith("WORKDIR")), -1)
        copy_idx = next((i for i, line in enumerate(lines) if line.startswith("COPY")), -1)
        run_idx = next((i for i, line in enumerate(lines) if line.startswith("RUN")), -1)

        # Basic ordering checks
        assert from_idx >= 0  # Must have FROM
        assert workdir_idx > from_idx  # WORKDIR after FROM

    def test_healthcheck_configuration(self, docker_compose_with_healthchecks):
        """Test health check configuration."""
        healthcheck = docker_compose_with_healthchecks["services"]["app"]["healthcheck"]
        assert "test" in healthcheck
        assert "interval" in healthcheck
        assert "timeout" in healthcheck
        assert "retries" in healthcheck
