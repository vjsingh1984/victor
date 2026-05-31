"""Focused async-bridge tests for workflow deployment handlers."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.workflows.deployment import (
    DockerConfig,
    DockerDeploymentHandler,
    ECSConfig,
    ECSDeploymentHandler,
    KubernetesConfig,
    KubernetesDeploymentHandler,
)


@pytest.mark.asyncio
async def test_docker_deployment_prepare_uses_to_thread() -> None:
    handler = DockerDeploymentHandler(
        DockerConfig(
            image="victor-worker:test",
            environment={"API_KEY": "x"},
            volumes={"/tmp": "/workspace"},
            network="bridge",
            resource_limits={"memory": "512m", "cpu": "1.5"},
        )
    )
    client = MagicMock()
    container = MagicMock()
    container.id = "container-1234567890"
    client.containers.run.return_value = container
    handler._client = client

    async def call_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch(
        "victor.workflows.deployment.asyncio.to_thread",
        side_effect=call_to_thread,
    ) as mock_to_thread:
        await handler.prepare(workflow=MagicMock(), config=MagicMock())

    assert handler.container_id == "container-1234567890"
    mock_to_thread.assert_awaited_once()
    called = mock_to_thread.await_args
    assert called.args == (client.containers.run, "victor-worker:test")
    assert called.kwargs["environment"] == {"API_KEY": "x"}
    assert called.kwargs["volumes"] == {"/tmp": {"bind": "/workspace", "mode": "rw"}}
    assert called.kwargs["network_mode"] == "bridge"
    assert called.kwargs["mem_limit"] == "512m"
    assert called.kwargs["cpu_quota"] == 150000
    assert called.kwargs["command"] == "sleep infinity"


@pytest.mark.asyncio
async def test_docker_deployment_execute_node_uses_to_thread() -> None:
    handler = DockerDeploymentHandler(DockerConfig(image="victor-worker:test"))
    handler.container_id = "container-abcdef123456"

    client = MagicMock()
    container = MagicMock()
    container.exec_run.return_value = (
        0,
        json.dumps({"node_id": "step1", "state": {"value": 2}}).encode(),
    )
    client.containers.get.return_value = container
    handler._client = client

    node = MagicMock()
    node.id = "step1"

    async def call_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch(
        "victor.workflows.deployment.asyncio.to_thread",
        side_effect=call_to_thread,
    ) as mock_to_thread:
        result = await handler.execute_node(node, {"value": 2})

    assert result == {"value": 2}
    mock_to_thread.assert_awaited_once()
    called = mock_to_thread.await_args
    assert called.args[0] is container.exec_run
    assert called.args[1][0] == "python"


@pytest.mark.asyncio
async def test_kubernetes_deployment_execute_node_uses_to_thread() -> None:
    handler = KubernetesDeploymentHandler(
        KubernetesConfig(namespace="workflows", image="victor-worker:test")
    )
    handler.pod_name = "pod-123"

    api = MagicMock()
    api.connect_get_namespaced_pod_exec = MagicMock()
    handler._core_v1 = api

    node = MagicMock()
    node.id = "step1"

    stream_mock = MagicMock(return_value=json.dumps({"node_id": "step1", "state": {"value": 3}}))
    kubernetes_module = ModuleType("kubernetes")
    stream_module = ModuleType("kubernetes.stream")
    stream_module.stream = stream_mock
    kubernetes_module.stream = stream_module

    async def call_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with (
        patch.dict(
            sys.modules,
            {"kubernetes": kubernetes_module, "kubernetes.stream": stream_module},
        ),
        patch(
            "victor.workflows.deployment.asyncio.to_thread",
            side_effect=call_to_thread,
        ) as mock_to_thread,
    ):
        result = await handler.execute_node(node, {"value": 3})

    assert result == {"value": 3}
    mock_to_thread.assert_awaited_once()
    called = mock_to_thread.await_args
    assert called.args[0] is stream_mock
    assert called.args[1] is api.connect_get_namespaced_pod_exec
    assert called.args[2] == "pod-123"
    assert called.args[3] == "workflows"
    assert called.kwargs["container"] == "workflow-runner"


@pytest.mark.asyncio
async def test_ecs_deployment_prepare_uses_to_thread() -> None:
    handler = ECSDeploymentHandler(
        ECSConfig(
            cluster="prod-cluster",
            task_definition="victor-worker",
            launch_type="FARGATE",
            subnets=["subnet-123"],
            security_groups=["sg-123"],
            assign_public_ip=True,
        )
    )
    client = MagicMock()
    client.run_task.return_value = {"tasks": [{"taskArn": "arn:aws:ecs:task/123"}]}
    handler._ecs_client = client

    async def call_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with (
        patch.object(handler, "_wait_for_task_running", new=AsyncMock()) as mock_wait,
        patch(
            "victor.workflows.deployment.asyncio.to_thread",
            side_effect=call_to_thread,
        ) as mock_to_thread,
    ):
        await handler.prepare(workflow=MagicMock(), config=MagicMock())

    assert handler.task_arn == "arn:aws:ecs:task/123"
    mock_wait.assert_awaited_once()
    mock_to_thread.assert_awaited_once()
    called = mock_to_thread.await_args
    assert called.args[0] is client.run_task
    assert called.kwargs == {
        "cluster": "prod-cluster",
        "taskDefinition": "victor-worker",
        "launchType": "FARGATE",
        "count": 1,
        "networkConfiguration": {
            "awsvpcConfiguration": {
                "subnets": ["subnet-123"],
                "securityGroups": ["sg-123"],
                "assignPublicIp": "ENABLED",
            }
        },
    }
