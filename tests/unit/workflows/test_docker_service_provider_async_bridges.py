"""Focused async-bridge tests for the Docker workflow service provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from victor.workflows.services.definition import ServiceConfig, ServiceHandle
from victor.workflows.services.providers import docker as docker_module


def _provider() -> docker_module.DockerServiceProvider:
    with patch.object(docker_module, "DOCKER_AVAILABLE", True):
        provider = docker_module.DockerServiceProvider()
    provider._client = MagicMock()
    return provider


@pytest.mark.asyncio
async def test_docker_provider_get_logs_uses_to_thread() -> None:
    provider = _provider()
    handle = ServiceHandle.create(ServiceConfig(name="svc", image="redis:7"))
    handle.container_id = "container-123"

    container = MagicMock()
    container.logs.return_value = b"log line\n"
    provider._client.containers.get.return_value = container

    async def call_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch(
        "victor.workflows.services.providers.docker.asyncio.to_thread",
        side_effect=call_to_thread,
    ) as mock_to_thread:
        logs = await provider.get_logs(handle, tail=25)

    assert logs == "log line\n"
    assert mock_to_thread.await_count == 2
    first = mock_to_thread.await_args_list[0]
    second = mock_to_thread.await_args_list[1]
    assert first.args == (provider.client.containers.get, "container-123")
    assert second.args[0] is container.logs
    assert second.kwargs == {"tail": 25, "timestamps": True}


@pytest.mark.asyncio
async def test_docker_provider_run_command_uses_to_thread() -> None:
    provider = _provider()
    handle = ServiceHandle.create(ServiceConfig(name="svc", image="redis:7"))
    handle.container_id = "container-456"

    container = MagicMock()
    container.exec_run.return_value = SimpleNamespace(
        exit_code=0, output=(b"ok\n", b"")
    )
    provider._client.containers.get.return_value = container

    async def call_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch(
        "victor.workflows.services.providers.docker.asyncio.to_thread",
        side_effect=call_to_thread,
    ) as mock_to_thread:
        exit_code, output = await provider._run_command_in_service(handle, "echo ok")

    assert exit_code == 0
    assert output == "ok\n"
    assert mock_to_thread.await_count == 2
    first = mock_to_thread.await_args_list[0]
    second = mock_to_thread.await_args_list[1]
    assert first.args == (provider.client.containers.get, "container-456")
    assert second.args[0] is container.exec_run
    assert second.args[1] == "echo ok"
    assert second.kwargs == {"demux": True}
