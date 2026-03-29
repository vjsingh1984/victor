"""Focused async-bridge tests for the Kubernetes workflow service provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from victor.workflows.services.definition import ServiceConfig, ServiceHandle
from victor.workflows.services.providers import kubernetes as k8s_module


def _provider() -> k8s_module.KubernetesServiceProvider:
    fake_client = SimpleNamespace(
        CoreV1Api=lambda api_client=None: MagicMock(),
        AppsV1Api=lambda api_client=None: MagicMock(),
    )
    with patch.object(k8s_module, "K8S_AVAILABLE", True), patch.object(
        k8s_module, "client", fake_client
    ):
        provider = k8s_module.KubernetesServiceProvider()
    provider._api_client = object()
    return provider


@pytest.mark.asyncio
async def test_kubernetes_provider_get_logs_uses_to_thread() -> None:
    core_v1 = MagicMock()
    pod = SimpleNamespace(metadata=SimpleNamespace(name="pod-123"))
    core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[pod])
    core_v1.read_namespaced_pod_log.return_value = "log line\n"
    provider = _provider()

    handle = ServiceHandle.create(ServiceConfig(name="svc", image="redis:7"))
    handle.metadata["namespace"] = "workflows"

    async def call_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with (
        patch.object(
            k8s_module.KubernetesServiceProvider,
            "core_v1",
            new_callable=PropertyMock,
            return_value=core_v1,
        ),
        patch(
            "victor.workflows.services.providers.kubernetes.asyncio.to_thread",
            side_effect=call_to_thread,
        ) as mock_to_thread,
    ):
        logs = await provider.get_logs(handle, tail=25)

    assert logs == "log line\n"
    assert mock_to_thread.await_count == 2
    first = mock_to_thread.await_args_list[0]
    second = mock_to_thread.await_args_list[1]
    assert first.args[0] is core_v1.list_namespaced_pod
    assert first.kwargs == {
        "namespace": "workflows",
        "label_selector": f"victor.ai/id={handle.service_id}",
    }
    assert second.args[0] is core_v1.read_namespaced_pod_log
    assert second.kwargs == {
        "name": "pod-123",
        "namespace": "workflows",
        "tail_lines": 25,
    }


@pytest.mark.asyncio
async def test_kubernetes_provider_run_command_uses_to_thread() -> None:
    core_v1 = MagicMock()
    pod = SimpleNamespace(metadata=SimpleNamespace(name="pod-456"))
    core_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[pod])
    core_v1.connect_get_namespaced_pod_exec = MagicMock()
    provider = _provider()

    handle = ServiceHandle.create(ServiceConfig(name="svc", image="redis:7"))
    handle.metadata["namespace"] = "workflows"

    stream_mock = MagicMock(return_value="command output")

    async def call_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with (
        patch.object(
            k8s_module.KubernetesServiceProvider,
            "core_v1",
            new_callable=PropertyMock,
            return_value=core_v1,
        ),
        patch.object(k8s_module, "stream", stream_mock),
        patch(
            "victor.workflows.services.providers.kubernetes.asyncio.to_thread",
            side_effect=call_to_thread,
        ) as mock_to_thread,
    ):
        exit_code, output = await provider._run_command_in_service(handle, "echo ok")

    assert exit_code == 0
    assert output == "command output"
    assert mock_to_thread.await_count == 2
    second = mock_to_thread.await_args_list[1]
    assert second.args[0] is stream_mock
    assert second.args[1] is core_v1.connect_get_namespaced_pod_exec
    assert second.kwargs["name"] == "pod-456"
    assert second.kwargs["namespace"] == "workflows"
    assert second.kwargs["command"] == ["/bin/sh", "-c", "echo ok"]
