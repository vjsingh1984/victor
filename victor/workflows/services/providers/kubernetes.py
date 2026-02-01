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

"""Kubernetes service provider for K8s deployments.

Manages services as Kubernetes Deployments/StatefulSets with proper
pod lifecycle, health checks, and service discovery.

Supports:
- Local K8s (minikube, kind, Docker Desktop)
- Cloud-managed K8s (EKS, AKS, GKE)
- Custom K8s clusters

Example:
    from victor.workflows.services.credentials import get_credential_manager

    k8s_creds = get_credential_manager().get_kubernetes("my-cluster")
    provider = KubernetesServiceProvider(credentials=k8s_creds)

    config = ServiceConfig(
        name="postgres",
        provider="kubernetes",
        image="postgres:15",
        k8s_namespace="workflows",
    )

    handle = await provider.start(config)
    # ... use service ...
    await provider.stop(handle)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import tempfile
from datetime import datetime, timezone
from typing import Any, Optional

from victor.workflows.services.definition import (
    ServiceConfig,
    ServiceHandle,
    ServiceHealthError,
    ServiceStartError,
    ServiceState,
)
from victor.workflows.services.providers.base import BaseServiceProvider

logger = logging.getLogger(__name__)

# Optional kubernetes import
try:
    from kubernetes import client, config as k8s_config  # type: ignore[import-not-found]
    from kubernetes.client.rest import ApiException  # type: ignore[import-not-found]
    from kubernetes.stream import stream  # type: ignore[import-not-found]

    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    client = None
    k8s_config = None


class KubernetesServiceProvider(BaseServiceProvider):
    """Service provider for Kubernetes deployments.

    Creates Deployment + Service pairs for each workflow service.

    Attributes:
        api_client: Kubernetes API client
        namespace: Default namespace
        label_prefix: Label prefix for managed resources
    """

    def __init__(
        self,
        credentials: Optional[Any] = None,  # KubernetesCredentials
        kubeconfig_path: Optional[str] = None,
        context: Optional[str] = None,
        namespace: str = "default",
        label_prefix: str = "victor.ai",
    ):
        """Initialize Kubernetes provider.

        Args:
            credentials: KubernetesCredentials object
            kubeconfig_path: Path to kubeconfig file
            context: Kubernetes context to use
            namespace: Default namespace
            label_prefix: Label prefix for managed resources
        """
        if not K8S_AVAILABLE:
            raise ImportError("kubernetes not available. Install with: pip install kubernetes")

        self._credentials = credentials
        self._kubeconfig_path = kubeconfig_path
        self._context = context
        self._namespace = namespace
        self._label_prefix = label_prefix
        self._api_client: Optional[client.ApiClient] = None

    def _load_config(self) -> None:
        """Load Kubernetes configuration."""
        if self._credentials:
            # Use provided credentials
            if self._credentials.kubeconfig_content:
                # Write to temp file
                content = base64.b64decode(self._credentials.kubeconfig_content)
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".yaml", delete=False) as f:
                    f.write(content)
                    kubeconfig_path = f.name
                k8s_config.load_kube_config(
                    config_file=kubeconfig_path,
                    context=self._credentials.context or self._context,
                )
            elif self._credentials.kubeconfig_path:
                k8s_config.load_kube_config(
                    config_file=self._credentials.kubeconfig_path,
                    context=self._credentials.context or self._context,
                )
            elif self._credentials.token:
                # Use token-based auth
                configuration = client.Configuration()
                configuration.host = self._credentials.server
                configuration.api_key["authorization"] = self._credentials.token
                configuration.api_key_prefix["authorization"] = "Bearer"
                if self._credentials.certificate_authority:
                    configuration.ssl_ca_cert = self._credentials.certificate_authority
                else:
                    configuration.verify_ssl = False
                self._api_client = client.ApiClient(configuration)
                return
        elif self._kubeconfig_path:
            k8s_config.load_kube_config(config_file=self._kubeconfig_path, context=self._context)
        else:
            # Try in-cluster config, then default kubeconfig
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config(context=self._context)

    @property
    def core_v1(self) -> client.CoreV1Api:
        """Get Core V1 API client."""
        if self._api_client is None:
            self._load_config()
        return client.CoreV1Api(api_client=self._api_client)

    @property
    def apps_v1(self) -> client.AppsV1Api:
        """Get Apps V1 API client."""
        if self._api_client is None:
            self._load_config()
        return client.AppsV1Api(api_client=self._api_client)

    async def _do_start(self, config: ServiceConfig) -> ServiceHandle:
        """Create Kubernetes Deployment and Service."""
        if not config.image:
            raise ServiceStartError(config.name, "No image specified")

        handle = ServiceHandle.create(config)
        namespace = config.k8s_namespace or self._namespace
        loop = asyncio.get_event_loop()

        try:
            # Create deployment
            deployment = self._build_deployment(config, handle)
            await loop.run_in_executor(
                None,
                lambda: self.apps_v1.create_namespaced_deployment(
                    namespace=namespace, body=deployment
                ),
            )

            # Create service if ports defined
            if config.ports:
                service = self._build_service(config, handle)
                await loop.run_in_executor(
                    None,
                    lambda: self.core_v1.create_namespaced_service(
                        namespace=namespace, body=service
                    ),
                )

            handle.state = ServiceState.STARTING
            handle.metadata["namespace"] = namespace
            handle.metadata["deployment_name"] = handle.service_id

            # Wait for deployment to be ready
            await self._wait_for_deployment_ready(handle, namespace)

            # Get service endpoint
            if config.ports:
                endpoint = await self._get_service_endpoint(handle, namespace)
                handle.host = endpoint["host"]
                handle.ports = endpoint["ports"]

            logger.info(f"Started K8s deployment '{handle.service_id}' in namespace '{namespace}'")
            return handle

        except ApiException as e:
            handle.state = ServiceState.FAILED
            handle.error = str(e)
            raise ServiceStartError(config.name, f"Kubernetes API error: {e}")
        except Exception as e:
            handle.state = ServiceState.FAILED
            handle.error = str(e)
            raise ServiceStartError(config.name, str(e))

    def _build_deployment(
        self, config: ServiceConfig, handle: ServiceHandle
    ) -> client.V1Deployment:
        """Build Kubernetes Deployment spec."""
        labels = {
            f"{self._label_prefix}/managed": "true",
            f"{self._label_prefix}/service": config.name,
            f"{self._label_prefix}/id": handle.service_id,
        }

        # Container
        container = client.V1Container(
            name=config.name,
            image=config.image,
            ports=[client.V1ContainerPort(container_port=pm.container_port) for pm in config.ports],
            env=[client.V1EnvVar(name=k, value=v) for k, v in config.environment.items()],
        )

        if config.command:
            container.command = config.command
        if config.entrypoint:
            container.args = config.entrypoint

        # Resource limits
        if config.memory_limit or config.cpu_limit:
            limits = {}
            if config.memory_limit:
                limits["memory"] = config.memory_limit
            if config.cpu_limit:
                limits["cpu"] = str(config.cpu_limit)
            container.resources = client.V1ResourceRequirements(limits=limits)

        # Health check as liveness probe
        if config.health_check:
            hc = config.health_check
            if hc.type.value in ("tcp", "http", "https"):
                container.liveness_probe = client.V1Probe(
                    initial_delay_seconds=int(hc.start_period),
                    period_seconds=int(hc.interval),
                    timeout_seconds=int(hc.timeout),
                    failure_threshold=hc.retries,
                )
                if hc.type.value == "tcp":
                    container.liveness_probe.tcp_socket = client.V1TCPSocketAction(port=hc.port)
                else:
                    container.liveness_probe.http_get = client.V1HTTPGetAction(
                        path=hc.path, port=hc.port
                    )

        # Pod spec
        pod_spec = client.V1PodSpec(containers=[container])

        if config.k8s_service_account:
            pod_spec.service_account_name = config.k8s_service_account

        if config.k8s_node_selector:
            pod_spec.node_selector = config.k8s_node_selector

        # Deployment spec
        return client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=handle.service_id, labels=labels),
            spec=client.V1DeploymentSpec(
                replicas=config.k8s_replicas,
                selector=client.V1LabelSelector(match_labels=labels),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels=labels),
                    spec=pod_spec,
                ),
            ),
        )

    def _build_service(self, config: ServiceConfig, handle: ServiceHandle) -> client.V1Service:
        """Build Kubernetes Service spec."""
        labels = {
            f"{self._label_prefix}/managed": "true",
            f"{self._label_prefix}/service": config.name,
            f"{self._label_prefix}/id": handle.service_id,
        }

        ports = [
            client.V1ServicePort(
                name=f"port-{pm.container_port}",
                port=pm.container_port,
                target_port=pm.container_port,
                protocol=pm.protocol.upper(),
            )
            for pm in config.ports
        ]

        return client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=handle.service_id,
                labels=labels,
            ),
            spec=client.V1ServiceSpec(
                selector=labels,
                ports=ports,
                type="ClusterIP",  # Use ClusterIP for internal access
            ),
        )

    async def _wait_for_deployment_ready(
        self, handle: ServiceHandle, namespace: str, timeout: int = 300
    ) -> None:
        """Wait for deployment to have ready replicas."""
        loop = asyncio.get_event_loop()
        start_time = datetime.now(timezone.utc)
        deployment_name = handle.service_id

        while True:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed > timeout:
                raise ServiceHealthError(
                    handle.config.name,
                    "Deployment not ready within timeout",
                    attempts=int(elapsed / 5),
                )

            try:
                deployment = await loop.run_in_executor(
                    None,
                    lambda: self.apps_v1.read_namespaced_deployment(
                        name=deployment_name, namespace=namespace
                    ),
                )

                ready = deployment.status.ready_replicas or 0
                desired = deployment.spec.replicas or 1

                if ready >= desired:
                    logger.info(
                        f"Deployment '{deployment_name}' ready ({ready}/{desired} replicas)"
                    )
                    return

                logger.debug(f"Waiting for deployment '{deployment_name}': {ready}/{desired} ready")

            except ApiException as e:
                logger.warning(f"Error checking deployment status: {e}")

            await asyncio.sleep(5)

    async def _get_service_endpoint(self, handle: ServiceHandle, namespace: str) -> dict[str, Any]:
        """Get service endpoint (host and ports)."""
        loop = asyncio.get_event_loop()
        service_name = handle.service_id

        try:
            service = await loop.run_in_executor(
                None,
                lambda: self.core_v1.read_namespaced_service(
                    name=service_name, namespace=namespace
                ),
            )

            # For ClusterIP, use service DNS name
            host = f"{service_name}.{namespace}.svc.cluster.local"

            # Map ports
            ports = {}
            for port in service.spec.ports:
                ports[port.target_port] = port.port

            return {"host": host, "ports": ports}

        except ApiException as e:
            logger.warning(f"Failed to get service endpoint: {e}")
            return {"host": "localhost", "ports": {}}

    async def _do_stop(self, handle: ServiceHandle, grace_period: float) -> None:
        """Delete Kubernetes Deployment and Service."""
        loop = asyncio.get_event_loop()
        namespace = handle.metadata.get("namespace", self._namespace)
        deployment_name = handle.service_id

        try:
            # Delete deployment
            await loop.run_in_executor(
                None,
                lambda: self.apps_v1.delete_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=client.V1DeleteOptions(
                        grace_period_seconds=int(grace_period),
                        propagation_policy="Foreground",
                    ),
                ),
            )

            # Delete service
            if handle.config.ports:
                try:
                    await loop.run_in_executor(
                        None,
                        lambda: self.core_v1.delete_namespaced_service(
                            name=deployment_name, namespace=namespace
                        ),
                    )
                except ApiException:
                    pass

            logger.info(f"Deleted K8s deployment '{deployment_name}'")

        except ApiException as e:
            if e.status != 404:
                logger.error(f"Failed to delete deployment: {e}")
                raise

    async def _do_cleanup(self, handle: ServiceHandle) -> None:
        """Force delete Kubernetes resources."""
        await self._do_stop(handle, grace_period=0)

    async def get_logs(self, handle: ServiceHandle, tail: int = 100) -> str:
        """Get pod logs."""
        loop = asyncio.get_event_loop()
        namespace = handle.metadata.get("namespace", self._namespace)

        try:
            # Get pods for deployment
            pods = await loop.run_in_executor(
                None,
                lambda: self.core_v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=f"{self._label_prefix}/id={handle.service_id}",
                ),
            )

            if not pods.items:
                return "[No pods found]"

            # Get logs from first pod
            pod = pods.items[0]
            logs = await loop.run_in_executor(
                None,
                lambda: self.core_v1.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=namespace,
                    tail_lines=tail,
                ),
            )

            logs_str: str = logs
            return logs_str

        except ApiException as e:
            return f"[Error getting logs: {e}]"

    async def _run_command_in_service(self, handle: ServiceHandle, command: str) -> tuple[int, str]:
        """Execute command in pod."""
        loop = asyncio.get_event_loop()
        namespace = handle.metadata.get("namespace", self._namespace)

        try:
            # Get pods for deployment
            pods = await loop.run_in_executor(
                None,
                lambda: self.core_v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=f"{self._label_prefix}/id={handle.service_id}",
                ),
            )

            if not pods.items:
                return 1, "No pods found"

            pod = pods.items[0]

            # Execute command
            resp = await loop.run_in_executor(
                None,
                lambda: stream(
                    self.core_v1.connect_get_namespaced_pod_exec,
                    name=pod.metadata.name,
                    namespace=namespace,
                    command=["/bin/sh", "-c", command],
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                ),
            )

            return 0, resp

        except ApiException as e:
            return 1, str(e)

    async def cleanup_all(self, namespace: Optional[str] = None) -> int:
        """Clean up all managed resources in namespace."""
        loop = asyncio.get_event_loop()
        ns = namespace or self._namespace
        count = 0

        try:
            # Delete all managed deployments
            deployments = await loop.run_in_executor(
                None,
                lambda: self.apps_v1.list_namespaced_deployment(
                    namespace=ns,
                    label_selector=f"{self._label_prefix}/managed=true",
                ),
            )

            for dep in deployments.items:
                try:
                    from kubernetes.client import V1Deployment  # type: ignore[import-not-found]

                    def delete_deployment(d: V1Deployment = dep) -> None:
                        self.apps_v1.delete_namespaced_deployment(
                            name=d.metadata.name, namespace=ns
                        )

                    await loop.run_in_executor(
                        None,
                        delete_deployment,
                    )
                    count += 1
                except ApiException:
                    pass

            # Delete all managed services
            services = await loop.run_in_executor(
                None,
                lambda: self.core_v1.list_namespaced_service(
                    namespace=ns,
                    label_selector=f"{self._label_prefix}/managed=true",
                ),
            )

            for svc in services.items:
                try:
                    from kubernetes.client import V1Service

                    def delete_service(s: V1Service = svc) -> None:
                        self.core_v1.delete_namespaced_service(name=s.metadata.name, namespace=ns)

                    await loop.run_in_executor(
                        None,
                        delete_service,
                    )
                except ApiException:
                    pass

            logger.info(f"Cleaned up {count} K8s deployments in namespace '{ns}'")
            return count

        except ApiException as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
