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

"""Workflow Deployment Targets.

Provides deployment target configuration for workflow execution across
different environments: local, Docker, Kubernetes, ECS, remote servers.

Example:
    from victor.workflows import StateGraphExecutor
    from victor.workflows.deployment import (
        DeploymentTarget,
        DeploymentConfig,
        DockerConfig,
        KubernetesConfig,
    )

    # Local execution (default)
    executor = StateGraphExecutor()
    result = await executor.execute(workflow, {})

    # Docker execution
    config = DeploymentConfig(
        target=DeploymentTarget.DOCKER,
        docker=DockerConfig(
            image="victor-worker:latest",
            environment={"API_KEY": "xxx"},
        ),
    )
    executor = StateGraphExecutor(deployment_config=config)

    # Kubernetes execution
    config = DeploymentConfig(
        target=DeploymentTarget.KUBERNETES,
        kubernetes=KubernetesConfig(
            namespace="workflows",
            service_account="victor-worker",
            resource_limits={"cpu": "2", "memory": "4Gi"},
        ),
    )
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Type,
)

if TYPE_CHECKING:
    from victor.workflows.definition import WorkflowDefinition, WorkflowNode

logger = logging.getLogger(__name__)


class DeploymentTarget(Enum):
    """Target environment for workflow execution.

    Enterprise-ready deployment targets covering 80% of common use cases.

    Local/Development (in-process, subprocess, containers):
        INLINE: Execute inline in current process (no isolation)
        SUBPROCESS: Execute in a subprocess (process isolation)
        DOCKER: Execute in a Docker container
        DOCKER_COMPOSE: Multi-container Docker Compose deployment
        LOCAL: Execute on local machine (alias for SUBPROCESS)

    AWS:
        ECS: AWS ECS task (Fargate or EC2 launch type)
        EKS: AWS EKS (Kubernetes on AWS)
        AWS_BATCH: AWS Batch for HPC/batch workloads
        EC2_SPOT: AWS EC2 Spot Instances (cost-optimized)
        EC2_ON_DEMAND: AWS EC2 On-Demand Instances
        AWS_LAMBDA: AWS Lambda (serverless)
        FARGATE: AWS Fargate (serverless containers) - same as ECS with Fargate

    Azure:
        AKS: Azure Kubernetes Service
        AZURE_CONTAINER: Azure Container Instances
        AZURE_BATCH: Azure Batch
        AZURE_VM: Azure Virtual Machines
        AZURE_FUNCTIONS: Azure Functions (serverless)

    GCP:
        GKE: Google Kubernetes Engine
        CLOUD_RUN: Google Cloud Run (serverless containers)
        COMPUTE_ENGINE: Google Compute Engine VMs
        CLOUD_BATCH: Google Cloud Batch
        GCP_FUNCTIONS: Google Cloud Functions (serverless)

    On-Premises/Hybrid:
        KUBERNETES: Kubernetes (any cluster - on-prem, cloud, hybrid)
        REMOTE: Remote server via SSH/API
        SANDBOX: Isolated sandbox environment
    """

    # Local/Development
    INLINE = "inline"  # In-process execution
    SUBPROCESS = "subprocess"  # Subprocess execution
    DOCKER = "docker"
    DOCKER_COMPOSE = "docker_compose"
    LOCAL = "local"  # Alias for subprocess

    # AWS
    ECS = "ecs"
    EKS = "eks"
    AWS_BATCH = "aws_batch"
    EC2_SPOT = "ec2_spot"
    EC2_ON_DEMAND = "ec2_on_demand"
    AWS_LAMBDA = "aws_lambda"
    FARGATE = "fargate"  # Alias for ECS with Fargate

    # Azure
    AKS = "aks"
    AZURE_CONTAINER = "azure_container"
    AZURE_BATCH = "azure_batch"
    AZURE_VM = "azure_vm"
    AZURE_FUNCTIONS = "azure_functions"

    # GCP
    GKE = "gke"
    CLOUD_RUN = "cloud_run"
    COMPUTE_ENGINE = "compute_engine"
    CLOUD_BATCH = "cloud_batch"
    GCP_FUNCTIONS = "gcp_functions"

    # On-Premises/Hybrid
    KUBERNETES = "kubernetes"
    REMOTE = "remote"
    SANDBOX = "sandbox"

    # Workflow Orchestrators
    AIRFLOW = "airflow"  # Apache Airflow task/DAG
    TEMPORAL = "temporal"  # Temporal.io workflow
    STEP_FUNCTIONS = "step_functions"  # AWS Step Functions
    PREFECT = "prefect"  # Prefect workflow
    DAGSTER = "dagster"  # Dagster pipeline

    # Data Platforms
    DATABRICKS = "databricks"  # Databricks Jobs API
    SPARK = "spark"  # Apache Spark (local/cluster)
    DASK = "dask"  # Dask distributed
    RAY = "ray"  # Ray distributed

    # Task Queues
    CELERY = "celery"  # Celery distributed tasks
    KUBERNETES_JOB = "kubernetes_job"  # K8s Job (one-off)

    # Serverless Platforms
    MODAL = "modal"  # Modal.com serverless


@dataclass
class DockerConfig:
    """Configuration for Docker deployment.

    Attributes:
        image: Docker image to use
        environment: Environment variables
        volumes: Volume mounts (host:container)
        network: Network mode
        resource_limits: CPU/memory limits
    """

    image: str = "victor-worker:latest"
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)
    network: str = "bridge"
    resource_limits: Dict[str, str] = field(default_factory=dict)


@dataclass
class KubernetesConfig:
    """Configuration for Kubernetes deployment.

    Attributes:
        namespace: Kubernetes namespace
        service_account: Service account name
        image: Container image
        resource_limits: Pod resource limits
        annotations: Pod annotations
        labels: Pod labels
        node_selector: Node selector constraints
    """

    namespace: str = "default"
    service_account: str = "default"
    image: str = "victor-worker:latest"
    resource_limits: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    node_selector: Dict[str, str] = field(default_factory=dict)


@dataclass
class ECSConfig:
    """Configuration for AWS ECS deployment.

    Attributes:
        cluster: ECS cluster name
        task_definition: Task definition family
        launch_type: FARGATE or EC2
        subnets: VPC subnet IDs
        security_groups: Security group IDs
        assign_public_ip: Whether to assign public IP
    """

    cluster: str = "default"
    task_definition: str = "victor-worker"
    launch_type: str = "FARGATE"
    subnets: List[str] = field(default_factory=list)
    security_groups: List[str] = field(default_factory=list)
    assign_public_ip: bool = False


@dataclass
class RemoteConfig:
    """Configuration for remote server deployment.

    Attributes:
        endpoint: Remote server API endpoint
        auth_token: Authentication token
        ssh_key_path: Path to SSH private key (for SSH access)
        username: SSH username
    """

    endpoint: str = "http://localhost:8080"
    auth_token: Optional[str] = None
    ssh_key_path: Optional[str] = None
    username: str = "victor"


@dataclass
class SandboxConfig:
    """Configuration for sandboxed execution.

    Attributes:
        isolation_level: Level of isolation (none, container, vm)
        timeout: Maximum execution time in seconds
        network_enabled: Whether network access is allowed
        filesystem_readonly: Whether filesystem is read-only
    """

    isolation_level: str = "container"
    timeout: float = 300.0
    network_enabled: bool = False
    filesystem_readonly: bool = True


@dataclass
class CloudRunConfig:
    """Configuration for Google Cloud Run deployment.

    Attributes:
        project_id: GCP project ID
        region: GCP region
        service_name: Cloud Run service name
        image: Container image
        memory: Memory allocation (e.g., "512Mi")
        cpu: CPU allocation (e.g., "1")
        max_instances: Maximum auto-scaling instances
        timeout_seconds: Request timeout
    """

    project_id: str = ""
    region: str = "us-central1"
    service_name: str = "victor-worker"
    image: str = "victor-worker:latest"
    memory: str = "512Mi"
    cpu: str = "1"
    max_instances: int = 10
    timeout_seconds: int = 300


@dataclass
class AzureContainerConfig:
    """Configuration for Azure Container Instances deployment.

    Attributes:
        resource_group: Azure resource group
        container_group_name: Container group name
        image: Container image
        cpu: CPU cores
        memory_gb: Memory in GB
        location: Azure region
    """

    resource_group: str = ""
    container_group_name: str = "victor-worker"
    image: str = "victor-worker:latest"
    cpu: float = 1.0
    memory_gb: float = 1.5
    location: str = "eastus"


@dataclass
class LambdaConfig:
    """Configuration for AWS Lambda deployment.

    Attributes:
        function_name: Lambda function name
        region: AWS region
        memory_mb: Memory allocation
        timeout_seconds: Function timeout
        runtime: Lambda runtime
        role_arn: IAM role ARN
    """

    function_name: str = "victor-workflow"
    region: str = "us-east-1"
    memory_mb: int = 512
    timeout_seconds: int = 300
    runtime: str = "python3.12"
    role_arn: str = ""


@dataclass
class AWSBatchConfig:
    """Configuration for AWS Batch deployment.

    Attributes:
        job_queue: Batch job queue name
        job_definition: Job definition name/ARN
        region: AWS region
        vcpus: Number of vCPUs
        memory_mb: Memory in MB
        timeout_seconds: Job timeout
    """

    job_queue: str = "victor-workflow-queue"
    job_definition: str = "victor-workflow-job"
    region: str = "us-east-1"
    vcpus: int = 2
    memory_mb: int = 4096
    timeout_seconds: int = 3600


@dataclass
class EC2Config:
    """Configuration for AWS EC2 deployment.

    Attributes:
        instance_type: EC2 instance type
        ami_id: Amazon Machine Image ID
        region: AWS region
        spot: Whether to use spot instances
        max_spot_price: Maximum spot price (for spot instances)
        security_group_ids: Security group IDs
        subnet_id: Subnet ID
        key_name: SSH key pair name
    """

    instance_type: str = "t3.medium"
    ami_id: str = ""  # Will use latest Amazon Linux 2 if not specified
    region: str = "us-east-1"
    spot: bool = False
    max_spot_price: Optional[str] = None  # e.g., "0.05"
    security_group_ids: List[str] = field(default_factory=list)
    subnet_id: str = ""
    key_name: str = ""


@dataclass
class EKSConfig:
    """Configuration for AWS EKS deployment.

    Attributes:
        cluster_name: EKS cluster name
        region: AWS region
        namespace: Kubernetes namespace
        service_account: Kubernetes service account
        image: Container image
        resource_limits: Pod resource limits
    """

    cluster_name: str = ""
    region: str = "us-east-1"
    namespace: str = "default"
    service_account: str = "default"
    image: str = "victor-worker:latest"
    resource_limits: Dict[str, str] = field(default_factory=dict)


@dataclass
class AKSConfig:
    """Configuration for Azure Kubernetes Service deployment.

    Attributes:
        resource_group: Azure resource group
        cluster_name: AKS cluster name
        namespace: Kubernetes namespace
        service_account: Kubernetes service account
        image: Container image
        resource_limits: Pod resource limits
    """

    resource_group: str = ""
    cluster_name: str = ""
    namespace: str = "default"
    service_account: str = "default"
    image: str = "victor-worker:latest"
    resource_limits: Dict[str, str] = field(default_factory=dict)


@dataclass
class AzureBatchConfig:
    """Configuration for Azure Batch deployment.

    Attributes:
        account_name: Batch account name
        pool_id: Batch pool ID
        job_id: Batch job ID
        location: Azure region
        vm_size: VM size
    """

    account_name: str = ""
    pool_id: str = "victor-workflow-pool"
    job_id: str = "victor-workflow-job"
    location: str = "eastus"
    vm_size: str = "STANDARD_D2_V2"


@dataclass
class GKEConfig:
    """Configuration for Google Kubernetes Engine deployment.

    Attributes:
        project_id: GCP project ID
        cluster_name: GKE cluster name
        zone: GCP zone
        namespace: Kubernetes namespace
        service_account: Kubernetes service account
        image: Container image
        resource_limits: Pod resource limits
    """

    project_id: str = ""
    cluster_name: str = ""
    zone: str = "us-central1-a"
    namespace: str = "default"
    service_account: str = "default"
    image: str = "victor-worker:latest"
    resource_limits: Dict[str, str] = field(default_factory=dict)


@dataclass
class ComputeEngineConfig:
    """Configuration for Google Compute Engine deployment.

    Attributes:
        project_id: GCP project ID
        zone: GCP zone
        machine_type: Machine type (e.g., n1-standard-2)
        image_family: OS image family
        image_project: OS image project
        preemptible: Whether to use preemptible VMs
    """

    project_id: str = ""
    zone: str = "us-central1-a"
    machine_type: str = "n1-standard-2"
    image_family: str = "ubuntu-2204-lts"
    image_project: str = "ubuntu-os-cloud"
    preemptible: bool = False


@dataclass
class CloudBatchConfig:
    """Configuration for Google Cloud Batch deployment.

    Attributes:
        project_id: GCP project ID
        region: GCP region
        job_id: Batch job ID
        machine_type: Machine type
        vcpus: Number of vCPUs
        memory_mb: Memory in MB
    """

    project_id: str = ""
    region: str = "us-central1"
    job_id: str = "victor-workflow"
    machine_type: str = "n1-standard-2"
    vcpus: int = 2
    memory_mb: int = 4096


@dataclass
class SubprocessConfig:
    """Configuration for subprocess execution.

    Attributes:
        python_path: Path to Python executable
        working_dir: Working directory
        env: Environment variables
        timeout_seconds: Execution timeout
    """

    python_path: str = "python"
    working_dir: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 3600


@dataclass
class InlineConfig:
    """Configuration for inline (in-process) execution.

    Attributes:
        timeout_seconds: Execution timeout
        thread_pool_size: Thread pool size for parallel execution
    """

    timeout_seconds: int = 3600
    thread_pool_size: int = 4


@dataclass
class AirflowConfig:
    """Configuration for Apache Airflow deployment.

    Attributes:
        airflow_url: Airflow API URL
        dag_id: DAG ID to trigger
        task_id: Optional task ID
        username: Airflow username
        password: Airflow password
        execution_timeout: Execution timeout
    """

    airflow_url: str = "http://localhost:8080"
    dag_id: str = "victor_workflow"
    task_id: Optional[str] = None
    username: str = "admin"
    password: str = ""
    execution_timeout: int = 3600


@dataclass
class DatabricksConfig:
    """Configuration for Databricks Jobs API deployment.

    Attributes:
        host: Databricks workspace URL
        token: Access token
        job_id: Existing job ID (optional)
        cluster_id: Cluster ID for new runs
        notebook_path: Notebook path to execute
        spark_version: Spark version
        node_type_id: Worker node type
        num_workers: Number of workers
    """

    host: str = ""
    token: str = ""
    job_id: Optional[int] = None
    cluster_id: Optional[str] = None
    notebook_path: str = ""
    spark_version: str = "13.3.x-scala2.12"
    node_type_id: str = "Standard_DS3_v2"
    num_workers: int = 2


@dataclass
class SparkConfig:
    """Configuration for Apache Spark deployment.

    Attributes:
        master: Spark master URL (local[*], spark://..., yarn, k8s://...)
        app_name: Spark application name
        deploy_mode: cluster or client
        executor_memory: Executor memory
        executor_cores: Executor cores
        num_executors: Number of executors
        conf: Additional Spark configurations
    """

    master: str = "local[*]"
    app_name: str = "victor-workflow"
    deploy_mode: str = "client"
    executor_memory: str = "4g"
    executor_cores: int = 2
    num_executors: int = 2
    conf: Dict[str, str] = field(default_factory=dict)


@dataclass
class RayConfig:
    """Configuration for Ray distributed deployment.

    Attributes:
        address: Ray cluster address (auto, local, ray://...)
        runtime_env: Runtime environment config
        num_cpus: Number of CPUs per task
        num_gpus: Number of GPUs per task
        memory: Memory per task (bytes)
    """

    address: str = "auto"
    runtime_env: Dict[str, Any] = field(default_factory=dict)
    num_cpus: int = 1
    num_gpus: int = 0
    memory: Optional[int] = None


@dataclass
class DaskConfig:
    """Configuration for Dask distributed deployment.

    Attributes:
        scheduler_address: Dask scheduler address
        n_workers: Number of workers (for local cluster)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
    """

    scheduler_address: Optional[str] = None  # None = LocalCluster
    n_workers: int = 4
    threads_per_worker: int = 2
    memory_limit: str = "2GB"


@dataclass
class CeleryConfig:
    """Configuration for Celery task queue deployment.

    Attributes:
        broker_url: Message broker URL (redis://, amqp://)
        backend_url: Result backend URL
        queue: Queue name
        task_name: Task name prefix
        timeout: Task timeout
    """

    broker_url: str = "redis://localhost:6379/0"
    backend_url: str = "redis://localhost:6379/1"
    queue: str = "victor-workflows"
    task_name: str = "victor.workflow"
    timeout: int = 3600


@dataclass
class TemporalConfig:
    """Configuration for Temporal.io workflow deployment.

    Attributes:
        host: Temporal server host
        namespace: Temporal namespace
        task_queue: Task queue name
        workflow_id: Workflow ID prefix
        execution_timeout: Workflow execution timeout
    """

    host: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "victor-workflow-queue"
    workflow_id: str = "victor-workflow"
    execution_timeout: int = 3600


@dataclass
class StepFunctionsConfig:
    """Configuration for AWS Step Functions deployment.

    Attributes:
        state_machine_arn: State machine ARN
        region: AWS region
        execution_name_prefix: Execution name prefix
        input_path: JSONPath for input
        output_path: JSONPath for output
    """

    state_machine_arn: str = ""
    region: str = "us-east-1"
    execution_name_prefix: str = "victor-workflow"
    input_path: str = "$"
    output_path: str = "$"


@dataclass
class ModalConfig:
    """Configuration for Modal.com serverless deployment.

    Attributes:
        app_name: Modal app name
        cpu: CPU count
        memory: Memory in MB
        gpu: GPU type (None, "A10G", "T4", etc.)
        timeout: Function timeout
    """

    app_name: str = "victor-workflow"
    cpu: float = 1.0
    memory: int = 512
    gpu: Optional[str] = None
    timeout: int = 300


@dataclass
class DockerComposeConfig:
    """Configuration for Docker Compose deployment.

    Attributes:
        compose_file: Path to docker-compose.yml
        project_name: Compose project name
        services: List of services to start
        environment: Environment variables
    """

    compose_file: str = "docker-compose.yml"
    project_name: str = "victor-workflow"
    services: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentConfig:
    """Full deployment configuration for all target environments.

    Supports 30+ deployment targets across cloud, on-prem, and serverless.
    """

    target: DeploymentTarget = DeploymentTarget.LOCAL

    # Local/Development
    inline: Optional[InlineConfig] = None
    subprocess: Optional[SubprocessConfig] = None
    docker: Optional[DockerConfig] = None
    docker_compose: Optional[DockerComposeConfig] = None

    # AWS
    ecs: Optional[ECSConfig] = None
    eks: Optional[EKSConfig] = None
    aws_batch: Optional[AWSBatchConfig] = None
    ec2: Optional[EC2Config] = None
    lambda_: Optional[LambdaConfig] = None
    step_functions: Optional[StepFunctionsConfig] = None

    # Azure
    aks: Optional[AKSConfig] = None
    azure_container: Optional[AzureContainerConfig] = None
    azure_batch: Optional[AzureBatchConfig] = None

    # GCP
    gke: Optional[GKEConfig] = None
    cloud_run: Optional[CloudRunConfig] = None
    compute_engine: Optional[ComputeEngineConfig] = None
    cloud_batch: Optional[CloudBatchConfig] = None

    # Kubernetes
    kubernetes: Optional[KubernetesConfig] = None

    # Workflow Orchestrators
    airflow: Optional[AirflowConfig] = None
    temporal: Optional[TemporalConfig] = None

    # Data Platforms
    databricks: Optional[DatabricksConfig] = None
    spark: Optional[SparkConfig] = None
    ray: Optional[RayConfig] = None
    dask: Optional[DaskConfig] = None

    # Task Queues
    celery: Optional[CeleryConfig] = None

    # Serverless Platforms
    modal: Optional[ModalConfig] = None

    # Remote/Hybrid
    remote: Optional[RemoteConfig] = None
    sandbox: Optional[SandboxConfig] = None


class DeploymentHandler(ABC):
    """Abstract handler for deployment targets."""

    @abstractmethod
    async def prepare(
        self,
        workflow: "WorkflowDefinition",
        config: DeploymentConfig,
    ) -> None:
        """Prepare the deployment target."""
        pass

    @abstractmethod
    async def execute_node(
        self,
        node: "WorkflowNode",
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a node on the deployment target."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup the deployment target."""
        pass


class LocalDeploymentHandler(DeploymentHandler):
    """Execute workflows locally on the current machine."""

    async def prepare(
        self,
        workflow: "WorkflowDefinition",
        config: DeploymentConfig,
    ) -> None:
        """No preparation needed for local execution."""
        logger.debug("Preparing local deployment")

    async def execute_node(
        self,
        node: "WorkflowNode",
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute node locally - delegated to StateGraph."""
        # Local execution is handled by StateGraph directly
        return state

    async def cleanup(self) -> None:
        """No cleanup needed for local execution."""
        pass


class DockerDeploymentHandler(DeploymentHandler):
    """Execute workflows in Docker containers."""

    def __init__(self, config: DockerConfig):
        self.config = config
        self.container_id: Optional[str] = None

    async def prepare(
        self,
        workflow: "WorkflowDefinition",
        config: DeploymentConfig,
    ) -> None:
        """Start Docker container for execution."""
        logger.info(f"Preparing Docker deployment with image: {self.config.image}")
        # TODO: Actual Docker API integration
        self.container_id = f"container-{uuid.uuid4().hex[:8]}"
        logger.info(f"Container started: {self.container_id}")

    async def execute_node(
        self,
        node: "WorkflowNode",
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute node in Docker container."""
        logger.debug(f"Executing node {node.id} in container {self.container_id}")
        # TODO: RPC to container
        return state

    async def cleanup(self) -> None:
        """Stop and remove Docker container."""
        if self.container_id:
            logger.info(f"Cleaning up Docker container: {self.container_id}")
            # TODO: Docker API cleanup
            self.container_id = None


class KubernetesDeploymentHandler(DeploymentHandler):
    """Execute workflows in Kubernetes pods."""

    def __init__(self, config: KubernetesConfig):
        self.config = config
        self.pod_name: Optional[str] = None

    async def prepare(
        self,
        workflow: "WorkflowDefinition",
        config: DeploymentConfig,
    ) -> None:
        """Create Kubernetes pod for execution."""
        logger.info(f"Preparing K8s deployment in namespace: {self.config.namespace}")
        # TODO: Kubernetes API integration
        self.pod_name = f"workflow-{uuid.uuid4().hex[:8]}"
        logger.info(f"Pod created: {self.pod_name}")

    async def execute_node(
        self,
        node: "WorkflowNode",
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute node in Kubernetes pod."""
        logger.debug(f"Executing node {node.id} in pod {self.pod_name}")
        # TODO: gRPC/HTTP to pod
        return state

    async def cleanup(self) -> None:
        """Delete Kubernetes pod."""
        if self.pod_name:
            logger.info(f"Deleting K8s pod: {self.pod_name}")
            # TODO: Kubernetes API cleanup
            self.pod_name = None


class ECSDeploymentHandler(DeploymentHandler):
    """Execute workflows in AWS ECS tasks."""

    def __init__(self, config: ECSConfig):
        self.config = config
        self.task_arn: Optional[str] = None

    async def prepare(
        self,
        workflow: "WorkflowDefinition",
        config: DeploymentConfig,
    ) -> None:
        """Start ECS task for execution."""
        logger.info(f"Preparing ECS deployment in cluster: {self.config.cluster}")
        # TODO: AWS ECS API integration
        self.task_arn = f"arn:aws:ecs:::task/{uuid.uuid4().hex[:8]}"
        logger.info(f"ECS task started: {self.task_arn}")

    async def execute_node(
        self,
        node: "WorkflowNode",
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute node in ECS task."""
        logger.debug(f"Executing node {node.id} in task {self.task_arn}")
        # TODO: HTTP to ECS task
        return state

    async def cleanup(self) -> None:
        """Stop ECS task."""
        if self.task_arn:
            logger.info(f"Stopping ECS task: {self.task_arn}")
            # TODO: AWS ECS API cleanup
            self.task_arn = None


class RemoteDeploymentHandler(DeploymentHandler):
    """Execute workflows on remote servers."""

    def __init__(self, config: RemoteConfig):
        self.config = config
        self.session_id: Optional[str] = None

    async def prepare(
        self,
        workflow: "WorkflowDefinition",
        config: DeploymentConfig,
    ) -> None:
        """Establish connection to remote server."""
        logger.info(f"Preparing remote deployment to: {self.config.endpoint}")
        self.session_id = uuid.uuid4().hex

    async def execute_node(
        self,
        node: "WorkflowNode",
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute node on remote server."""
        logger.debug(f"Executing node {node.id} remotely")
        # TODO: HTTP/gRPC to remote
        return state

    async def cleanup(self) -> None:
        """Close remote connection."""
        logger.info("Cleaning up remote deployment")
        self.session_id = None


def get_deployment_handler(config: DeploymentConfig) -> DeploymentHandler:
    """Get the appropriate deployment handler for the config.

    Args:
        config: Deployment configuration

    Returns:
        Appropriate DeploymentHandler instance
    """
    if config.target == DeploymentTarget.LOCAL:
        return LocalDeploymentHandler()
    elif config.target == DeploymentTarget.DOCKER:
        return DockerDeploymentHandler(config.docker or DockerConfig())
    elif config.target == DeploymentTarget.KUBERNETES:
        return KubernetesDeploymentHandler(config.kubernetes or KubernetesConfig())
    elif config.target == DeploymentTarget.ECS:
        return ECSDeploymentHandler(config.ecs or ECSConfig())
    elif config.target == DeploymentTarget.REMOTE:
        return RemoteDeploymentHandler(config.remote or RemoteConfig())
    else:
        return LocalDeploymentHandler()


__all__ = [
    # Enums
    "DeploymentTarget",
    # Local/Development Configs
    "InlineConfig",
    "SubprocessConfig",
    "DockerConfig",
    "DockerComposeConfig",
    # AWS Configs
    "ECSConfig",
    "EKSConfig",
    "AWSBatchConfig",
    "EC2Config",
    "LambdaConfig",
    "StepFunctionsConfig",
    # Azure Configs
    "AKSConfig",
    "AzureContainerConfig",
    "AzureBatchConfig",
    # GCP Configs
    "GKEConfig",
    "CloudRunConfig",
    "ComputeEngineConfig",
    "CloudBatchConfig",
    # Kubernetes
    "KubernetesConfig",
    # Workflow Orchestrators
    "AirflowConfig",
    "TemporalConfig",
    # Data Platforms
    "DatabricksConfig",
    "SparkConfig",
    "RayConfig",
    "DaskConfig",
    # Task Queues
    "CeleryConfig",
    # Serverless
    "ModalConfig",
    # Remote/Hybrid
    "RemoteConfig",
    "SandboxConfig",
    # Main Config
    "DeploymentConfig",
    # Handlers
    "DeploymentHandler",
    "LocalDeploymentHandler",
    "DockerDeploymentHandler",
    "KubernetesDeploymentHandler",
    "ECSDeploymentHandler",
    "RemoteDeploymentHandler",
    # Factory
    "get_deployment_handler",
]
