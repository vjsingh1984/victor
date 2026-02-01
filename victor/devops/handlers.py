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

"""DevOps vertical compute handlers.

Domain-specific handlers for DevOps workflows:
- container_ops: Docker/Podman container operations
- terraform_apply: Infrastructure as Code execution
- mlops: MLOps pipeline operations (MLflow, model registry, deployment)

Usage:
    from victor.devops import handlers
    handlers.register_handlers()

    # In YAML workflow:
    - id: build_image
      type: compute
      handler: container_ops
      inputs:
        operation: build
        dockerfile: Dockerfile
        tag: myapp:latest
      output: build_result

    # MLOps workflow:
    - id: register_model
      type: compute
      handler: mlops
      inputs:
        operation: register
        model_name: my_model
        model_path: $ctx.model_path
        metrics: $ctx.metrics
      output: registry_result
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from victor.framework.workflows.base_handler import BaseHandler
from victor.framework.handler_registry import handler_decorator

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import ComputeNode
    from victor.workflows.executor import WorkflowContext

logger = logging.getLogger(__name__)


@handler_decorator("container_ops", description="Docker/Podman container operations")
@dataclass
class ContainerOpsHandler(BaseHandler):
    """Docker/Podman container operations.

    Manages container lifecycle (build, run, stop, etc.).

    Example YAML:
        - id: build_image
          type: compute
          handler: container_ops
          inputs:
            operation: build
            dockerfile: Dockerfile
            tag: myapp:latest
          output: build_result
    """

    runtime: str = "docker"

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> tuple[Any, int]:
        """Execute container operations."""
        operation = node.input_mapping.get("operation", "build")
        dockerfile = node.input_mapping.get("dockerfile", "Dockerfile")
        tag = node.input_mapping.get("tag", "latest")
        image = node.input_mapping.get("image", "")

        if operation == "build":
            cmd = f"{self.runtime} build -f {dockerfile} -t {tag} ."
        elif operation == "push":
            cmd = f"{self.runtime} push {tag}"
        elif operation == "pull":
            cmd = f"{self.runtime} pull {image}"
        elif operation == "run":
            cmd = f"{self.runtime} run -d {image}"
        elif operation == "stop":
            container_id = node.input_mapping.get("container_id", "")
            cmd = f"{self.runtime} stop {container_id}"
        else:
            raise ValueError(f"Unknown operation: {operation}")

        result = await tool_registry.execute("shell", command=cmd, _exec_ctx={})

        # Raise exception if operation failed
        if not result.success:
            raise Exception(f"Container operation failed: {result.error}")

        output = {
            "operation": operation,
            "success": result.success,
            "output": result.output,
        }

        return output, 1


@handler_decorator("terraform_apply", description="Terraform/OpenTofu IaC operations")
@dataclass
class TerraformHandler(BaseHandler):
    """Terraform/OpenTofu IaC operations.

    Manages infrastructure provisioning.

    Example YAML:
        - id: apply_infra
          type: compute
          handler: terraform_apply
          inputs:
            operation: apply
            workspace: production
            auto_approve: true
          output: terraform_result
    """

    binary: str = "terraform"

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> tuple[Any, int]:
        """Execute Terraform operations."""
        operation = node.input_mapping.get("operation", "plan")
        workspace = node.input_mapping.get("workspace")
        auto_approve = node.input_mapping.get("auto_approve", False)
        tool_calls = 0

        if workspace:
            result = await tool_registry.execute(
                "shell", command=f"{self.binary} workspace select {workspace}", _exec_ctx={}
            )
            tool_calls += 1
            if not result.success:
                raise Exception(f"Workspace selection failed: {result.error}")

        if operation == "init":
            cmd = f"{self.binary} init"
        elif operation == "plan":
            cmd = f"{self.binary} plan -out=tfplan"
        elif operation == "apply":
            cmd = f"{self.binary} apply"
            if auto_approve:
                cmd += " -auto-approve"
            else:
                cmd += " tfplan"
        elif operation == "destroy":
            cmd = f"{self.binary} destroy"
            if auto_approve:
                cmd += " -auto-approve"
        else:
            raise ValueError(f"Unknown operation: {operation}")

        result = await tool_registry.execute("shell", command=cmd, _exec_ctx={})
        tool_calls += 1

        # Raise exception if operation failed
        if not result.success:
            raise Exception(f"Terraform operation failed: {result.error}")

        output = {
            "operation": operation,
            "workspace": workspace,
            "success": result.success,
            "output": result.output,
        }

        return output, tool_calls


@handler_decorator("mlops", description="MLOps pipeline operations with MLflow")
@dataclass
class MLOpsHandler(BaseHandler):
    """MLOps pipeline operations handler.

    Supports model lifecycle management using MLflow:
    - Model registration and versioning
    - Experiment tracking
    - Model deployment (local serving)
    - Model comparison and promotion

    Example YAML:
        - id: register_model
          type: compute
          handler: mlops
          inputs:
            operation: register
            model_name: my_classifier
            model_path: models/classifier.pkl
            metrics:
              accuracy: 0.95
              f1_score: 0.93
          output: registry_result

    Supported operations:
        - register: Register a model with metrics
        - log_experiment: Log experiment run
        - serve: Start model serving endpoint
        - compare: Compare model versions
        - promote: Promote model to production stage
        - list_models: List registered models

    Output varies by operation:
        - register: {model_uri, version, run_id}
        - log_experiment: {run_id, experiment_id}
        - serve: {endpoint_url, status}
        - compare: {comparison_table, best_model}
        - promote: {model_name, stage, version}
    """

    tracking_uri: str = "mlruns"

    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> tuple[Any, int]:
        """Execute MLOps operations."""
        operation = node.input_mapping.get("operation", "register")
        model_name = node.input_mapping.get("model_name", "")
        model_path = node.input_mapping.get("model_path", "")
        metrics_input: Any = node.input_mapping.get("metrics", {})
        params_input: Any = node.input_mapping.get("params", {})
        metrics: dict[str, Any] = metrics_input if isinstance(metrics_input, dict) else {}
        params: dict[str, Any] = params_input if isinstance(params_input, dict) else {}
        experiment_name = node.input_mapping.get("experiment_name", "default")
        stage = node.input_mapping.get("stage", "Staging")
        version = node.input_mapping.get("version", "")
        port = int(node.input_mapping.get("port", 5001))

        result = await self._run_mlops(
            operation=operation,
            model_name=model_name,
            model_path=model_path,
            metrics=metrics,
            params=params,
            experiment_name=experiment_name,
            stage=stage,
            version=version,
            port=port,
        )

        # Raise exception if MLOps operation failed
        if not result.get("success"):
            raise Exception(result.get("error", "MLOps operation failed"))

        return result, 0

    async def _run_mlops(
        self,
        operation: str,
        model_name: str,
        model_path: str,
        metrics: dict[str, Any],
        params: dict[str, Any],
        experiment_name: str,
        stage: str,
        version: str,
        port: int,
    ) -> dict[str, Any]:
        """Run MLOps operation asynchronously."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._run_mlops_sync,
            operation,
            model_name,
            model_path,
            metrics,
            params,
            experiment_name,
            stage,
            version,
            port,
        )

    def _run_mlops_sync(
        self,
        operation: str,
        model_name: str,
        model_path: str,
        metrics: dict[str, Any],
        params: dict[str, Any],
        experiment_name: str,
        stage: str,
        version: str,
        port: int,
    ) -> dict[str, Any]:
        """Synchronous MLOps execution."""
        try:
            # Runtime import for mlflow (optional dependency)
            try:
                import mlflow
                from mlflow.tracking import MlflowClient
            except ImportError as e:
                return {
                    "success": False,
                    "error": f"MLflow not installed. Install with: pip install mlflow. Error: {e}",
                }

            mlflow.set_tracking_uri(self.tracking_uri)
            client = MlflowClient()

            if operation == "register":
                return self._register_model(mlflow, client, model_name, model_path, metrics, params)
            elif operation == "log_experiment":
                return self._log_experiment(mlflow, experiment_name, metrics, params)
            elif operation == "serve":
                return self._serve_model(model_name, version, port)
            elif operation == "compare":
                return self._compare_models(client, model_name)
            elif operation == "promote":
                return self._promote_model(client, model_name, version, stage)
            elif operation == "list_models":
                return self._list_models(client)
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}. Use: register, log_experiment, serve, compare, promote, list_models",
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"MLOps operation failed: {e}",
            }

    def _register_model(
        self,
        mlflow: Any,
        client: Any,
        model_name: str,
        model_path: str,
        metrics: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Register a model with MLflow."""
        with mlflow.start_run() as run:
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log model
            if model_path:
                mlflow.log_artifact(model_path)

            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"
            result = mlflow.register_model(model_uri, model_name)

            return {
                "success": True,
                "operation": "register",
                "model_name": model_name,
                "model_uri": model_uri,
                "version": result.version,
                "run_id": run.info.run_id,
            }

    def _log_experiment(
        self, mlflow: Any, experiment_name: str, metrics: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Log an experiment run."""
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            for key, value in params.items():
                mlflow.log_param(key, value)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            return {
                "success": True,
                "operation": "log_experiment",
                "experiment_name": experiment_name,
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
            }

    def _serve_model(self, model_name: str, version: str, port: int) -> dict[str, Any]:
        """Start model serving (returns command, doesn't actually start)."""
        model_uri = f"models:/{model_name}/{version or 'latest'}"
        serve_cmd = f"mlflow models serve -m {model_uri} -p {port} --no-conda"

        return {
            "success": True,
            "operation": "serve",
            "model_name": model_name,
            "version": version or "latest",
            "port": port,
            "command": serve_cmd,
            "endpoint_url": f"http://localhost:{port}/invocations",
            "note": "Run the command to start serving. Use tool_registry.execute('shell', command=...) to start.",
        }

    def _compare_models(self, client: Any, model_name: str) -> dict[str, Any]:
        """Compare versions of a registered model."""
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            comparison = []

            for v in versions:
                run = client.get_run(v.run_id)
                comparison.append(
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                        "metrics": run.data.metrics,
                        "status": v.status,
                    }
                )

            # Find best version by first metric
            best_version = None
            if comparison and comparison[0].get("metrics"):
                first_metric = list(comparison[0]["metrics"].keys())[0]
                best_version = max(comparison, key=lambda x: x["metrics"].get(first_metric, 0))

            return {
                "success": True,
                "operation": "compare",
                "model_name": model_name,
                "versions": comparison,
                "best_version": best_version,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Model comparison failed: {e}",
            }

    def _promote_model(
        self, client: Any, model_name: str, version: str, stage: str
    ) -> dict[str, Any]:
        """Promote a model version to a stage."""
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )

        return {
            "success": True,
            "operation": "promote",
            "model_name": model_name,
            "version": version,
            "stage": stage,
        }

    def _list_models(self, client: Any) -> dict[str, Any]:
        """List all registered models."""
        models = client.search_registered_models()

        return {
            "success": True,
            "operation": "list_models",
            "models": [
                {
                    "name": m.name,
                    "latest_versions": [
                        {"version": v.version, "stage": v.current_stage} for v in m.latest_versions
                    ],
                }
                for m in models
            ],
            "count": len(models),
        }


__all__ = [
    "ContainerOpsHandler",
    "TerraformHandler",
    "MLOpsHandler",
    "HANDLERS",
    "register_handlers",
]

# Handler registry for tests and programmatic access
HANDLERS: dict[str, type[object]] = {
    "container_ops": ContainerOpsHandler,
    "terraform_apply": TerraformHandler,
    "mlops": MLOpsHandler,
}


def register_handlers() -> None:
    """Register DevOps handlers with the framework.

    This function is called during vertical initialization to register
    all DevOps-specific handlers with the workflow engine.

    Note: In the current architecture, handlers are registered implicitly
    through the HANDLERS dictionary. This function is provided for API
    compatibility and may be used in the future for explicit registration.
    """
    # Handlers are pre-registered via HANDLERS dict
    # This function exists for API compatibility and future extensibility
    pass
