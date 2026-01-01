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
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import ComputeNode
    from victor.workflows.executor import NodeResult, NodeStatus, WorkflowContext

logger = logging.getLogger(__name__)


@dataclass
class ContainerOpsHandler:
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

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

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
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=f"Unknown operation: {operation}",
                duration_seconds=time.time() - start_time,
            )

        try:
            result = await tool_registry.execute("shell", command=cmd)

            output = {
                "operation": operation,
                "success": result.success,
                "output": result.output,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED if result.success else NodeStatus.FAILED,
                output=output,
                duration_seconds=time.time() - start_time,
                tool_calls_used=1,
            )
        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class TerraformHandler:
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

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        operation = node.input_mapping.get("operation", "plan")
        workspace = node.input_mapping.get("workspace")
        auto_approve = node.input_mapping.get("auto_approve", False)
        tool_calls = 0

        if workspace:
            await tool_registry.execute(
                "shell", command=f"{self.binary} workspace select {workspace}"
            )
            tool_calls += 1

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
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=f"Unknown operation: {operation}",
                duration_seconds=time.time() - start_time,
            )

        try:
            result = await tool_registry.execute("shell", command=cmd)
            tool_calls += 1

            output = {
                "operation": operation,
                "workspace": workspace,
                "success": result.success,
                "output": result.output,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED if result.success else NodeStatus.FAILED,
                output=output,
                duration_seconds=time.time() - start_time,
                tool_calls_used=tool_calls,
            )
        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


HANDLERS = {
    "container_ops": ContainerOpsHandler(),
    "terraform_apply": TerraformHandler(),
}


def register_handlers() -> None:
    """Register DevOps handlers with the workflow executor."""
    from victor.workflows.executor import register_compute_handler

    for name, handler in HANDLERS.items():
        register_compute_handler(name, handler)
        logger.debug(f"Registered DevOps handler: {name}")


__all__ = [
    "ContainerOpsHandler",
    "TerraformHandler",
    "HANDLERS",
    "register_handlers",
]
