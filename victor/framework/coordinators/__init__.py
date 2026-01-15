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

"""Workflow coordinators for framework-agnostic workflow infrastructure.

This package provides domain-agnostic workflow execution infrastructure that can be
reused across all verticals (Coding, DevOps, RAG, DataAnalysis, Research) and
external projects.

Architecture Layer: FRAMEWORK
-----------------------------
This is the FRAMEWORK LAYER of Victor's two-layer coordinator architecture:
- Application Layer (victor.agent.coordinators): Victor-specific orchestration logic
- Framework Layer (this package): Domain-agnostic workflow infrastructure

These coordinators provide reusable workflow execution primitives:
- How to execute YAML-defined workflows (any vertical)
- How to execute StateGraph/CompiledGraph workflows (any use case)
- How to handle human-in-the-loop approvals (generic)
- How to cache workflow results (generic)

Key Coordinators:
-----------------
YAMLWorkflowCoordinator:
    YAML workflow loading, execution, and streaming
    Features: Two-level caching (definition + node), checkpointing, escape hatches
    Example: Execute "research/deep_research.yaml" with initial state

GraphExecutionCoordinator:
    StateGraph/CompiledGraph execution and streaming
    Features: LSP-compliant result handling, workflow graph compilation
    Example: Execute compiled StateGraph with streaming events

HITLCoordinator:
    Human-in-the-loop workflow integration
    Features: Approval handling, auto-approve/reject modes
    Example: Execute workflow with human approval at critical steps

CacheCoordinator:
    Workflow caching management
    Features: Definition-level and node-level caching, TTL expiration
    Example: Enable 1-hour cache for workflow execution results

Design Principles:
------------------
1. Domain Agnostic: No Victor-specific logic or business rules
2. Vertical Independent: Works with any vertical or external project
3. Protocol-Based: Clean interfaces (IYAMLLoader, IGraphExecutor, IHITLExecutor)
4. Reusable: Can be used standalone without Victor's application layer

SOLID Compliance:
-----------------
- SRP: YAMLWorkflowCoordinator only handles YAML workflows (not chat, tools, config)
- ISP: IGraphExecutor defines only graph execution methods
- DIP: Depends on UnifiedWorkflowCompiler protocol
- OCP: Extend through new workflow sources (JSON, S3, database)

Layer Interaction:
------------------
Framework coordinators are USED BY application layer coordinators:
    ChatCoordinator (app) --> GraphExecutionCoordinator (framework)
    WorkflowCoordinator (app) --> YAMLWorkflowCoordinator (framework)

Framework coordinators have NO dependency on application layer.

Reuse Pattern:
--------------
All verticals reuse framework coordinators:

    CodingVertical --> YAMLWorkflowCoordinator
    DevOpsVertical --> YAMLWorkflowCoordinator
    RAGVertical --> GraphExecutionCoordinator
    ResearchVertical --> YAMLWorkflowCoordinator
    ExternalVertical --> YAMLWorkflowCoordinator

This provides consistency and reduces code duplication across verticals.

Architecture:
-------------
    WorkflowEngine (Facade)
    ├── YAMLWorkflowCoordinator     # execute_yaml(), stream_yaml()
    ├── GraphExecutionCoordinator   # execute_graph(), stream_graph()
    ├── HITLCoordinator             # execute_with_hitl()
    └── CacheCoordinator            # enable_caching(), clear_cache()

Benefits:
---------
- Reusability: Same infrastructure across all verticals
- Consistency: Uniform workflow execution regardless of vertical
- Testability: Framework coordinators tested independently
- Extensibility: External verticals can use framework coordinators
- Performance: Two-level caching improves execution speed

For application layer coordinators, see victor.agent.coordinators.
For detailed architecture documentation, see docs/architecture/coordinator_separation.md
"""

from victor.framework.coordinators.protocols import (
    IWorkflowExecutor,
    IStreamingExecutor,
    IYAMLLoader,
    IGraphExecutor,
    IHITLExecutor,
    ICacheManager,
)
from victor.framework.coordinators.yaml_coordinator import YAMLWorkflowCoordinator
from victor.framework.coordinators.graph_coordinator import GraphExecutionCoordinator
from victor.framework.coordinators.hitl_coordinator import HITLCoordinator
from victor.framework.coordinators.cache_coordinator import CacheCoordinator

__all__ = [
    # Protocols
    "IWorkflowExecutor",
    "IStreamingExecutor",
    "IYAMLLoader",
    "IGraphExecutor",
    "IHITLExecutor",
    "ICacheManager",
    # Coordinators
    "YAMLWorkflowCoordinator",
    "GraphExecutionCoordinator",
    "HITLCoordinator",
    "CacheCoordinator",
]
