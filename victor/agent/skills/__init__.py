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

"""Dynamic skill acquisition and chaining system.

This package provides runtime skill discovery and skill chaining for dynamic
tool composition.

Key Components:
    - SkillDiscoveryEngine: Discover tools and compose skills
    - SkillChainer: Plan and execute skill chains

Usage:
    from victor.agent.skills import SkillDiscoveryEngine, SkillChainer

    # Discover and compose skills
    discovery = SkillDiscoveryEngine(tool_registry=registry)
    tools = await discovery.discover_tools()
    skill = await discovery.compose_skill("analyzer", tools, "Analyzes code")

    # Chain skills together
    chainer = SkillChainer()
    chain = await chainer.plan_chain("Fix bugs", [skill])
    result = await chainer.execute_chain(chain)
"""

from .skill_chaining import (
    ChainExecutionStatus,
    ChainResult,
    ChainStep,
    SkillChain,
    SkillChainer,
    StepResult,
    ValidationResult,
)
from .skill_discovery import (
    AvailableTool,
    MCPTool,
    Skill,
    SkillDiscoveryEngine,
    ToolSignature,
)

__all__ = [
    # Skill discovery
    "SkillDiscoveryEngine",
    "AvailableTool",
    "ToolSignature",
    "MCPTool",
    "Skill",
    # Skill chaining
    "SkillChainer",
    "SkillChain",
    "ChainStep",
    "ChainResult",
    "StepResult",
    "ChainExecutionStatus",
    "ValidationResult",
]
