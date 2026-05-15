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

"""Readable and token-efficient JSON schema for LLM task planning.

This module provides a Pydantic schema for LLM-generated task plans that balances:
- **Readability**: Clear, self-documenting field names for LLM reliability
- **Token Efficiency**: Still ~40% savings vs verbose JSON through list format
- **Type Safety**: Full Pydantic validation
- **LLM-Friendly**: Structure optimized for LLM generation
- **Context-Aware Tool Selection**: Step-based tool filtering for 50-80% token savings

Key Design Decision: Readable keywords over abbreviations
- Better for LLM reliability (clear structure)
- Easier to debug and maintain
- Still achieves token savings through list format
- Self-documenting for developers

Example Readable JSON (120 tokens vs 180 for verbose JSON):
    {"name":"Add auth","complexity":"moderate","desc":"OAuth2 login",
     "steps":[[1,"research","Analyze patterns","overview"],
              [2,"feature","Create module","write,test"]]}

Usage:
    from victor.agent.planning import TaskPlan, generate_compact_plan

    # LLM generates readable compact JSON
    json_str = '{"name":"auth","complexity":"simple","desc":"Fix bug","steps":[[1,"analyze","Find bug","grep"]]}'

    # Validate and expand
    task_plan = TaskPlan.model_validate_json(json_str)
    execution_plan = task_plan.to_execution_plan()

    # Get tools for a specific step
    tools = task_plan.get_contextual_tools(tool_selector, step_index=0)
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, ValidationError

from victor.agent.planning.base import (
    ExecutionPlan,
    PlanStep,
    StepStatus,
    StepType,
)

logger = logging.getLogger(__name__)

# Import unified response parser from framework
try:
    from victor.processing.response_parser import (
        extract_content_from_response,
        extract_json_from_response,
    )

    # Create aliases for backward compatibility
    extract_llm_response_content = extract_content_from_response
    extract_json_from_llm_response = extract_json_from_response
    _UNIFIED_PARSER_AVAILABLE = True
except ImportError:
    logger.warning("Unified response parser not available, using local fallback")
    _UNIFIED_PARSER_AVAILABLE = False

    # Fallback implementations if unified parser not available
    def extract_llm_response_content(
        response: Union[str, Dict[str, Any], Any],
    ) -> Optional[str]:
        """Fallback content extraction."""
        if response is None:
            return None
        if isinstance(response, str):
            return response
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
            return content if isinstance(content, str) else None
        if hasattr(response, "content"):
            content = response.content
            return content if isinstance(content, str) else None
        return None

    def extract_json_from_llm_response(
        response: Union[str, Dict[str, Any], Any],
    ) -> Optional[str]:
        """Fallback JSON extraction."""
        content = extract_llm_response_content(response)
        if not content:
            return None
        try:
            json.loads(content.strip())
            return content.strip()
        except json.JSONDecodeError:
            return None


class TaskComplexity(str, Enum):
    """Task complexity levels for planning."""

    SIMPLE = "simple"  # Auto mode, 2-3 steps, <30 min
    MODERATE = "moderate"  # Plan-mode, 3-5 steps, 30min-2hr
    COMPLEX = "complex"  # Plan-mode, 5-8 steps, >2hr


# ---------------------------------------------------------------------------
# Exec-type inference helpers (module-level, domain-agnostic)
# ---------------------------------------------------------------------------
# These patterns run as a post-processing pass when the LLM generates steps
# without explicit ``exec`` field annotations.  They are intentionally generic —
# no language or domain keywords are embedded here.

_COND_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*route\s*:", re.I),
    re.compile(r"\bdetermine\s+if\b", re.I),
    re.compile(r"\bif\s+this\s+is\b.{0,60}\bor\b", re.I),
    re.compile(r"\bbranch\b.{0,30}\bon\b", re.I),
    re.compile(r"(?:multi|multiple).{0,30}(?:vs|versus|or).{0,30}single", re.I),
    re.compile(r"single.{0,30}(?:vs|versus|or).{0,30}(?:multi|multiple)", re.I),
    re.compile(r"\bchoose\s+between\b", re.I),
    re.compile(r"\bselect\b.{0,30}\bstrategy\b", re.I),
]

_LOOP_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bloop\s+over\b", re.I),
    re.compile(r"\bfor\s+each\b", re.I),
    re.compile(r"\biterate\s+over\b", re.I),
    re.compile(r"\breview\s+each\b", re.I),
    re.compile(r"\bper\s+\w+\s+(?:do|perform|review|analyze)\b", re.I),
]

_APPROVAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bpresent\b.{0,50}\bto\s+(?:the\s+)?user\b", re.I),
    re.compile(r"\bshow\b.{0,50}\bfor\s+(?:user\s+)?(?:review|approval)\b", re.I),
    re.compile(r"\b(?:review|approve|confirm)\b.{0,50}\bbefore\s+(?:begin|continu|proceed)", re.I),
    re.compile(r"\bfor\s+user\s+(?:review|approval|confirmation)\b", re.I),
    re.compile(r"\buser\s+(?:review|approval)\s+before\b", re.I),
]

# Verbs that indicate a step is "producing" a collection for later steps.
# These are matched as substrings of the step description (case-insensitive),
# so prefix variants (e.g. "inventori" catches "inventorying") also match.
_PRODUCER_VERBS = frozenset(
    ["inventory", "inventori", "list", "enumerate", "discover", "identify",
     "collect", "gather", "catalog", "find", "scan", "map", "read all",
     "parse all", "extract all", "detect all", "locate all"]
)


def _infer_exec_type(desc: str) -> Optional[str]:
    """Return inferred execution node type for a plain-dict step, or None."""
    if any(p.search(desc) for p in _COND_PATTERNS):
        return "conditional"
    if any(p.search(desc) for p in _LOOP_PATTERNS):
        return "loop"
    if any(p.search(desc) for p in _APPROVAL_PATTERNS):
        return "approval"
    return None


def _infer_loop_over_key(desc: str) -> Optional[str]:
    """Extract a snake_case plural key from a loop step's description."""
    # Capture the noun phrase after loop/iterate/review each
    candidates = [
        re.compile(
            r"loop\s+over\s+(?:each\s+)?([\w][\w\s]{1,30}?)(?:\s+(?:performing|and\b|,|:|\.|$))",
            re.I,
        ),
        re.compile(
            r"for\s+each\s+([\w][\w\s]{1,30}?)(?:\s+(?:do\b|perform|review|analyze|check|scan|,|\.|$))",
            re.I,
        ),
        re.compile(
            r"iterate\s+over\s+(?:each\s+)?([\w][\w\s]{1,20}?)(?:\s+and|\s*,|\s*$)",
            re.I,
        ),
        re.compile(
            r"review\s+each\s+([\w][\w\s]{1,20}?)(?:\s+and|\s*,|\s*$)",
            re.I,
        ),
    ]
    for pat in candidates:
        m = pat.search(desc)
        if m:
            noun = m.group(1).strip().rstrip(".")
            words = noun.lower().split()[:3]  # at most 3 words
            if not words:
                continue
            key = "_".join(words)
            if not key.endswith("s"):
                key += "s"
            return key
    return None


def _best_matching_key(candidate: str, known_keys: List[str]) -> Optional[str]:
    """Return the known key whose words overlap most with *candidate*."""
    cand_words = {w.rstrip("s") for w in candidate.replace("_", " ").lower().split()}
    best: Optional[str] = None
    best_score = 0
    for key in known_keys:
        key_words = {w.rstrip("s") for w in key.replace("_", " ").lower().split()}
        score = len(cand_words & key_words)
        if score > best_score:
            best_score = score
            best = key
    return best if best_score > 0 else None


def _infer_condition_key(desc: str, known_keys: List[str]) -> Optional[str]:
    """Return a plan_state key that the condition should test, or None."""
    desc_lower = desc.lower()
    for key in known_keys:
        key_readable = key.replace("_", " ").lower()
        if key_readable in desc_lower or key.lower() in desc_lower:
            return key
    # Looser: any significant word from a known key appears in description
    for key in known_keys:
        parts = [p for p in key.replace("_", " ").split() if len(p) > 4]
        if any(p in desc_lower for p in parts):
            return key
    return None


def _infer_branches(step_id: str, all_ids: List[str]) -> Optional[Dict[str, List[str]]]:
    """Derive true/false branch IDs from sibling IDs with 'a'/'b' suffixes."""
    num_match = re.match(r"^(\d+)$", str(step_id))
    if not num_match:
        return None
    base = num_match.group(1)
    for try_base in (base, str(int(base) + 1)):
        sa = next((sid for sid in all_ids if re.fullmatch(rf"{re.escape(try_base)}a", sid, re.I)), None)
        sb = next((sid for sid in all_ids if re.fullmatch(rf"{re.escape(try_base)}b", sid, re.I)), None)
        if sa and sb:
            return {"true": [sa], "false": [sb]}
    return None


def _step_likely_produces(desc: str, key: str) -> bool:
    """Heuristic: does this step's description suggest it produces *key*?"""
    desc_lower = desc.lower()
    key_parts = [p for p in key.replace("_", " ").split() if len(p) > 3]
    has_noun = any(p.rstrip("s") in desc_lower for p in key_parts)
    has_verb = any(v in desc_lower for v in _PRODUCER_VERBS)
    return has_noun and has_verb


class ReadableTaskPlan(BaseModel):
    """Readable and token-efficient task plan schema for LLM generation.

    This schema uses READABLE field names (not cryptic abbreviations) while
    maintaining token efficiency through list-based step representation.

    Token Efficiency Strategy:
    - Readable field names: name, desc, steps (not n, d, s)
    - List format for steps: [id, type, desc, tools, deps]
    - Achieves ~40% token savings while remaining fully readable

    Field Name Mapping:
    - name: task name (short, clear)
    - complexity: simple|moderate|complex (not c)
    - desc: description (not d)
    - steps: list of step data (not s)
    - duration: estimated duration (not e, optional)
    - approval: requires approval (not a, optional)

    Step Data Format:
    - List: [id, type, description, tools, dependencies]
    - Types: research, feature, bugfix, refactor, test, review, deploy, analyze, doc

    Example:
        {
          "name": "Add authentication",
          "complexity": "moderate",
          "desc": "Implement OAuth2 login",
          "steps": [
            [1, "research", "Analyze patterns", "overview"],
            [2, "feature", "Create module", "write,test"]
          ],
          "duration": "30min"
        }
    """

    # Readable field names (not cryptic abbreviations)
    name: str = Field(..., description="Task name (short, clear)")
    complexity: TaskComplexity = Field(..., description="Complexity level")
    desc: str = Field(..., description="Task description")
    steps: List[Union[List, Dict[str, Any]]] = Field(
        ...,
        description=(
            "Steps: [[id, type, desc, tools, deps, exec], ...] or "
            "[{id, type, desc, tools, deps, exec, node, exit}, ...]"
        ),
    )
    duration: Optional[str] = Field(None, description="Estimated duration (e.g., '30min', '2hr')")
    approval: bool = Field(False, description="Requires user approval")

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, v: List[Union[List, Dict[str, Any]]]) -> List[Union[List, Dict[str, Any]]]:
        """Validate step data format — accepts both list tuples and rich dicts."""
        for i, step_data in enumerate(v, 1):
            if isinstance(step_data, dict):
                has_desc = "desc" in step_data or "description" in step_data
                if "id" not in step_data or "type" not in step_data or not has_desc:
                    raise ValueError(
                        f"Step {i}: dict must have 'id', 'type', and 'desc'/'description' keys, "
                        f"got {list(step_data.keys())}"
                    )
            elif isinstance(step_data, list):
                if len(step_data) < 3:
                    raise ValueError(
                        f"Step {i}: must be list with at least [id, type, desc], got {step_data}"
                    )
                if not isinstance(step_data[0], (int, str)):
                    raise ValueError(f"Step {i}: id must be int or str, got {type(step_data[0])}")
            else:
                raise ValueError(f"Step {i}: must be list or dict, got {type(step_data)}")
        return v

    @classmethod
    def _enrich_step_dicts(
        cls, steps: List[Union[List, Dict[str, Any]]]
    ) -> List[Union[List, Dict[str, Any]]]:
        """Infer exec types and data-flow keys when the LLM omits them.

        This is a best-effort inference pass on the raw step list.  Explicitly
        set fields are **never** overwritten.  The inference is generic — no
        domain or language keywords are assumed.

        Pass 1 — exec type: pattern-match description to conditional/loop/approval.
        Pass 2 — produces:  find upstream steps that "produce" a collection that
                            loop/conditional nodes will consume.
        Pass 3 — loop_over / condition_on: resolve to the best matching produces key.
        Pass 4 — branches:  derive true/false step IDs from numeric+alpha sibling IDs.
        """
        result: List[Union[List, Dict[str, Any]]] = [
            dict(s) if isinstance(s, dict) else list(s) for s in steps
        ]
        all_ids = [
            str(s.get("id", "") if isinstance(s, dict) else (s[0] if s else ""))
            for s in result
        ]

        # --- Pass 1: infer exec types ---
        for step in result:
            if not isinstance(step, dict):
                continue
            if step.get("exec") or step.get("execution"):
                continue
            desc = str(step.get("desc", step.get("description", "")))
            inferred = _infer_exec_type(desc)
            if inferred:
                step["exec"] = inferred
                logger.debug(
                    "Step %s: inferred exec=%s from description", step.get("id"), inferred
                )

        # --- Pass 2: infer produces on steps that feed downstream consumers ---
        # First collect keys that consumer steps need (loop_over / condition_on).
        needed_keys: List[str] = []
        for step in result:
            if not isinstance(step, dict):
                continue
            exec_type = str(step.get("exec", "")).lower()
            if exec_type not in ("loop", "conditional"):
                continue
            desc = str(step.get("desc", step.get("description", "")))
            if exec_type == "loop" and not step.get("loop_over"):
                key = _infer_loop_over_key(desc)
                if key and key not in needed_keys:
                    needed_keys.append(key)
            elif exec_type == "conditional" and not step.get("condition_on"):
                # Placeholder — resolved in Pass 3 once produces is known
                pass

        # Back-populate 'produces' on the first upstream step whose description
        # suggests it inventories/lists the needed collection.
        current_produces: List[str] = [
            str(s.get("produces", ""))
            for s in result
            if isinstance(s, dict) and s.get("produces")
        ]
        for key in needed_keys:
            if key in current_produces:
                continue
            for step in result:
                if not isinstance(step, dict) or step.get("produces"):
                    continue
                desc = str(step.get("desc", step.get("description", "")))
                if _step_likely_produces(desc, key):
                    step["produces"] = key
                    current_produces.append(key)
                    logger.debug(
                        "Step %s: inferred produces=%s from description", step.get("id"), key
                    )
                    break

        # --- Pass 3: resolve loop_over and condition_on ---
        for step in result:
            if not isinstance(step, dict):
                continue
            exec_type = str(step.get("exec", "")).lower()
            desc = str(step.get("desc", step.get("description", "")))

            if exec_type == "loop" and not step.get("loop_over"):
                raw_key = _infer_loop_over_key(desc)
                if raw_key:
                    # Prefer an exact match in known produces, fall back to raw
                    aligned = _best_matching_key(raw_key, current_produces) or raw_key
                    step["loop_over"] = aligned
                    logger.debug(
                        "Step %s: inferred loop_over=%s", step.get("id"), aligned
                    )

            elif exec_type == "conditional":
                if not step.get("condition_on"):
                    key = _infer_condition_key(desc, current_produces)
                    if key:
                        step["condition_on"] = key
                        logger.debug(
                            "Step %s: inferred condition_on=%s", step.get("id"), key
                        )
                if not step.get("condition"):
                    # Only default to "multiple" when the description implies a
                    # quantity comparison (multi vs single, more than one, etc.).
                    # Otherwise leave unset so _parse_step_dict uses "non_empty".
                    desc_lower = desc.lower()
                    if re.search(r"\bmulti(?:ple)?\b|\bmore\s+than\s+one\b|\bseveral\b", desc_lower):
                        step["condition"] = "multiple"
                if not step.get("produces"):
                    # Store the boolean result so later steps can read it
                    cond_key = str(step.get("condition_on", ""))
                    if cond_key:
                        step["produces"] = f"is_{cond_key}_multiple"

        # --- Pass 4: infer branches for conditional steps ---
        for step in result:
            if not isinstance(step, dict):
                continue
            if str(step.get("exec", "")).lower() != "conditional":
                continue
            if step.get("branches"):
                continue
            step_id = str(step.get("id", ""))
            branches = _infer_branches(step_id, all_ids)
            if branches:
                step["branches"] = branches
                logger.debug(
                    "Step %s: inferred branches=%s", step_id, branches
                )

        return result

    def to_execution_plan(self) -> ExecutionPlan:
        """Convert readable task plan to full ExecutionPlan."""
        import uuid

        # Enrich steps with inferred exec types and data-flow keys when the LLM
        # did not annotate them (e.g. smaller models that ignore the exec field).
        enriched_steps = self._enrich_step_dicts(self.steps)

        steps = []
        for step_data in enriched_steps:
            plan_step = self._parse_step_data(step_data)
            steps.append(plan_step)

        # Generate a unique plan ID
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # Build full execution plan
        return ExecutionPlan(
            id=plan_id,
            goal=self.desc,
            steps=steps,
            metadata={
                "task_name": self.name,
                "complexity": self.complexity.value,
                "estimated_duration": self.duration,
                "requires_approval": self.approval,
            },
        )

    def _parse_step_data(self, step_data: Union[List, Dict[str, Any]]) -> PlanStep:
        """Parse step data (list tuple or rich dict) into PlanStep."""
        if isinstance(step_data, dict):
            return self._parse_step_dict(step_data)
        return self._parse_step_list(step_data)

    def _parse_step_dict(self, step_data: Dict[str, Any]) -> PlanStep:
        """Parse rich dict step — supports all execution node fields."""
        step_id = str(step_data["id"])
        step_type_str = str(step_data.get("type", "analyze"))
        description = str(step_data.get("desc", step_data.get("description", "")))
        step_type = self._map_step_type(step_type_str)

        tools_raw = step_data.get("tools", "")
        if isinstance(tools_raw, list):
            tools = [str(t).strip() for t in tools_raw if str(t).strip()]
        elif isinstance(tools_raw, str):
            tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
        else:
            tools = []

        deps_raw = step_data.get("deps", step_data.get("depends_on", []))
        dependencies = [str(d) for d in deps_raw] if isinstance(deps_raw, list) else []

        execution = str(step_data.get("exec", step_data.get("execution", ""))).lower()
        node = str(step_data.get("node", ""))
        exit_criteria = list(step_data.get("exit", step_data.get("exit_criteria", [])))
        requires_approval = (
            step_type == StepType.DEPLOYMENT or step_type == StepType.PLANNING or self.approval
        )

        loop_over = str(step_data.get("loop_over", ""))
        produces = str(step_data.get("produces", ""))
        condition_on = str(step_data.get("condition_on", ""))
        condition = str(step_data.get("condition", "non_empty"))
        branches_raw = step_data.get("branches", {})
        branches: Dict[str, Any] = (
            {str(k): [str(v) for v in vals] for k, vals in branches_raw.items()}
            if isinstance(branches_raw, dict)
            else {}
        )

        ctx: Dict[str, Any] = {}
        if tools:
            ctx["tools"] = tools
        if node:
            ctx["node"] = node
        if execution:
            ctx["execution"] = execution
        if loop_over:
            ctx["loop_over"] = loop_over
        if produces:
            ctx["produces"] = produces
        items_raw = step_data.get("items", [])
        if items_raw:
            ctx["items"] = list(items_raw)
        if condition_on:
            ctx["condition_on"] = condition_on
            ctx["condition"] = condition
        if branches:
            ctx["branches"] = branches

        return PlanStep(
            id=step_id,
            description=description,
            step_type=step_type,
            depends_on=dependencies,
            estimated_tool_calls=int(step_data.get("tool_calls", 10)),
            requires_approval=requires_approval,
            sub_agent_role=self._get_sub_agent_role(step_type),
            allowed_tools=tools,
            context=ctx,
            execution=execution,
            exit_criteria=[str(c) for c in exit_criteria],
        )

    def _parse_step_list(self, step_data: List) -> PlanStep:
        """Parse compact list step: [id, type, desc, tools, deps, exec?]."""
        step_id = str(step_data[0])
        step_type_str = step_data[1]
        description = step_data[2]

        # Map readable step type to full StepType
        step_type = self._map_step_type(step_type_str)

        # Parse tools and dependencies. Tool hints are preserved on PlanStep so
        # execution adapters can constrain sub-agents without dropping required tools.
        tools = []
        dependencies = []

        if len(step_data) > 3:
            # Fourth element can be tools (string) or dependencies (list)
            fourth = step_data[3]
            if isinstance(fourth, list):
                dependencies = [str(d) for d in fourth]
            elif isinstance(fourth, str):
                tools = [tool.strip() for tool in fourth.split(",") if tool.strip()]

        if len(step_data) > 4:
            # Fifth element is dependencies if fourth was tools
            deps = step_data[4]
            if isinstance(deps, list):
                dependencies = [str(d) for d in deps]

        # Optional 6th element: explicit execution node type
        execution = ""
        if len(step_data) > 5:
            execution = str(step_data[5]).lower()

        # Check if deployment or high-risk step
        requires_approval = (
            step_type == StepType.DEPLOYMENT or step_type == StepType.PLANNING or self.approval
        )

        ctx: Dict[str, Any] = {}
        if tools:
            ctx["tools"] = tools
        if execution:
            ctx["execution"] = execution

        return PlanStep(
            id=step_id,
            description=description,
            step_type=step_type,
            depends_on=dependencies,
            estimated_tool_calls=10,
            requires_approval=requires_approval,
            sub_agent_role=self._get_sub_agent_role(step_type),
            allowed_tools=tools,
            context=ctx,
            execution=execution,
        )

    def _map_step_type(self, step_type_str: str) -> StepType:
        """Map readable step type to StepType enum."""
        type_map = {
            # Primary readable mappings
            "research": StepType.RESEARCH,
            "planning": StepType.PLANNING,
            "feature": StepType.IMPLEMENTATION,
            "implementation": StepType.IMPLEMENTATION,
            "bugfix": StepType.IMPLEMENTATION,
            "bug": StepType.IMPLEMENTATION,
            "refactor": StepType.IMPLEMENTATION,
            "test": StepType.TESTING,
            "testing": StepType.TESTING,
            "review": StepType.REVIEW,
            "deploy": StepType.DEPLOYMENT,
            "deployment": StepType.DEPLOYMENT,
            "analyze": StepType.RESEARCH,
            "analysis": StepType.RESEARCH,
            "doc": StepType.RESEARCH,
            "documentation": StepType.RESEARCH,
        }
        return type_map.get(
            step_type_str.lower(), StepType.IMPLEMENTATION  # Default to implementation
        )

    def _get_sub_agent_role(self, step_type: StepType) -> Optional[str]:
        """Map step type to sub-agent role."""
        role_map = {
            StepType.RESEARCH: "researcher",
            StepType.PLANNING: "planner",
            StepType.IMPLEMENTATION: "executor",
            StepType.TESTING: "tester",
            StepType.REVIEW: "reviewer",
            StepType.DEPLOYMENT: "deployer",
        }
        return role_map.get(step_type)

    def to_yaml(self) -> str:
        """Convert to YAML format for storage/human editing."""
        import yaml

        plan = self.to_execution_plan()

        # Build YAML structure
        yaml_data = {
            "workflows": {
                self.name: {
                    "description": self.desc,
                    "metadata": {
                        "complexity": self.complexity.value,
                        "estimated_duration": self.duration,
                        "requires_approval": self.approval,
                    },
                    "nodes": [],
                }
            }
        }

        # Convert steps to nodes
        for step in plan.steps:
            node = {
                "id": step.id,
                "type": "agent",
                "role": step.sub_agent_role or "executor",
                "goal": step.description,
                "description": step.description,
                "tool_budget": step.estimated_tool_calls or 10,
            }

            if step.depends_on:
                node["depends_on"] = step.depends_on

            if step.requires_approval:
                node["requires_approval"] = True

            yaml_data["workflows"][self.name]["nodes"].append(node)

        return yaml.safe_dump(yaml_data, sort_keys=False)

    def to_markdown(self) -> str:
        """Convert to markdown for display."""
        plan = self.to_execution_plan()
        lines = [
            f"# {self.name}",
            "",
            f"**Description**: {self.desc}",
            f"**Complexity**: {self.complexity.value}",
            f"**Estimated**: {self.duration or 'Unknown'}",
            f"**Approval**: {'Required' if self.approval else 'Not required'}",
            "",
            "## Steps",
            "",
        ]

        for step in plan.steps:
            status_icon = "⏳"
            lines.append(f"{status_icon} **Step {step.id}**: {step.description}")
            lines.append(f"   - Type: {step.step_type.value}")
            if step.depends_on:
                lines.append(f"   - Depends on: {', '.join(step.depends_on)}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def from_execution_plan(cls, plan: ExecutionPlan) -> "ReadableTaskPlan":
        """Create readable task plan from full ExecutionPlan."""
        steps_data = []

        for step in plan.steps:
            # Map StepType to readable type string
            type_str = cls._step_type_to_readable(step.step_type)

            # Build step data list
            step_list = [int(step.id), type_str, step.description]

            # Add tools if available
            if step.allowed_tools:
                tools_str = ",".join(step.allowed_tools)
                step_list.append(tools_str)

            # Add dependencies
            if step.depends_on:
                if not step.allowed_tools:
                    step_list.append([])  # Placeholder for tools
                step_list.append([int(d) for d in step.depends_on])
            elif step.allowed_tools:
                step_list.append([])  # Empty deps if no dependencies

            steps_data.append(step_list)

        metadata = plan.metadata or {}

        return cls(
            name=metadata.get("task_name", plan.goal[:50]),
            complexity=TaskComplexity(metadata.get("complexity", "moderate")),
            desc=plan.goal,  # ExecutionPlan uses goal, not description
            steps=steps_data,
            duration=metadata.get("estimated_duration"),
            approval=metadata.get("requires_approval", False),
        )

    @staticmethod
    def _step_type_to_readable(step_type: StepType) -> str:
        """Convert StepType enum to readable string."""
        return {
            StepType.RESEARCH: "research",
            StepType.PLANNING: "planning",
            StepType.IMPLEMENTATION: "feature",
            StepType.TESTING: "test",
            StepType.REVIEW: "review",
            StepType.DEPLOYMENT: "deploy",
        }.get(step_type, "feature")

    def get_contextual_tools(
        self,
        tool_selector,
        step_index: int,
        conversation_stage=None,
    ) -> List:
        """Get context-appropriate tools for a specific plan step.

        This method uses StepAwareToolSelector to return only the tools
        that are relevant for this step type, complexity, and stage.
        This provides 50-80% token savings compared to exposing all tools.

        Args:
            tool_selector: ToolSelector instance for tool registry access
            step_index: Index of step in plan (0-based)
            conversation_stage: Optional conversation stage

        Returns:
            List of ToolDefinition objects for this step

        Example:
            from victor.agent.planning import ReadableTaskPlan
            from victor.agent.tool_selection import ToolSelector

            plan = ReadableTaskPlan(...)
            tool_selector = ToolSelector(...)

            # Get tools for first step
            tools = plan.get_contextual_tools(tool_selector, step_index=0)
            print(f"Tools for step 0: {[t.name for t in tools]}")
        """
        from victor.agent.planning.tool_selection import StepAwareToolSelector

        if step_index >= len(self.steps):
            return []

        step_data = self.steps[step_index]
        if isinstance(step_data, dict):
            step_type = str(step_data.get("type", ""))
            step_description = str(step_data.get("desc", step_data.get("description", "")))
        else:
            step_type = step_data[1]  # [id, type, desc, tools, deps]
            step_description = step_data[2]

        # Create step-aware selector
        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
        )

        # Get tools for this step
        return selector.get_tools_for_step(
            step_type=step_type,
            complexity=self.complexity,
            step_description=step_description,
            conversation_stage=conversation_stage,
        )

    def to_json(self, verbose: bool = False) -> str:
        """Convert to JSON string.

        Args:
            verbose: If False, use compact list representation (default)
                    If True, use expanded object representation

        Returns:
            JSON string
        """
        if verbose:
            # Expanded to verbose JSON
            plan = self.to_execution_plan()
            return json.dumps(
                {
                    "name": self.name,
                    "complexity": self.complexity.value,
                    "description": self.desc,
                    "steps": [
                        {
                            "id": step.id,
                            "type": step.step_type.value,
                            "description": step.description,
                            "depends_on": step.depends_on,
                            "estimated_tool_calls": step.estimated_tool_calls,
                            "requires_approval": step.requires_approval,
                        }
                        for step in plan.steps
                    ],
                    "estimated_duration": self.duration,
                    "requires_approval": self.approval,
                },
                indent=2,
            )
        else:
            # Use compact schema (this is what LLM generates)
            return self.model_dump_json(exclude_none=True)

    @classmethod
    def get_llm_prompt(cls) -> str:
        """Get optimized prompt for LLM to generate readable task plans.

        Uses readable keywords for LLM reliability while maintaining
        token efficiency through list-based format.
        """
        return """You are a task planning assistant. Create a task plan in JSON format for the following task.

{
  "name": "short task name",
  "complexity": "simple|moderate|complex",
  "desc": "task description",
  "steps": [
    [step_id, type, description, tools, dependencies, exec]
  ],
  "duration": "estimated time (optional)",
  "approval": false (optional, set true for risky tasks)
}

Step types (use lowercase):
  research, planning, feature, bugfix, refactor, test, review, deploy, analyze, doc

Tools: read, write, grep, git, shell, test, code_search, overview, scaffold

Execution node type (6th element, optional — choose the right shape):
  compute     — deterministic function, NO model call (e.g. build a checklist, format output)
  tool        — deterministic tool calls only, NO model reasoning (e.g. read manifest, ls files)
  agent       — single model-backed worker (default when exec is omitted)
  team        — UnifiedTeamCoordinator formation (use for parallel/hierarchical multi-agent work)
  loop        — iterate over a collection (e.g. workspace members) with exit criteria
  conditional — evaluate a plan-state condition and branch-route downstream steps (no model call)
  approval    — user checkpoint; plan pauses until the user approves before continuing

Rules:
- Use "compute" for steps that produce deterministic structured output (checklists, inventories,
  formatted reports). These NEVER call the model.
- Use "tool" for pure file/grep/shell steps that need no reasoning.
- Use "team" when multiple independent subagents should work in parallel or hierarchy.
- Use "conditional" when the path forward depends on what a prior step discovered
  (e.g. single crate vs multi-crate workspace). Always use the dict format.
- Use "approval" before destructive or expensive steps the user should review first.
- Omit exec (or use "agent") for single model calls.

Format: [id, type, description, "tool1,tool2", [dep_id1, dep_id2], "exec"]

Rich dict format is required for loop, conditional, compute, and approval nodes:
  {"id": N, "type": "...", "desc": "...", "tools": [...], "deps": [...],
   "exec": "loop",
   "loop_over": "workspace_members",   ← key produced by a prior tool/compute step
   "exit": ["all members reviewed"]}

  {"id": N, "type": "...", "desc": "...", "exec": "conditional",
   "condition_on": "workspace_members",   ← plan_state key to test
   "condition": "multiple",               ← "non_empty"|"multiple"|"single"|"empty"
   "produces": "has_multiple_crates",     ← store bool result in plan state
   "branches": {
     "true":  ["5a"],   ← step IDs to run when condition is true
     "false": ["5b"]    ← step IDs to run when condition is false (skip when true)
   }}

  {"id": N, "type": "...", "desc": "...", "exec": "compute",
   "node": "my_checklist_node",        ← name registered by a vertical plugin
   "produces": "checklist_output",     ← key stored in plan state for later steps
   "exit": ["checklist has 12+ items"]}

  {"id": N, "type": "review", "desc": "Approve before running fixes", "exec": "approval"}

State passing — "produces" / "loop_over":
- A tool or compute step with "produces": "KEY" stores its list output in plan state as KEY.
- A loop step with "loop_over": "KEY" iterates over plan_state["KEY"] at runtime.
- Use this to discover workspace members in step 2, then loop over them in step 5.

Examples:
{
  "name": "Fix bug",
  "complexity": "simple",
  "desc": "Fix login bug",
  "steps": [
    [1, "analyze", "Find the bug", "grep", [], "tool"],
    [2, "feature", "Fix the bug", "write"]
  ]
}
{
  "name": "Rust best practices review",
  "complexity": "complex",
  "desc": "Review Rust codebase workspace by workspace",
  "steps": [
    [1, "analyze", "Read root Cargo.toml", "read", [], "tool"],
    {"id": 2, "type": "analyze", "desc": "Inventory workspace members", "tools": ["shell"],
     "deps": [1], "exec": "tool", "produces": "workspace_members",
     "exit": ["list of crate directories returned"]},
    {"id": 3, "type": "doc", "desc": "Create best practices checklist", "tools": [],
     "deps": [2], "exec": "compute", "node": "rust_best_practices_checklist",
     "exit": ["checklist covers Arc, immutability, cloning, concurrency, error handling"]},
    [4, "review", "Present checklist to user for approval", "", [3], "approval"],
    {"id": 5, "type": "analyze", "desc": "Route: multi-crate vs single-crate workspace",
     "deps": [4], "exec": "conditional", "condition_on": "workspace_members",
     "condition": "multiple", "produces": "is_workspace",
     "branches": {
       "true":  ["6a"],   "false": ["6b"]}},
    {"id": "6a", "type": "analyze", "desc": "Review all workspace members via loop",
     "tools": ["read", "grep", "code_search"], "deps": ["5"], "exec": "loop",
     "loop_over": "workspace_members",
     "exit": ["all workspace members reviewed", "each has a findings report"]},
    {"id": "6b", "type": "analyze", "desc": "Review single crate directly",
     "tools": ["read", "grep", "code_search"], "deps": ["5"], "exec": "agent"},
    [7, "doc", "Synthesize findings report", "write", ["6a", "6b"]]
  ],
  "duration": "4-6hr"
}

Please generate the task plan as valid JSON above. Do not include markdown code blocks."""

    @classmethod
    def get_complexity_prompt(cls) -> str:
        """Get prompt for classifying task complexity."""
        return """Classify the task complexity:

Task: {user_request}

Consider:
- SIMPLE: Single file, well-defined scope, <30 minutes, 2-3 steps
- MODERATE: Multiple files, some uncertainty, 30min-2 hours, 3-5 steps
- COMPLEX: Multiple components, high uncertainty, >2 hours, 5-8 steps

Respond with ONLY valid JSON:
{
  "complexity": "simple|moderate|complex",
  "reason": "brief explanation"
}"""


class TaskPlannerContext:
    """Session context manager for task planning.

    Manages task plans within a conversation session, allowing plans
    to be referenced, updated, and executed across multiple turns.
    """

    def __init__(self):
        self.current_plan: Optional[ExecutionPlan] = None
        self.plans_history: List[ExecutionPlan] = []
        self.approved_plans: List[str] = []

    def set_plan(self, plan: ExecutionPlan) -> None:
        """Set the current active plan."""
        self.current_plan = plan
        logger.info(f"Set current plan: {plan.goal}")

    def approve_plan(self) -> None:
        """Mark current plan as approved."""
        if self.current_plan:
            plan_id = id(self.current_plan)
            self.approved_plans.append(str(plan_id))
            logger.info(f"Approved plan: {self.current_plan.goal}")

    def archive_plan(self) -> None:
        """Archive current plan to history."""
        if self.current_plan:
            self.plans_history.append(self.current_plan)
            self.current_plan = None
            logger.info("Archived current plan")

    def get_plan_summary(self) -> Dict[str, Any]:
        """Get summary of all plans in context."""
        return {
            "current_plan": self.current_plan.goal if self.current_plan else None,
            "total_plans": len(self.plans_history) + (1 if self.current_plan else 0),
            "approved_plans": len(self.approved_plans),
            "history": [
                {"goal": plan.goal, "steps": len(plan.steps)}
                for plan in self.plans_history[-5:]  # Last 5 plans
            ],
        }

    def to_context_dict(self) -> Dict[str, Any]:
        """Export context for inclusion in LLM prompts."""
        summary = self.get_plan_summary()

        context = {
            "task_planner": {
                "active": self.current_plan is not None,
                "total_plans": summary["total_plans"],
                "approved_count": summary["approved_plans"],
            }
        }

        if self.current_plan:
            plan_dict = self._plan_to_dict(self.current_plan)
            context["task_planner"]["current_plan"] = plan_dict

        if self.plans_history:
            context["task_planner"]["recent_plans"] = summary["history"]

        return context

    def _plan_to_dict(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Convert plan to dictionary for context."""
        return {
            "goal": plan.goal,
            "steps": [
                {
                    "id": step.id,
                    "description": step.description,
                    "type": step.step_type.value,
                    "status": step.status.value if step.status else "pending",
                }
                for step in plan.steps
            ],
            "metadata": plan.metadata,
        }

    @classmethod
    def from_context_dict(cls, context: Dict[str, Any]) -> "TaskPlannerContext":
        """Restore context from dictionary (e.g., from session storage)."""
        ctx = cls()

        if "task_planner" in context:
            tp_data = context["task_planner"]
            # Reconstruct plans from context if needed
            # This is a simplified version - full restoration would need more logic
            if tp_data.get("current_plan"):
                # Would need to reconstruct ExecutionPlan from dict
                pass

        return ctx


# Helper functions for workflow integration


async def generate_task_plan(
    provider,
    user_request: str,
    complexity: Optional[TaskComplexity] = None,
    model: Optional[str] = None,
    max_retries: int = 2,
    conversation_context: Optional[str] = None,
) -> ReadableTaskPlan:
    """Generate a readable task plan using LLM.

    This function uses the framework's response parsing utilities for robust
    JSON extraction and validation. Includes retry logic for reliability.

    Args:
        provider: LLM provider instance
        user_request: Natural language task description
        complexity: Optional pre-classified complexity level
        model: Optional model identifier (if None, will try to get from provider)
        max_retries: Maximum number of retries for plan generation (default: 2)
        conversation_context: Optional prior conversation summary to ground the plan in
            specific findings rather than generating a generic template.

    Returns:
        Validated ReadableTaskPlan

    Raises:
        ValueError: If model identifier cannot be determined
        ValidationError: If LLM response cannot be validated as ReadableTaskPlan
    """
    from victor.providers.base import Message

    # Get model identifier
    if not model:
        # Try to get from orchestrator if available
        if hasattr(provider, "model") and provider.model:
            model = provider.model
        elif hasattr(provider, "_provider") and hasattr(provider._provider, "model"):
            model = provider._provider.model
        else:
            raise ValueError("Model identifier must be provided or available from provider")

    # Classify complexity if not provided
    task_complexity = complexity
    if not task_complexity:
        complexity_prompt = ReadableTaskPlan.get_complexity_prompt()
        complexity_prompt = complexity_prompt.replace("{user_request}", user_request)

        logger.debug("Classifying task complexity...")
        complexity_response = await provider.chat(
            messages=[Message(role="user", content=complexity_prompt)],
            model=model,
            temperature=0.1,
            max_tokens=200,
        )

        # Extract JSON using framework utilities
        complexity_json = extract_json_from_llm_response(complexity_response)
        if not complexity_json:
            raise ValueError(
                "Failed to extract JSON from complexity classification response. "
                f"Response: {extract_llm_response_content(complexity_response)[:200]}"
            )

        complexity_data = json.loads(complexity_json)
        task_complexity = TaskComplexity(complexity_data["complexity"])
        logger.debug(f"Classified complexity as: {task_complexity.value}")

    # Generate task plan with retries
    plan_prompt = ReadableTaskPlan.get_llm_prompt()
    if conversation_context:
        # Inject prior analysis so the plan is grounded in actual findings, not generic steps.
        plan_prompt = (
            f"{plan_prompt}\n\n"
            f"PRIOR ANALYSIS (ground the plan in these specific findings):\n"
            f"{conversation_context}\n\n"
            f"Task: {user_request}"
        )
    else:
        plan_prompt = f"{plan_prompt}\n\nTask: {user_request}"

    logger.debug(f"Planning prompt length: {len(plan_prompt)} chars")
    logger.debug(f"Planning prompt preview: {plan_prompt[:500]}...")

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"Retry {attempt}/{max_retries} for plan generation")

            # Increase max_tokens for complex models
            # Some models (like qwen2.5-coder) need more tokens to generate structured JSON
            plan_max_tokens = 3000  # Increased from 1500

            plan_response = await provider.chat(
                messages=[Message(role="user", content=plan_prompt)],
                model=model,
                temperature=0.2,  # Lower temp for consistent structure
                max_tokens=plan_max_tokens,
            )

            # Log raw response for debugging
            logger.debug(f"Raw plan response type: {type(plan_response)}")

            # Log response details before extraction
            if isinstance(plan_response, dict):
                logger.debug(f"Response dict keys: {list(plan_response.keys())}")
                for key in plan_response.keys():
                    value = plan_response[key]
                    if isinstance(value, str):
                        logger.debug(f"Response dict[{key}] length: {len(value)}")
                        logger.debug(f"Response dict[{key}] preview: {value[:200]}")
                    else:
                        logger.debug(f"Response dict[{key}] type: {type(value)}")
            elif hasattr(plan_response, "__dict__"):
                logger.debug(f"Response object attributes: {list(plan_response.__dict__.keys())}")

            response_content = extract_llm_response_content(plan_response)
            logger.info(
                f"Plan attempt {attempt + 1}: response length={len(response_content) if response_content else 0}, "
                f"response_preview={response_content[:200] if response_content else 'empty'}..."
            )

            # Extract JSON using framework utilities
            plan_json = extract_json_from_llm_response(plan_response)
            if not plan_json:
                raise ValueError(
                    f"No valid JSON found in plan response. "
                    f"Response: {response_content[:500] if response_content else 'empty'}"
                )

            logger.debug(f"Extracted JSON: {plan_json[:200]}...")

            # Validate and return
            plan = ReadableTaskPlan.model_validate_json(plan_json)
            logger.info(
                f"Generated plan: {plan.name} with {len(plan.steps)} steps, "
                f"complexity={plan.complexity.value}"
            )
            return plan

        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_error = e
            logger.warning(
                f"Plan generation attempt {attempt + 1} failed: {e}. "
                f"Response was: {response_content[:200] if response_content else 'empty'}"
            )
            # Continue to retry

    # All retries exhausted
    raise ValueError(
        f"Failed to generate valid task plan after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    ) from last_error


def plan_to_workflow_yaml(plan: ReadableTaskPlan) -> str:
    """Convert readable task plan to YAML workflow format.

    This converts the task plan into a YAML workflow that can be
    executed by the WorkflowEngine.

    Args:
        plan: ReadableTaskPlan to convert

    Returns:
        YAML workflow string
    """
    return plan.to_yaml()


def plan_to_session_context(
    plan: ReadableTaskPlan,
    session_id: str,
    context_store=None,
) -> Dict[str, Any]:
    """Add plan to session context for persistence.

    This allows the task plan to be referenced in future conversation turns
    and persisted across sessions.

    Args:
        plan: ReadableTaskPlan to add to context
        session_id: Session identifier
        context_store: Optional context storage backend

    Returns:
        Updated context dictionary
    """
    context = {
        "session_id": session_id,
        "task_plan": {
            "name": plan.name,
            "complexity": plan.complexity.value,
            "description": plan.desc,
            "estimated_duration": plan.duration,
            "requires_approval": plan.approval,
            "steps": [
                {
                    "id": step.get("id") if isinstance(step, dict) else step[0],
                    "type": step.get("type") if isinstance(step, dict) else step[1],
                    "description": step.get("desc", step.get("description", "")) if isinstance(step, dict) else (step[2] if len(step) > 2 else ""),
                }
                for step in plan.steps
            ],
        },
        "created_at": plan.model_dump_json(
            include={"name", "complexity", "desc", "duration", "approval"}
        ),
    }

    # Store in context backend if provided
    if context_store:
        context_store.set(session_id, "task_plan", context["task_plan"])

    return context


# Legacy aliases for backward compatibility
CompactTaskPlan = ReadableTaskPlan
CompactStepType = None  # Removed, using readable strings instead
