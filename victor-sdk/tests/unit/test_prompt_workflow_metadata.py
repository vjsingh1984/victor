"""Unit tests for SDK prompt and workflow metadata helpers."""

from victor_sdk import (
    PromptMetadata,
    PromptTemplateDefinition,
    TaskTypeHintDefinition,
    WorkflowMetadata,
    normalize_prompt_metadata,
    normalize_prompt_templates,
    normalize_task_type_hints,
    normalize_workflow_metadata,
)


def test_normalize_prompt_templates_supports_dict_shapes() -> None:
    """Prompt template normalization should accept string and dict forms."""

    templates = normalize_prompt_templates(
        {
            "analysis": "Analyze the target carefully.",
            "review": {
                "template": "Review the code for issues.",
                "description": "Code review prompt",
                "metadata": {"style": "strict"},
            },
        }
    )

    assert templates[0].task_type == "analysis"
    assert templates[0].template == "Analyze the target carefully."
    assert templates[1].description == "Code review prompt"
    assert templates[1].metadata["style"] == "strict"


def test_normalize_task_type_hints_supports_existing_vertical_shape() -> None:
    """Task-type hint normalization should accept current dict-based hint payloads."""

    hints = normalize_task_type_hints(
        {
            "vulnerability_scan": {
                "hint": "[SCAN] Run automated scans first.",
                "tool_budget": 25,
                "priority_tools": ["shell", "code_search", "read"],
            }
        }
    )

    assert hints[0].task_type == "vulnerability_scan"
    assert hints[0].tool_budget == 25
    assert hints[0].priority_tools == ["shell", "code_search", "read"]


def test_prompt_and_workflow_metadata_normalize_and_serialize() -> None:
    """PromptMetadata and WorkflowMetadata should normalize from dict payloads."""

    prompt_metadata = normalize_prompt_metadata(
        {
            "templates": {
                "analysis": "Analyze the request."
            },
            "task_type_hints": {
                "analysis": {
                    "hint": "Prefer read-first workflows.",
                    "tool_budget": 12,
                }
            },
        }
    )
    workflow_metadata = normalize_workflow_metadata(
        {
            "initial_stage": "planning",
            "workflow_spec": {"stage_order": ["planning", "execution"]},
            "provider_hints": {"preferred_providers": ["anthropic"]},
            "evaluation_criteria": ["accuracy", "coverage"],
        }
    )

    assert isinstance(prompt_metadata, PromptMetadata)
    assert isinstance(workflow_metadata, WorkflowMetadata)
    assert prompt_metadata.to_dict()["templates"][0]["task_type"] == "analysis"
    assert workflow_metadata.to_dict()["initial_stage"] == "planning"
    assert workflow_metadata.provider_hints["preferred_providers"] == ["anthropic"]
