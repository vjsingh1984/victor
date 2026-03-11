"""Regression coverage for RAG prompt metadata definition/runtime parity."""

from victor_sdk import StageDefinition, TieredToolConfig

from victor.verticals.contrib.rag.assistant import RAGAssistant
from victor.verticals.contrib.rag.prompts import RAGPromptContributor


def test_rag_definition_exposes_serializable_prompt_metadata() -> None:
    """RAG should expose prompt metadata through the SDK definition contract."""

    definition = RAGAssistant.get_definition()
    templates = {
        template.task_type: template.template for template in definition.prompt_metadata.templates
    }
    hints = {hint.task_type: hint for hint in definition.prompt_metadata.task_type_hints}

    assert templates["rag_operations"].startswith("## RAG Operations")
    assert set(hints) == {
        "document_ingestion",
        "knowledge_search",
        "question_answering",
        "knowledge_management",
    }
    assert hints["question_answering"].priority_tools == ["rag_query", "rag_search"]
    assert hints["question_answering"].metadata["description"] == (
        "Answering questions from the knowledge base"
    )


def test_rag_prompt_contributor_wraps_shared_prompt_metadata() -> None:
    """Runtime prompt contributor should derive from the shared metadata payload."""

    contributor = RAGPromptContributor()
    hints = contributor.get_task_type_hints()

    assert contributor.get_system_prompt_section().startswith("## RAG Operations")
    assert "question_answering" in hints
    assert hints["knowledge_management"].priority_tools == [
        "rag_list",
        "rag_delete",
        "rag_stats",
    ]


def test_rag_assistant_uses_sdk_stage_and_tier_contracts() -> None:
    """RAG assistant should use SDK-owned stage and tier config types."""

    stages = RAGAssistant.get_stages()
    tiered = RAGAssistant.get_tiered_tool_config()

    assert isinstance(stages["INITIAL"], StageDefinition)
    assert "rag_query" in stages["QUERYING"].tools
    assert stages["SEARCHING"].next_stages == {"INITIAL", "QUERYING", "SYNTHESIZING"}

    assert isinstance(tiered, TieredToolConfig)
    assert tiered.mandatory == {"read", "ls"}
    assert tiered.vertical_core == {
        "rag_search",
        "rag_query",
        "rag_list",
        "rag_stats",
    }
    assert tiered.get_tools_for_stage("QUERYING") == {
        "read",
        "ls",
        "rag_search",
        "rag_query",
        "rag_list",
        "rag_stats",
    }
