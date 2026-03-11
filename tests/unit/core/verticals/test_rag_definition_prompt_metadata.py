"""Regression coverage for RAG prompt metadata definition/runtime parity."""

from victor.verticals.contrib.rag.assistant import RAGAssistant
from victor.verticals.contrib.rag.prompts import RAGPromptContributor


def test_rag_definition_exposes_serializable_prompt_metadata() -> None:
    """RAG should expose prompt metadata through the SDK definition contract."""

    definition = RAGAssistant.get_definition()
    templates = {
        template.task_type: template.template for template in definition.prompt_metadata.templates
    }
    hints = {
        hint.task_type: hint for hint in definition.prompt_metadata.task_type_hints
    }

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
