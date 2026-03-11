"""Regression coverage for RAG definition-layer capability requirements."""

from victor_sdk import CapabilityIds

from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
from victor.verticals.contrib.rag.assistant import RAGAssistant


def test_rag_definition_declares_sdk_capability_requirements() -> None:
    """RAG should expose retrieval/document/runtime needs through the SDK contract."""

    definition = RAGAssistant.get_definition()
    requirements = {
        requirement.capability_id: requirement for requirement in definition.capability_requirements
    }

    assert set(requirements) == {
        CapabilityIds.FILE_OPS,
        CapabilityIds.DOCUMENT_INGESTION,
        CapabilityIds.RETRIEVAL,
        CapabilityIds.VECTOR_INDEXING,
        CapabilityIds.WEB_ACCESS,
    }
    assert requirements[CapabilityIds.FILE_OPS].optional is False
    assert requirements[CapabilityIds.DOCUMENT_INGESTION].optional is False
    assert requirements[CapabilityIds.RETRIEVAL].optional is False
    assert requirements[CapabilityIds.VECTOR_INDEXING].optional is False
    assert requirements[CapabilityIds.WEB_ACCESS].optional is True


def test_rag_definition_capability_requirements_round_trip_into_runtime_config() -> None:
    """Runtime binding metadata should preserve the SDK capability requirements."""

    binding = VerticalRuntimeAdapter.build_runtime_binding(RAGAssistant)
    requirements = binding.runtime_config.metadata["capability_requirements"]

    assert [requirement["capability_id"] for requirement in requirements] == [
        CapabilityIds.FILE_OPS,
        CapabilityIds.DOCUMENT_INGESTION,
        CapabilityIds.RETRIEVAL,
        CapabilityIds.VECTOR_INDEXING,
        CapabilityIds.WEB_ACCESS,
    ]
