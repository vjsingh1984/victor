"""Tests for TaskTypeClassifier vertical phrase extension."""

from unittest.mock import MagicMock, patch

from victor.classification import TaskType
from victor.storage.embeddings.task_classifier import TaskTypeClassifier


class TestMergeVerticalPhrases:
    """Test that _merge_vertical_phrases correctly extends phrase lists."""

    def _make_stub(self):
        classifier = TaskTypeClassifier.__new__(TaskTypeClassifier)
        classifier._phrase_lists = {TaskType.EDIT: ["edit a file"]}
        return classifier

    def test_no_enhanced_provider_leaves_phrases_unchanged(self):
        classifier = self._make_stub()
        with patch("victor.core.capability_registry.CapabilityRegistry.get_instance") as mock_reg:
            registry = MagicMock()
            registry.get.return_value = None
            mock_reg.return_value = registry
            classifier._merge_vertical_phrases()
        assert classifier._phrase_lists[TaskType.EDIT] == ["edit a file"]

    def test_enhanced_provider_merges_phrases(self):
        classifier = self._make_stub()
        contributor = MagicMock()
        contributor.get_classifier_phrases.return_value = {
            "edit": ["modify config", "change settings"],
        }
        with patch("victor.core.capability_registry.CapabilityRegistry.get_instance") as mock_reg:
            registry = MagicMock()
            registry.get.return_value = contributor
            registry.is_enhanced.return_value = True
            mock_reg.return_value = registry
            classifier._merge_vertical_phrases()
        assert len(classifier._phrase_lists[TaskType.EDIT]) == 3
        assert "modify config" in classifier._phrase_lists[TaskType.EDIT]

    def test_new_task_type_creates_entry(self):
        classifier = self._make_stub()
        contributor = MagicMock()
        contributor.get_classifier_phrases.return_value = {
            "search": ["find documents"],
        }
        with patch("victor.core.capability_registry.CapabilityRegistry.get_instance") as mock_reg:
            registry = MagicMock()
            registry.get.return_value = contributor
            registry.is_enhanced.return_value = True
            mock_reg.return_value = registry
            classifier._merge_vertical_phrases()
        assert TaskType.SEARCH in classifier._phrase_lists
        assert classifier._phrase_lists[TaskType.SEARCH] == ["find documents"]

    def test_unknown_task_type_is_skipped(self):
        classifier = self._make_stub()
        contributor = MagicMock()
        contributor.get_classifier_phrases.return_value = {
            "nonexistent_type": ["some phrase"],
        }
        with patch("victor.core.capability_registry.CapabilityRegistry.get_instance") as mock_reg:
            registry = MagicMock()
            registry.get.return_value = contributor
            registry.is_enhanced.return_value = True
            mock_reg.return_value = registry
            classifier._merge_vertical_phrases()
        assert classifier._phrase_lists[TaskType.EDIT] == ["edit a file"]
        assert len(classifier._phrase_lists) == 1

    def test_exception_is_caught_gracefully(self):
        classifier = self._make_stub()
        with patch(
            "victor.core.capability_registry.CapabilityRegistry.get_instance",
            side_effect=RuntimeError("registry unavailable"),
        ):
            classifier._merge_vertical_phrases()
        assert classifier._phrase_lists[TaskType.EDIT] == ["edit a file"]

    def test_protocol_exists(self):
        from victor.framework.vertical_protocols import (
            TaskClassifierPhraseProtocol,
        )

        assert hasattr(TaskClassifierPhraseProtocol, "get_classifier_phrases")
