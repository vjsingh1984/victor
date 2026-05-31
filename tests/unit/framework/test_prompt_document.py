"""Tests for canonical prompt document model and processors."""

from victor.framework.prompt_document import (
    PromptBlock,
    PromptDeduplicationProcessor,
    PromptDocument,
    PromptPriorityTrimProcessor,
)


class TestPromptDocument:
    """Verify the canonical prompt document behavior."""

    def test_render_orders_blocks_by_priority(self):
        document = PromptDocument()
        document.upsert(PromptBlock(name="later", content="Later", priority=40, header=""))
        document.upsert(PromptBlock(name="first", content="First", priority=10, header=""))

        assert document.render() == "First\n\nLater"

    def test_deduplication_processor_removes_duplicate_and_empty_blocks(self):
        document = PromptDocument(
            [
                PromptBlock(name="a", content="Line A\nLine B", priority=10),
                PromptBlock(name="b", content="Line A\nLine B", priority=20),
                PromptBlock(name="blank", content="   ", priority=30),
            ]
        )

        deduped = PromptDeduplicationProcessor().process(document)

        assert deduped.has_block("a")
        assert not deduped.has_block("b")
        assert not deduped.has_block("blank")

    def test_priority_trim_processor_protects_named_blocks(self):
        document = PromptDocument(
            [
                PromptBlock(name="identity", content="ID", priority=10, header=""),
                PromptBlock(name="context_a", content="A" * 100, priority=70, header=""),
                PromptBlock(name="context_b", content="B" * 120, priority=71, header=""),
            ]
        )

        trimmed = PromptPriorityTrimProcessor(
            max_total_chars=150,
            protected_blocks=("identity",),
            min_priority=70,
        ).process(document)

        assert trimmed.has_block("identity")
        assert trimmed.has_block("context_a") != trimmed.has_block("context_b")
