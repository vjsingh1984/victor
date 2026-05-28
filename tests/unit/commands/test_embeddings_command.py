from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

import victor.ui.commands.embeddings as embeddings_cmd


class TestEmbeddingsCommand:
    def test_rebuild_tool_embeddings_sync_uses_shared_sync_bridge(self) -> None:
        registry = MagicMock()
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(embeddings_cmd, "_build_tool_registry", return_value=registry),
            patch.object(embeddings_cmd, "_rebuild_tool_embeddings_async", mock_async),
            patch.object(embeddings_cmd, "run_sync", return_value=7) as mock_run_sync,
        ):
            result = embeddings_cmd._rebuild_tool_embeddings_sync()

        assert result == 7
        mock_async.assert_called_once_with(registry)
        mock_run_sync.assert_called_once_with(coro)

    def test_rebuild_conversation_embeddings_sync_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(
                embeddings_cmd, "_rebuild_conversation_embeddings_async", mock_async
            ),
            patch.object(embeddings_cmd, "run_sync", return_value=11) as mock_run_sync,
        ):
            result = embeddings_cmd._rebuild_conversation_embeddings_sync()

        assert result == 11
        mock_async.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)

    @pytest.mark.asyncio
    async def test_rebuild_tool_embeddings_async_closes_selector(self) -> None:
        registry = MagicMock()
        registry.list_tools.return_value = ["tool_a", "tool_b"]
        selector = MagicMock()
        selector.initialize_tool_embeddings = AsyncMock()
        selector.close = AsyncMock()

        with patch(
            "victor.tools.semantic_selector.SemanticToolSelector",
            return_value=selector,
        ):
            result = await embeddings_cmd._rebuild_tool_embeddings_async(registry)

        assert result == 2
        selector.initialize_tool_embeddings.assert_awaited_once_with(registry)
        selector.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rebuild_conversation_embeddings_async_closes_store_on_error(
        self,
    ) -> None:
        store = MagicMock()
        store.initialize = AsyncMock(side_effect=RuntimeError("boom"))
        store.rebuild = AsyncMock()
        store.close = AsyncMock()
        embedding_service = MagicMock()

        with (
            patch(
                "victor.storage.embeddings.service.EmbeddingService.get_instance",
                return_value=embedding_service,
            ),
            patch(
                "victor.agent.conversation_embedding_store.ConversationEmbeddingStore",
                return_value=store,
            ),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await embeddings_cmd._rebuild_conversation_embeddings_async()

        store.rebuild.assert_not_awaited()
        store.close.assert_awaited_once()
