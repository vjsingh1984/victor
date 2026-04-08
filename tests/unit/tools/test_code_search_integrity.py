# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for LanceDB integrity probe in code_search_tool."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestIndexIntegrityProbe:
    """Tests for persistent index integrity validation."""

    @pytest.mark.asyncio
    async def test_corrupt_index_detected_and_rebuilt(self, tmp_path):
        """Corrupt index triggers full rebuild."""
        # Create fake persistent embeddings directory
        embed_dir = tmp_path / ".victor" / "embeddings"
        embed_dir.mkdir(parents=True)
        (embed_dir / "embeddings.lance").mkdir()

        mock_index = MagicMock()
        mock_index._is_indexed = True
        mock_index.semantic_search = AsyncMock(
            side_effect=Exception("LanceError: file not found")
        )
        mock_index.index_codebase = AsyncMock()

        # The probe should detect corruption and rebuild
        from victor.tools.code_search_tool import _probe_index_integrity

        rebuilt = await _probe_index_integrity(mock_index)
        assert rebuilt is True
        mock_index.index_codebase.assert_called_once()

    @pytest.mark.asyncio
    async def test_healthy_index_not_rebuilt(self, tmp_path):
        """Healthy index passes probe without rebuild."""
        mock_index = MagicMock()
        mock_index._is_indexed = True
        mock_index.semantic_search = AsyncMock(return_value=[])
        mock_index.index_codebase = AsyncMock()

        from victor.tools.code_search_tool import _probe_index_integrity

        rebuilt = await _probe_index_integrity(mock_index)
        assert rebuilt is False
        mock_index.index_codebase.assert_not_called()

    @pytest.mark.asyncio
    async def test_probe_timeout_triggers_rebuild(self, tmp_path):
        """Probe that hangs triggers rebuild after timeout."""

        async def hanging_search(*args, **kwargs):
            await asyncio.sleep(60)

        mock_index = MagicMock()
        mock_index._is_indexed = True
        mock_index.semantic_search = hanging_search
        mock_index.index_codebase = AsyncMock()

        from victor.tools.code_search_tool import _probe_index_integrity

        rebuilt = await _probe_index_integrity(mock_index, timeout=0.1)
        assert rebuilt is True
        mock_index.index_codebase.assert_called_once()
