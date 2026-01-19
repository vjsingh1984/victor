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

"""Unit tests for Multi-Modal Processor."""

import pytest
from victor.multimodal import MultiModalProcessor, ProcessingConfig, MediaType, ProcessingStatus


class TestMultiModalProcessor:
    """Test suite for MultiModalProcessor."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        config = ProcessingConfig(
            max_image_size=1024,
            max_audio_duration=300,
            enable_ocr=False,  # Disable for tests
        )
        return MultiModalProcessor(config=config)

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.config is not None
        assert processor.config.max_image_size == 1024

    def test_supported_formats(self, processor):
        """Test getting supported formats."""
        formats = processor.get_supported_formats()

        assert MediaType.IMAGE in formats
        assert MediaType.DOCUMENT in formats
        assert MediaType.AUDIO in formats

        # Check some known formats
        assert ".png" in formats[MediaType.IMAGE]
        assert ".pdf" in formats[MediaType.DOCUMENT]
        assert ".mp3" in formats[MediaType.AUDIO]

    @pytest.mark.asyncio
    async def test_process_image_file_not_found(self, processor):
        """Test image processing with missing file."""
        result = await processor.process_image("/nonexistent/image.png")

        assert result.status == ProcessingStatus.FAILED
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_process_document_file_not_found(self, processor):
        """Test document processing with missing file."""
        result = await processor.process_document("/nonexistent/doc.pdf")

        assert result.status == ProcessingStatus.FAILED
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_process_audio_file_not_found(self, processor):
        """Test audio processing with missing file."""
        result = await processor.process_audio("/nonexistent/audio.mp3")

        assert result.status == ProcessingStatus.FAILED
        assert "not found" in result.error.lower()

    def test_is_supported_image(self, processor):
        """Test format checking for images."""
        assert processor.is_supported("test.png")
        assert processor.is_supported("test.jpg")
        assert processor.is_supported("test.jpeg")

    def test_is_supported_document(self, processor):
        """Test format checking for documents."""
        assert processor.is_supported("test.pdf")
        assert processor.is_supported("test.docx")
        assert processor.is_supported("test.txt")

    def test_is_supported_audio(self, processor):
        """Test format checking for audio."""
        assert processor.is_supported("test.mp3")
        assert processor.is_supported("test.wav")

    def test_is_not_supported(self, processor):
        """Test unsupported format."""
        assert not processor.is_supported("test.xyz")
        assert not processor.is_supported("test.unknown")

    @pytest.mark.asyncio
    async def test_batch_process_empty(self, processor):
        """Test batch processing with empty list."""
        results = await processor.batch_process([])

        assert results == []

    @pytest.mark.asyncio
    async def test_batch_process_with_invalid_paths(self, processor):
        """Test batch processing with invalid paths."""
        items = [
            {"path": "/nonexistent/file.png", "type": "image"},
            {"path": "/nonexistent/file.pdf", "type": "document"},
        ]

        results = await processor.batch_process(items, max_concurrency=2)

        assert len(results) == 2
        assert all(r.status == ProcessingStatus.FAILED for r in results)


class TestProcessingConfig:
    """Test suite for ProcessingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ProcessingConfig()

        assert config.max_image_size == 4096
        assert config.max_audio_duration == 600
        assert config.max_document_size == 50
        assert config.enable_ocr is True
        assert config.enable_transcription is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProcessingConfig(
            max_image_size=2048,
            max_audio_duration=120,
            enable_ocr=False,
        )

        assert config.max_image_size == 2048
        assert config.max_audio_duration == 120
        assert config.enable_ocr is False


class TestProcessingResult:
    """Test suite for ProcessingResult."""

    def test_result_creation(self):
        """Test creating processing result."""
        from victor.multimodal.processor import ProcessingResult

        result = ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            content="Test content",
            metadata={"key": "value"},
            confidence=0.9,
            media_type=MediaType.IMAGE,
            source="test.png",
        )

        assert result.status == ProcessingStatus.SUCCESS
        assert result.content == "Test content"
        assert result.metadata["key"] == "value"
        assert result.confidence == 0.9
        assert result.media_type == MediaType.IMAGE

    def test_result_to_dict(self):
        """Test converting result to dict."""
        from victor.multimodal.processor import ProcessingResult

        result = ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            content="Test",
            media_type=MediaType.IMAGE,
        )

        result_dict = result.to_dict()

        assert result_dict["status"] == "success"
        assert result_dict["content"] == "Test"
        assert result_dict["media_type"] == "image"
