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

"""Unit tests for VisionAgent."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from victor.agent.multimodal.vision_agent import (
    ChartType,
    DetectedObject,
    ImageAnalysis,
    ImageComparison,
    PlotData,
    VisionAgent,
)
from victor.core.errors import ValidationError
from tests.factories import MockProviderFactory


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    return MockProviderFactory.create_anthropic()


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample image file for testing."""
    import io

    # Create a minimal valid PNG file
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="red")
    image_path = tmp_path / "test_image.png"
    img.save(image_path)

    return str(image_path)


@pytest.fixture
def sample_jpg_path(tmp_path):
    """Create a sample JPG file for testing."""
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="blue")
    image_path = tmp_path / "test_image.jpg"
    img.save(image_path)

    return str(image_path)


@pytest.fixture
def vision_agent(mock_provider):
    """Create VisionAgent instance."""
    return VisionAgent(provider=mock_provider)


class TestVisionAgentInit:
    """Tests for VisionAgent initialization."""

    def test_init_with_provider(self, mock_provider):
        """Test initialization with provider."""
        agent = VisionAgent(provider=mock_provider)
        assert agent.provider == mock_provider
        assert agent.model == "claude-sonnet-4-5-20250114"

    def test_init_with_custom_model(self, mock_provider):
        """Test initialization with custom model."""
        agent = VisionAgent(provider=mock_provider, model="gpt-4o")
        assert agent.model == "gpt-4o"

    def test_init_with_openai_provider(self):
        """Test initialization with OpenAI provider."""
        provider = MockProviderFactory.create_openai()
        agent = VisionAgent(provider=provider)
        assert agent.model == "gpt-4o"

    def test_get_default_model_for_unknown_provider(self):
        """Test default model for unknown provider."""
        unknown_provider = MockProviderFactory.create_with_response("test", name="unknown_provider")
        agent = VisionAgent(provider=unknown_provider)
        assert agent.model == "gpt-4o"

    def test_is_vision_capable_anthropic(self, mock_provider):
        """Test vision capability check for Anthropic."""
        agent = VisionAgent(provider=mock_provider)
        assert agent._is_vision_capable() is True

    def test_is_vision_capable_openai(self):
        """Test vision capability check for OpenAI."""
        provider = MockProviderFactory.create_openai()
        agent = VisionAgent(provider=provider)
        assert agent._is_vision_capable() is True

    def test_is_vision_capable_unknown(self):
        """Test vision capability check for unknown provider."""
        unknown_provider = MockProviderFactory.create_with_response("test", name="unknown")
        agent = VisionAgent(provider=unknown_provider)
        assert agent._is_vision_capable() is False


class TestImageEncoding:
    """Tests for image encoding functionality."""

    def test_encode_image_png(self, vision_agent, sample_image_path):
        """Test encoding PNG image."""
        from victor.agent.multimodal.vision_agent import ImageFormat

        base64_data, media_type = vision_agent._encode_image(sample_image_path)
        assert media_type == ImageFormat.PNG
        assert len(base64_data) > 0
        # Valid base64 should only contain specific characters
        assert all(
            c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
            for c in base64_data
        )

    def test_encode_image_jpg(self, vision_agent, sample_jpg_path):
        """Test encoding JPG image."""
        from victor.agent.multimodal.vision_agent import ImageFormat

        base64_data, media_type = vision_agent._encode_image(sample_jpg_path)
        assert media_type == ImageFormat.JPEG
        assert len(base64_data) > 0

    def test_encode_image_file_not_found(self, vision_agent):
        """Test encoding non-existent file."""
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            vision_agent._encode_image("/nonexistent/path.png")

    def test_encode_image_unsupported_format(self, vision_agent, tmp_path):
        """Test encoding unsupported file format."""
        # Create a text file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not an image")

        with pytest.raises(ValidationError, match="Unsupported image format"):
            vision_agent._encode_image(str(txt_file))

    def test_create_vision_message_anthropic(self, vision_agent, sample_image_path):
        """Test creating vision message for Anthropic."""
        message = vision_agent._create_vision_message(sample_image_path, "Describe this")
        assert message.role == "user"
        assert len(message.content) > 0


class TestAnalyzeImage:
    """Tests for image analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_image_basic(self, vision_agent, sample_image_path, mock_provider):
        """Test basic image analysis."""
        mock_provider.chat.return_value = Mock(
            content="A red image on a white background",
            role="assistant",
            model="claude-sonnet-4-5",
        )

        analysis = await vision_agent.analyze_image(sample_image_path)

        assert isinstance(analysis, ImageAnalysis)
        assert len(analysis.description) > 0
        assert isinstance(analysis.objects_found, list)
        assert isinstance(analysis.colors, list)

    @pytest.mark.asyncio
    async def test_analyze_image_with_custom_query(
        self, vision_agent, sample_image_path, mock_provider
    ):
        """Test image analysis with custom query."""
        mock_provider.chat.return_value = Mock(
            content="The image contains a red square",
            role="assistant",
        )

        analysis = await vision_agent.analyze_image(
            sample_image_path, "What shapes are in this image?"
        )

        assert "red square" in analysis.description

    @pytest.mark.asyncio
    async def test_analyze_image_file_not_found(self, vision_agent):
        """Test analyzing non-existent file."""
        with pytest.raises(FileNotFoundError):
            await vision_agent.analyze_image("/nonexistent/image.png")

    @pytest.mark.asyncio
    async def test_analyze_image_invalid_format(self, vision_agent, tmp_path):
        """Test analyzing invalid file format."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not an image")

        with pytest.raises(ValidationError):
            await vision_agent.analyze_image(str(txt_file))

    @pytest.mark.asyncio
    async def test_analyze_image_api_error(self, vision_agent, sample_image_path, mock_provider):
        """Test handling API errors during analysis."""
        mock_provider.chat.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await vision_agent.analyze_image(sample_image_path)


class TestExtractPlotData:
    """Tests for plot data extraction."""

    @pytest.mark.asyncio
    async def test_extract_plot_data_basic(self, vision_agent, sample_image_path, mock_provider):
        """Test basic plot data extraction."""
        mock_provider.chat.return_value = Mock(
            content="This is a bar chart showing sales data",
            role="assistant",
        )

        plot_data = await vision_agent.extract_data_from_plot(sample_image_path)

        assert isinstance(plot_data, PlotData)
        assert plot_data.chart_type == ChartType.UNKNOWN
        assert isinstance(plot_data.values, list)
        assert isinstance(plot_data.labels, dict)
        assert isinstance(plot_data.trends, list)

    @pytest.mark.asyncio
    async def test_extract_plot_data_file_not_found(self, vision_agent):
        """Test extracting from non-existent file."""
        with pytest.raises(FileNotFoundError):
            await vision_agent.extract_data_from_plot("/nonexistent/chart.png")

    @pytest.mark.asyncio
    async def test_extract_plot_data_api_error(
        self, vision_agent, sample_image_path, mock_provider
    ):
        """Test handling API errors during extraction."""
        mock_provider.chat.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            await vision_agent.extract_data_from_plot(sample_image_path)


class TestDetectObjects:
    """Tests for object detection."""

    @pytest.mark.asyncio
    async def test_detect_objects_basic(self, vision_agent, sample_image_path, mock_provider):
        """Test basic object detection."""
        mock_provider.chat.return_value = Mock(
            content="Detected: red square with 95% confidence",
            role="assistant",
        )

        objects = await vision_agent.detect_objects(sample_image_path)

        assert isinstance(objects, list)

    @pytest.mark.asyncio
    async def test_detect_objects_file_not_found(self, vision_agent):
        """Test detecting objects in non-existent file."""
        with pytest.raises(FileNotFoundError):
            await vision_agent.detect_objects("/nonexistent/image.png")

    @pytest.mark.asyncio
    async def test_detect_objects_api_error(self, vision_agent, sample_image_path, mock_provider):
        """Test handling API errors during detection."""
        mock_provider.chat.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            await vision_agent.detect_objects(sample_image_path)


class TestCompareImages:
    """Tests for image comparison."""

    @pytest.mark.asyncio
    async def test_compare_images_basic(
        self, vision_agent, sample_image_path, sample_jpg_path, mock_provider
    ):
        """Test basic image comparison."""
        mock_provider.chat.return_value = Mock(
            content="The images are different. One is red, one is blue.",
            role="assistant",
        )

        comparison = await vision_agent.compare_images(sample_image_path, sample_jpg_path)

        assert isinstance(comparison, ImageComparison)
        assert isinstance(comparison.similarity_score, float)
        assert isinstance(comparison.differences, list)
        assert isinstance(comparison.similarities, list)
        assert isinstance(comparison.common_elements, list)

    @pytest.mark.asyncio
    async def test_compare_images_first_not_found(self, vision_agent, sample_jpg_path):
        """Test comparing with first image missing."""
        with pytest.raises(FileNotFoundError):
            await vision_agent.compare_images("/nonexistent/image1.png", sample_jpg_path)

    @pytest.mark.asyncio
    async def test_compare_images_second_not_found(self, vision_agent, sample_image_path):
        """Test comparing with second image missing."""
        with pytest.raises(FileNotFoundError):
            await vision_agent.compare_images(sample_image_path, "/nonexistent/image2.png")

    @pytest.mark.asyncio
    async def test_compare_images_api_error(
        self, vision_agent, sample_image_path, sample_jpg_path, mock_provider
    ):
        """Test handling API errors during comparison."""
        mock_provider.chat.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            await vision_agent.compare_images(sample_image_path, sample_jpg_path)


class TestGenerateCaption:
    """Tests for image captioning."""

    @pytest.mark.asyncio
    async def test_generate_caption_descriptive(
        self, vision_agent, sample_image_path, mock_provider
    ):
        """Test generating descriptive caption."""
        mock_provider.chat.return_value = Mock(
            content="A small red square image on a white background",
            role="assistant",
        )

        caption = await vision_agent.generate_caption(sample_image_path, style="descriptive")

        assert isinstance(caption, str)
        assert len(caption) > 0
        assert "red square" in caption

    @pytest.mark.asyncio
    async def test_generate_caption_concise(self, vision_agent, sample_image_path, mock_provider):
        """Test generating concise caption."""
        mock_provider.chat.return_value = Mock(content="Red square", role="assistant")

        caption = await vision_agent.generate_caption(sample_image_path, style="concise")

        assert isinstance(caption, str)
        assert len(caption) > 0

    @pytest.mark.asyncio
    async def test_generate_caption_creative(self, vision_agent, sample_image_path, mock_provider):
        """Test generating creative caption."""
        mock_provider.chat.return_value = Mock(
            content="A vibrant red square dancing on a canvas of white",
            role="assistant",
        )

        caption = await vision_agent.generate_caption(sample_image_path, style="creative")

        assert isinstance(caption, str)
        assert len(caption) > 0

    @pytest.mark.asyncio
    async def test_generate_caption_accessibility(
        self, vision_agent, sample_image_path, mock_provider
    ):
        """Test generating accessibility-focused caption."""
        mock_provider.chat.return_value = Mock(
            content="Image shows a red square, 10x10 pixels, centered on white background",
            role="assistant",
        )

        caption = await vision_agent.generate_caption(sample_image_path, style="accessibility")

        assert isinstance(caption, str)
        assert len(caption) > 0

    @pytest.mark.asyncio
    async def test_generate_caption_file_not_found(self, vision_agent):
        """Test captioning non-existent file."""
        with pytest.raises(FileNotFoundError):
            await vision_agent.generate_caption("/nonexistent/image.png")

    @pytest.mark.asyncio
    async def test_generate_caption_api_error(self, vision_agent, sample_image_path, mock_provider):
        """Test handling API errors during captioning."""
        mock_provider.chat.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            await vision_agent.generate_caption(sample_image_path)


class TestOCRExtractText:
    """Tests for OCR text extraction."""

    @pytest.mark.asyncio
    async def test_ocr_extraction_basic(self, vision_agent, sample_image_path, mock_provider):
        """Test basic OCR text extraction."""
        mock_provider.chat.return_value = Mock(
            content="Sample text extracted from image",
            role="assistant",
        )

        text = await vision_agent.ocr_extraction(sample_image_path)

        assert isinstance(text, str)
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_ocr_extraction_with_language(
        self, vision_agent, sample_image_path, mock_provider
    ):
        """Test OCR with language hint."""
        mock_provider.chat.return_value = Mock(
            content="Texto en español",
            role="assistant",
        )

        text = await vision_agent.ocr_extraction(sample_image_path, language="es")

        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_ocr_extraction_file_not_found(self, vision_agent):
        """Test OCR on non-existent file."""
        with pytest.raises(FileNotFoundError):
            await vision_agent.ocr_extraction("/nonexistent/image.png")

    @pytest.mark.asyncio
    async def test_ocr_extraction_api_error(self, vision_agent, sample_image_path, mock_provider):
        """Test handling API errors during OCR."""
        mock_provider.chat.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            await vision_agent.ocr_extraction(sample_image_path)


class TestDataClasses:
    """Tests for dataclass validation."""

    def test_detected_object_valid(self):
        """Test creating valid DetectedObject."""
        from victor.agent.multimodal.vision_agent import BoundingBox

        bbox = BoundingBox(x=10, y=20, width=100, height=200)
        obj = DetectedObject(
            class_name="person",
            confidence=0.95,
            bbox=bbox,
            attributes={"color": "red"},
        )
        assert obj.class_name == "person"
        assert obj.confidence == 0.95
        assert obj.bbox.x == 10
        assert obj.bbox.y == 20

    def test_detected_object_invalid_confidence(self):
        """Test DetectedObject with invalid confidence."""
        from victor.agent.multimodal.vision_agent import BoundingBox

        bbox = BoundingBox(x=0, y=0, width=10, height=10)
        with pytest.raises(ValidationError, match="Confidence must be between 0 and 1"):
            DetectedObject(class_name="person", confidence=1.5, bbox=bbox)

    def test_detected_object_invalid_bbox(self):
        """Test DetectedObject with invalid bbox."""
        from victor.agent.multimodal.vision_agent import BoundingBox

        # Test BoundingBox validation with wrong list length
        with pytest.raises(ValidationError, match="BoundingBox requires 4 values"):
            BoundingBox.from_list([0, 0, 10])

    def test_image_analysis_creation(self):
        """Test creating ImageAnalysis."""
        from datetime import datetime

        analysis = ImageAnalysis(
            analysis="Test image",
            confidence=0.85,
            objects_found=[],
            text_content="Sample text",
            colors=[],
            faces=[],
        )
        assert analysis.description == "Test image"
        assert analysis.objects_found == []
        assert analysis.text_content == "Sample text"

    def test_plot_data_creation(self):
        """Test creating PlotData."""
        from victor.agent.multimodal.vision_agent import DataPoint

        plot = PlotData(
            chart_type=ChartType.BAR,
            data_series=["Series1"],
            labels={"x": "Time", "y": "Value"},
            values=[DataPoint(x=1, y=2)],
            trends=["Increasing"],
            title="Test Chart",
        )
        assert plot.chart_type == ChartType.BAR
        assert len(plot.values) == 1

    def test_image_comparison_creation(self):
        """Test creating ImageComparison."""
        comp = ImageComparison(
            diff_score=0.15,
            differences=["color"],
            similarities=["shape"],
            common_elements=["square"],
        )
        assert comp.similarity_score == 0.85  # 1 - diff_score
        assert "color" in comp.differences


class TestIntegrationScenarios:
    """Integration-style test scenarios."""

    @pytest.mark.asyncio
    async def test_full_vision_workflow(self, vision_agent, sample_image_path, mock_provider):
        """Test complete vision workflow."""
        # Setup mock responses
        mock_provider.chat.side_effect = [
            Mock(content="Detailed image analysis", role="assistant"),
            Mock(content="A red square", role="assistant"),
            Mock(content="Objects: red square", role="assistant"),
        ]

        # Analyze
        analysis = await vision_agent.analyze_image(sample_image_path)
        assert analysis.description == "Detailed image analysis"

        # Caption
        caption = await vision_agent.generate_caption(sample_image_path)
        assert caption == "A red square"

        # Detect objects
        objects = await vision_agent.detect_objects(sample_image_path)
        assert isinstance(objects, list)

    @pytest.mark.asyncio
    async def test_different_image_formats(
        self, vision_agent, sample_image_path, sample_jpg_path, mock_provider
    ):
        """Test processing different image formats."""
        mock_provider.chat.return_value = Mock(content="Analyzed", role="assistant")

        # PNG
        result_png = await vision_agent.analyze_image(sample_image_path)
        assert result_png.description == "Analyzed"

        # JPG
        result_jpg = await vision_agent.analyze_image(sample_jpg_path)
        assert result_jpg.description == "Analyzed"


class TestBatchProcessImages:
    """Tests for batch image processing."""

    @pytest.mark.asyncio
    async def test_batch_process_analyze(
        self, vision_agent, sample_image_path, sample_jpg_path, mock_provider
    ):
        """Test batch analysis of images."""
        mock_provider.chat.return_value = Mock(content="Analyzed", role="assistant")

        results = await vision_agent.batch_process_images(
            [sample_image_path, sample_jpg_path], operation="analyze"
        )

        assert len(results) == 2
        assert all(isinstance(r, ImageAnalysis) for r in results)

    @pytest.mark.asyncio
    async def test_batch_process_caption(
        self, vision_agent, sample_image_path, sample_jpg_path, mock_provider
    ):
        """Test batch caption generation."""
        mock_provider.chat.return_value = Mock(content="A test image", role="assistant")

        results = await vision_agent.batch_process_images(
            [sample_image_path, sample_jpg_path], operation="caption", style="concise"
        )

        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio
    async def test_batch_process_ocr(
        self, vision_agent, sample_image_path, sample_jpg_path, mock_provider
    ):
        """Test batch OCR extraction."""
        mock_provider.chat.return_value = Mock(content="Extracted text", role="assistant")

        results = await vision_agent.batch_process_images(
            [sample_image_path, sample_jpg_path], operation="ocr"
        )

        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio
    async def test_batch_process_invalid_operation(self, vision_agent, sample_image_path):
        """Test batch processing with invalid operation."""
        with pytest.raises(ValidationError, match="Invalid operation"):
            await vision_agent.batch_process_images([sample_image_path], operation="invalid_op")

    @pytest.mark.asyncio
    async def test_batch_process_handles_errors(
        self, vision_agent, sample_image_path, mock_provider
    ):
        """Test batch processing handles individual failures gracefully."""
        mock_provider.chat.side_effect = Exception("API Error")

        results = await vision_agent.batch_process_images(
            [sample_image_path, "/nonexistent/image.png"], operation="analyze"
        )

        # Should return results with None for failed items
        assert len(results) == 2
        assert results[0] is None  # API error
        assert results[1] is None  # File not error


class TestAnalyzeImageQuality:
    """Tests for image quality analysis."""

    @pytest.mark.asyncio
    async def test_analyze_image_quality_basic(
        self, vision_agent, sample_image_path, mock_provider
    ):
        """Test basic image quality analysis."""
        mock_provider.chat.return_value = Mock(
            content='{"sharpness": 0.8, "brightness": 0.7, "contrast": 0.75, "noise_level": 0.2, "resolution": "high", "overall_quality": 0.75, "issues": [], "strengths": ["good lighting"]}',
            role="assistant",
        )

        quality = await vision_agent.analyze_image_quality(sample_image_path)

        assert "sharpness" in quality
        assert "brightness" in quality
        assert "contrast" in quality
        assert "noise_level" in quality
        assert "resolution" in quality
        assert "overall_quality" in quality
        assert 0 <= quality["sharpness"] <= 1
        assert 0 <= quality["brightness"] <= 1

    @pytest.mark.asyncio
    async def test_analyze_image_quality_file_not_found(self, vision_agent):
        """Test quality analysis of non-existent file."""
        with pytest.raises(FileNotFoundError):
            await vision_agent.analyze_image_quality("/nonexistent/image.png")

    @pytest.mark.asyncio
    async def test_analyze_image_quality_parse_error(
        self, vision_agent, sample_image_path, mock_provider
    ):
        """Test quality analysis with non-JSON response."""
        mock_provider.chat.return_value = Mock(
            content="The image quality is good",
            role="assistant",
        )

        quality = await vision_agent.analyze_image_quality(sample_image_path)

        # Should return default values
        assert "sharpness" in quality
        assert quality["sharpness"] == 0.7


class TestExtractTextFromDocument:
    """Tests for document text extraction."""

    @pytest.mark.asyncio
    async def test_extract_text_with_structure(
        self, vision_agent, sample_image_path, mock_provider
    ):
        """Test document text extraction with structure."""
        mock_response = {
            "text": "Full document text",
            "title": "Test Document",
            "sections": [{"heading": "Introduction", "content": "Content"}],
            "tables": [],
            "fields": {},
            "confidence": 0.95,
        }
        mock_provider.chat.return_value = Mock(
            content=json.dumps(mock_response),
            role="assistant",
        )

        result = await vision_agent.extract_text_from_document(sample_image_path, structure=True)

        assert "text" in result
        assert "structured" in result
        assert "confidence" in result
        assert result["text"] == "Full document text"
        assert result["structured"]["title"] == "Test Document"

    @pytest.mark.asyncio
    async def test_extract_text_without_structure(
        self, vision_agent, sample_image_path, mock_provider
    ):
        """Test document text extraction without structure."""
        mock_provider.chat.return_value = Mock(
            content="Plain text content",
            role="assistant",
        )

        result = await vision_agent.extract_text_from_document(sample_image_path, structure=False)

        assert result["text"] == "Plain text content"
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_extract_text_file_not_found(self, vision_agent):
        """Test text extraction from non-existent file."""
        with pytest.raises(FileNotFoundError):
            await vision_agent.extract_text_from_document("/nonexistent/doc.png")


class TestDetectFacesDetailed:
    """Tests for detailed face detection."""

    @pytest.mark.asyncio
    async def test_detect_faces_with_attributes(
        self, vision_agent, sample_image_path, mock_provider
    ):
        """Test face detection with attributes."""
        mock_response = {
            "faces": [
                {
                    "bbox": [10, 20, 100, 150],
                    "confidence": 0.95,
                    "age_estimation": "adult",
                    "gender": "female",
                    "emotion": "happy",
                    "landmarks": {
                        "left_eye": [40, 60],
                        "right_eye": [80, 60],
                        "nose": [60, 90],
                        "mouth": [60, 120],
                    },
                }
            ],
            "total_faces": 1,
        }
        mock_provider.chat.return_value = Mock(
            content=json.dumps(mock_response),
            role="assistant",
        )

        faces = await vision_agent.detect_faces_detailed(sample_image_path, include_attributes=True)

        assert len(faces) == 1
        assert faces[0].confidence == 0.95
        assert faces[0].attributes["age_estimation"] == "adult"
        assert faces[0].attributes["gender"] == "female"
        assert faces[0].attributes["emotion"] == "happy"
        assert len(faces[0].landmarks) == 4

    @pytest.mark.asyncio
    async def test_detect_faces_without_attributes(
        self, vision_agent, sample_image_path, mock_provider
    ):
        """Test face detection without attributes."""
        mock_response = {
            "faces": [{"bbox": [10, 20, 100, 150], "confidence": 0.95}],
            "total_faces": 1,
        }
        mock_provider.chat.return_value = Mock(
            content=json.dumps(mock_response),
            role="assistant",
        )

        faces = await vision_agent.detect_faces_detailed(
            sample_image_path, include_attributes=False
        )

        assert len(faces) == 1
        assert len(faces[0].attributes) == 0
        assert len(faces[0].landmarks) == 0

    @pytest.mark.asyncio
    async def test_detect_faces_file_not_found(self, vision_agent):
        """Test face detection on non-existent file."""
        with pytest.raises(FileNotFoundError):
            await vision_agent.detect_faces_detailed("/nonexistent/photo.png")

    @pytest.mark.asyncio
    async def test_detect_faces_parse_error(self, vision_agent, sample_image_path, mock_provider):
        """Test face detection with parse error."""
        mock_provider.chat.return_value = Mock(
            content="Could not detect faces",
            role="assistant",
        )

        faces = await vision_agent.detect_faces_detailed(sample_image_path)

        # Should return empty list on parse error
        assert faces == []


class TestExtractColors:
    """Tests for color extraction."""

    @pytest.mark.asyncio
    async def test_extract_colors_basic(self, vision_agent, sample_image_path, mock_provider):
        """Test basic color extraction."""
        mock_response = {
            "colors": [
                {"hex": "#FF0000", "rgb": [255, 0, 0], "percentage": 45.5, "name": "red"},
                {"hex": "#00FF00", "rgb": [0, 255, 0], "percentage": 30.2, "name": "green"},
                {"hex": "#0000FF", "rgb": [0, 0, 255], "percentage": 24.3, "name": "blue"},
            ]
        }
        mock_provider.chat.return_value = Mock(
            content=json.dumps(mock_response),
            role="assistant",
        )

        colors = await vision_agent.extract_colors(sample_image_path, max_colors=10)

        assert len(colors) == 3
        assert colors[0].hex_color == "#FF0000"
        assert colors[0].rgb == (255, 0, 0)
        assert colors[0].percentage == 45.5
        assert colors[0].name == "red"

    @pytest.mark.asyncio
    async def test_extract_colors_limit(self, vision_agent, sample_image_path, mock_provider):
        """Test color extraction with limit."""
        # Return 10 colors but request only 5
        mock_response = {
            "colors": [
                {
                    "hex": f"#FF{i:02x}00",
                    "rgb": [255, i, 0],
                    "percentage": 10.0,
                    "name": f"color{i}",
                }
                for i in range(10)
            ]
        }
        mock_provider.chat.return_value = Mock(
            content=json.dumps(mock_response),
            role="assistant",
        )

        colors = await vision_agent.extract_colors(sample_image_path, max_colors=5)

        # Should return only 5 colors
        assert len(colors) == 5

    @pytest.mark.asyncio
    async def test_extract_colors_file_not_found(self, vision_agent):
        """Test color extraction from non-existent file."""
        with pytest.raises(FileNotFoundError):
            await vision_agent.extract_colors("/nonexistent/image.png")

    @pytest.mark.asyncio
    async def test_extract_colors_parse_error(self, vision_agent, sample_image_path, mock_provider):
        """Test color extraction with parse error."""
        mock_provider.chat.return_value = Mock(
            content="The image has red and blue colors",
            role="assistant",
        )

        colors = await vision_agent.extract_colors(sample_image_path)

        # Should return empty list on parse error
        assert colors == []
