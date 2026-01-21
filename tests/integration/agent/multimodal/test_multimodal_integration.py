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

"""Integration tests for multimodal capabilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from victor.agent.multimodal.audio_agent import AudioAgent, Transcription
from victor.agent.multimodal.vision_agent import ImageAnalysis, VisionAgent
from victor.config.settings import Settings
from tests.factories import MockProviderFactory


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image."""
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="blue")
    image_path = tmp_path / "test_image.png"
    img.save(image_path)
    return str(image_path)


@pytest.fixture
def sample_audio(tmp_path):
    """Create a sample test audio file."""
    audio_path = tmp_path / "test_audio.mp3"
    audio_path.write_bytes(b"fake audio content for testing")
    return str(audio_path)


@pytest.fixture
def sample_document(tmp_path):
    """Create a sample document with embedded image reference."""
    doc_path = tmp_path / "document.md"
    doc_path.write_text("# Test Document\n\nThis document references an image.")
    return str(doc_path)


class TestVisionAgentIntegration:
    """Integration tests for VisionAgent."""

    @pytest.mark.asyncio
    async def test_vision_with_mock_provider(self, sample_image):
        """Test vision agent with mock provider."""
        provider = MockProviderFactory.create_anthropic()
        agent = VisionAgent(provider=provider)

        provider.chat.return_value = Mock(
            content="A blue square image", role="assistant", model="claude-sonnet-4-5"
        )

        analysis = await agent.analyze_image(sample_image)

        assert analysis.description == "A blue square image"
        assert provider.chat.called

    @pytest.mark.asyncio
    async def test_vision_workflow_complete(self, sample_image):
        """Test complete vision workflow with multiple operations."""
        provider = MockProviderFactory.create_openai()
        agent = VisionAgent(provider=provider)

        # Setup mock responses
        provider.chat.side_effect = [
            Mock(content="Image analysis description", role="assistant"),
            Mock(content="Image caption", role="assistant"),
            Mock(content="Detected: blue square", role="assistant"),
            Mock(content="Extracted text", role="assistant"),
        ]

        # Run workflow
        analysis = await agent.analyze_image(sample_image)
        assert analysis.description == "Image analysis description"

        caption = await agent.generate_caption(sample_image)
        assert caption == "Image caption"

        objects = await agent.detect_objects(sample_image)
        assert isinstance(objects, list)

        text = await agent.ocr_extraction(sample_image)
        assert text == "Extracted text"

        # Verify all calls were made
        assert provider.chat.call_count == 4

    @pytest.mark.asyncio
    async def test_vision_different_providers(self, sample_image):
        """Test vision agent with different providers."""
        providers = [
            MockProviderFactory.create_anthropic(),
            MockProviderFactory.create_openai(),
        ]

        for provider in providers:
            provider.chat.return_value = Mock(
                content=f"Analysis from {provider.name}", role="assistant"
            )

            agent = VisionAgent(provider=provider)
            analysis = await agent.analyze_image(sample_image)

            assert provider.name in analysis.description

    @pytest.mark.asyncio
    async def test_vision_error_handling(self, sample_image):
        """Test vision agent error handling."""
        provider = MockProviderFactory.create_anthropic()
        agent = VisionAgent(provider=provider)

        # Test API error
        provider.chat.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await agent.analyze_image(sample_image)

    @pytest.mark.asyncio
    async def test_vision_with_real_image_file(self, sample_image):
        """Test vision agent processes real image file correctly."""
        provider = MockProviderFactory.create_anthropic()
        agent = VisionAgent(provider=provider)

        # Verify file is read and encoded
        provider.chat.return_value = Mock(
            content="Processed real image file", role="assistant"
        )

        analysis = await agent.analyze_image(sample_image)

        assert "real image file" in analysis.description
        assert Path(sample_image).exists()


class TestAudioAgentIntegration:
    """Integration tests for AudioAgent."""

    @pytest.mark.asyncio
    async def test_audio_with_mock_provider(self, sample_audio):
        """Test audio agent with mock provider."""
        provider = MockProviderFactory.create_openai()

        # Mock OpenAI client
        mock_transcription = Mock()
        mock_transcription.text = "Test transcription"
        mock_transcription.language = "en"
        mock_transcription.duration = 10.0
        mock_transcription.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
        provider.client = mock_client

        agent = AudioAgent(provider=provider)

        # Mock chat for summarization
        provider.chat.return_value = Mock(content="Summary of audio", role="assistant")

        transcription = await agent.transcribe_audio(sample_audio)

        assert transcription.text == "Test transcription"
        assert transcription.language == "en"

    @pytest.mark.asyncio
    async def test_audio_workflow_complete(self, sample_audio):
        """Test complete audio workflow with multiple operations."""
        provider = MockProviderFactory.create_openai()

        # Mock transcription response
        mock_transcription = Mock()
        mock_transcription.text = "This is a test audio transcript for integration testing."
        mock_transcription.language = "en"
        mock_transcription.duration = 15.0
        mock_transcription.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
        provider.client = mock_client

        agent = AudioAgent(provider=provider)

        # Mock chat for summarization
        provider.chat.return_value = Mock(content="Audio summary", role="assistant")

        # Run workflow
        transcription = await agent.transcribe_audio(sample_audio)
        assert "test audio" in transcription.text.lower()

        language = await agent.detect_language(sample_audio)
        assert language == "en"

        summary = await agent.summarize_audio(sample_audio)
        assert summary == "Audio summary"

        analysis = await agent.analyze_audio(sample_audio)
        assert analysis.duration >= 0

    @pytest.mark.asyncio
    async def test_audio_different_formats(self, tmp_path):
        """Test audio agent with different formats."""
        provider = MockProviderFactory.create_openai()

        mock_transcription = Mock()
        mock_transcription.text = "Transcription"
        mock_transcription.language = "en"
        mock_transcription.duration = 5.0
        mock_transcription.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
        provider.client = mock_client

        agent = AudioAgent(provider=provider)

        # Test different formats
        formats = [".mp3", ".wav", ".m4a"]

        for fmt in formats:
            audio_file = tmp_path / f"test{fmt}"
            audio_file.write_bytes(b"fake audio")

            transcription = await agent.transcribe_audio(str(audio_file))
            assert transcription.text == "Transcription"

    @pytest.mark.asyncio
    async def test_audio_error_handling(self, sample_audio):
        """Test audio agent error handling."""
        provider = MockProviderFactory.create_openai()
        # Disable local fallback to ensure API errors are propagated
        agent = AudioAgent(provider=provider, use_local_fallback=False)

        # Mock client that raises error
        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(side_effect=Exception("API Error"))
        provider.client = mock_client

        # When provider API fails and fallback is disabled, should raise the original exception
        with pytest.raises(Exception, match="API Error"):
            await agent.transcribe_audio(sample_audio)


class TestMultimodalOrchestratorIntegration:
    """Integration tests for multimodal with orchestrator."""

    @pytest.mark.asyncio
    async def test_vision_with_orchestrator(self, sample_image):
        """Test using vision agent within orchestrator context."""
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.config.settings import Settings

        # Create orchestrator with mock provider
        provider = MockProviderFactory.create_anthropic()
        settings = Settings()  # Create default settings
        orchestrator = AgentOrchestrator(
            settings=settings, provider=provider, model="claude-sonnet-4-5"
        )

        # Create vision agent
        vision_agent = VisionAgent(provider=provider)

        # Mock responses
        provider.chat.return_value = Mock(
            content="Orchestrator-integrated vision analysis", role="assistant"
        )

        # Use vision agent
        analysis = await vision_agent.analyze_image(sample_image)

        assert "orchestrator" in analysis.description.lower()

    @pytest.mark.asyncio
    async def test_audio_with_orchestrator(self, sample_audio):
        """Test using audio agent within orchestrator context."""
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.config.settings import Settings

        provider = MockProviderFactory.create_openai()

        # Mock transcription
        mock_transcription = Mock()
        mock_transcription.text = "Orchestrator audio transcription"
        mock_transcription.language = "en"
        mock_transcription.duration = 10.0
        mock_transcription.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
        provider.client = mock_client

        settings = Settings()
        orchestrator = AgentOrchestrator(settings=settings, provider=provider, model="gpt-4o")
        audio_agent = AudioAgent(provider=provider)

        transcription = await audio_agent.transcribe_audio(sample_audio)

        assert "orchestrator" in transcription.text.lower()

    @pytest.mark.asyncio
    async def test_multimodal_workflow(self, sample_image, sample_audio):
        """Test combined multimodal workflow."""
        provider = MockProviderFactory.create_anthropic()
        vision_agent = VisionAgent(provider=provider)

        provider_openai = MockProviderFactory.create_openai()
        mock_transcription = Mock()
        mock_transcription.text = "Audio content"
        mock_transcription.language = "en"
        mock_transcription.duration = 10.0
        mock_transcription.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
        provider_openai.client = mock_client

        audio_agent = AudioAgent(provider=provider_openai)

        # Mock vision response
        provider.chat.return_value = Mock(content="Image analysis", role="assistant")

        # Process both modalities
        image_result = await vision_agent.analyze_image(sample_image)
        audio_result = await audio_agent.transcribe_audio(sample_audio)

        assert image_result.description == "Image analysis"
        assert audio_result.text == "Audio content"

    @pytest.mark.asyncio
    async def test_multimodal_document_analysis(self, sample_document, sample_image):
        """Test analyzing document with embedded images."""
        provider = MockProviderFactory.create_anthropic()
        vision_agent = VisionAgent(provider=provider)

        provider.chat.return_value = Mock(
            content="Document contains a blue square image", role="assistant"
        )

        # Analyze document image
        analysis = await vision_agent.analyze_image(sample_image)

        assert "blue square" in analysis.description


class TestMultimodalWorkflowIntegration:
    """Integration tests for multimodal in StateGraph workflows."""

    @pytest.mark.asyncio
    async def test_multimodal_in_stategraph(self, sample_image, sample_audio):
        """Test using multimodal agents in StateGraph workflow."""
        from victor.framework import StateGraph

        provider = MockProviderFactory.create_anthropic()
        provider.chat.return_value = Mock(content="Multimodal analysis", role="assistant")

        vision_agent = VisionAgent(provider=provider)

        # Define workflow state
        async def analyze_vision_node(state: dict) -> dict:
            image_path = state.get("image_path")
            if image_path:
                analysis = await vision_agent.analyze_image(image_path)
                return {"vision_analysis": analysis.description}
            return {}

        # Create workflow
        workflow = StateGraph(state_schema=dict)
        workflow.add_node("vision", analyze_vision_node)
        workflow.set_entry_point("vision")
        workflow.set_finish_point("vision")

        compiled = workflow.compile()

        # Execute
        result = await compiled.invoke({"image_path": sample_image})

        # result is a GraphExecutionResult, need to access the state
        assert hasattr(result, "state") or "vision_analysis" in result
        if hasattr(result, "state"):
            assert "vision_analysis" in result.state
            assert result.state["vision_analysis"] == "Multimodal analysis"
        else:
            assert result["vision_analysis"] == "Multimodal analysis"

    @pytest.mark.asyncio
    async def test_multimodal_parallel_workflow(self, sample_image, sample_audio):
        """Test parallel processing of multiple modalities."""
        import asyncio

        provider = MockProviderFactory.create_anthropic()
        provider.chat.return_value = Mock(
            content="Parallel multimodal analysis", role="assistant"
        )

        vision_agent = VisionAgent(provider=provider)

        provider_openai = MockProviderFactory.create_openai()
        mock_transcription = Mock()
        mock_transcription.text = "Parallel audio"
        mock_transcription.language = "en"
        mock_transcription.duration = 5.0
        mock_transcription.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
        provider_openai.client = mock_client

        audio_agent = AudioAgent(provider=provider_openai)

        # Process in parallel
        results = await asyncio.gather(
            vision_agent.analyze_image(sample_image),
            audio_agent.transcribe_audio(sample_audio),
        )

        assert len(results) == 2
        assert results[0].description == "Parallel multimodal analysis"
        assert results[1].text == "Parallel audio"


class TestMultimodalErrorRecovery:
    """Integration tests for error recovery in multimodal operations."""

    @pytest.mark.asyncio
    async def test_vision_retries_on_failure(self, sample_image):
        """Test vision agent retries on transient failures."""
        provider = MockProviderFactory.create_anthropic()
        agent = VisionAgent(provider=provider)

        # First call fails, second succeeds
        provider.chat.side_effect = [
            Exception("Temporary error"),
            Mock(content="Success after retry", role="assistant"),
        ]

        # Should fail on first attempt (no automatic retry in current implementation)
        with pytest.raises(Exception, match="Temporary error"):
            await agent.analyze_image(sample_image)

    @pytest.mark.asyncio
    async def test_audio_fallback_on_provider_unavailable(self, sample_audio):
        """Test audio agent error when provider doesn't support audio."""
        provider = MockProviderFactory.create_anthropic()  # Non-audio provider
        agent = AudioAgent(provider=provider, use_local_fallback=False)

        # Should raise RuntimeError when provider doesn't support audio
        with pytest.raises(RuntimeError, match="Audio transcription not available"):
            await agent.transcribe_audio(sample_audio)


class TestMultimodalPerformance:
    """Integration tests for multimodal performance."""

    @pytest.mark.asyncio
    async def test_vision_large_image(self, tmp_path):
        """Test vision agent with large image."""
        from PIL import Image

        # Create larger image
        img = Image.new("RGB", (1000, 1000), color="red")
        image_path = tmp_path / "large_image.png"
        img.save(image_path)

        provider = MockProviderFactory.create_anthropic()
        agent = VisionAgent(provider=provider)

        provider.chat.return_value = Mock(
            content="Large image analysis complete", role="assistant"
        )

        analysis = await agent.analyze_image(str(image_path))

        assert "large image" in analysis.description.lower()

    @pytest.mark.asyncio
    async def test_audio_multiple_files(self, tmp_path):
        """Test processing multiple audio files."""
        import asyncio

        provider = MockProviderFactory.create_openai()

        mock_transcription = Mock()
        mock_transcription.text = "Transcription"
        mock_transcription.language = "en"
        mock_transcription.duration = 5.0
        mock_transcription.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
        provider.client = mock_client

        agent = AudioAgent(provider=provider)

        # Create multiple audio files
        audio_files = []
        for i in range(3):
            audio_path = tmp_path / f"audio_{i}.mp3"
            audio_path.write_bytes(b"audio content")
            audio_files.append(str(audio_path))

        # Process all files
        results = await asyncio.gather(*[agent.transcribe_audio(f) for f in audio_files])

        assert len(results) == 3
        for result in results:
            assert result.text == "Transcription"


class TestMultimodalRealWorldScenarios:
    """Integration tests for real-world multimodal scenarios."""

    @pytest.mark.asyncio
    async def test_analyze_screenshot(self, tmp_path):
        """Test analyzing a screenshot (common use case)."""
        from PIL import Image, ImageDraw, ImageFont

        # Create a fake screenshot with text
        img = Image.new("RGB", (800, 600), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Screenshot Title", fill="black")

        screenshot_path = tmp_path / "screenshot.png"
        img.save(screenshot_path)

        provider = MockProviderFactory.create_anthropic()
        agent = VisionAgent(provider=provider)

        provider.chat.return_value = Mock(
            content="Screenshot showing 'Screenshot Title'", role="assistant"
        )

        analysis = await agent.analyze_image(str(screenshot_path))

        assert "screenshot" in analysis.description.lower()

    @pytest.mark.asyncio
    async def test_transcribe_meeting_recording(self, tmp_path):
        """Test transcribing a meeting recording."""
        meeting_audio = tmp_path / "meeting.mp3"
        meeting_audio.write_bytes(b"meeting recording content")

        provider = MockProviderFactory.create_openai()

        mock_transcription = Mock()
        mock_transcription.text = "Welcome to the meeting. Today we discuss project updates."
        mock_transcription.language = "en"
        mock_transcription.duration = 300.0  # 5 minutes
        mock_transcription.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
        provider.client = mock_client

        agent = AudioAgent(provider=provider)
        provider.chat.return_value = Mock(content="Meeting summary", role="assistant")

        # Transcribe
        transcription = await agent.transcribe_audio(str(meeting_audio))
        assert "meeting" in transcription.text.lower()

        # Summarize
        summary = await agent.summarize_audio(str(meeting_audio))
        assert "summary" in summary.lower()

    @pytest.mark.asyncio
    async def test_analyze_data_visualization(self, tmp_path):
        """Test extracting data from chart."""
        from PIL import Image, ImageDraw

        # Create a fake bar chart
        img = Image.new("RGB", (400, 300), color="white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([(50, 50), (100, 250)], fill="blue")
        draw.rectangle([(150, 50), (200, 200)], fill="red")

        chart_path = tmp_path / "chart.png"
        img.save(chart_path)

        provider = MockProviderFactory.create_anthropic()
        agent = VisionAgent(provider=provider)

        provider.chat.return_value = Mock(
            content="Bar chart showing two data series", role="assistant"
        )

        plot_data = await agent.extract_data_from_plot(str(chart_path))

        assert plot_data.chart_type.value == "unknown"  # Would need parsing
        assert "bar chart" in plot_data.metadata.get("raw_response", "").lower()
