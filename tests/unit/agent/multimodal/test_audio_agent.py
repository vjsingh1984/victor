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

"""Unit tests for AudioAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from victor.agent.multimodal.audio_agent import (
    AudioAnalysis,
    AudioAgent,
    SpeechSegment,
    TranscriptionResult,
)

# Legacy alias for backward compatibility
Transcription = TranscriptionResult
from victor.core.errors import ValidationError
from tests.factories import MockProviderFactory


@pytest.fixture
def mock_openai_provider():
    """Create a mock OpenAI provider."""
    return MockProviderFactory.create_openai()


@pytest.fixture
def mock_anthropic_provider():
    """Create a mock Anthropic provider."""
    return MockProviderFactory.create_anthropic()


@pytest.fixture
def sample_audio_path(tmp_path):
    """Create a sample audio file for testing."""
    audio_path = tmp_path / "test_audio.mp3"
    # Create a minimal file (not a real audio file, but sufficient for testing)
    audio_path.write_bytes(b"fake audio data")
    return str(audio_path)


@pytest.fixture
def sample_wav_path(tmp_path):
    """Create a sample WAV file for testing."""
    wav_path = tmp_path / "test_audio.wav"
    wav_path.write_bytes(b"fake wav data")
    return str(wav_path)


@pytest.fixture
def audio_agent_openai(mock_openai_provider):
    """Create AudioAgent with OpenAI provider."""
    return AudioAgent(provider=mock_openai_provider)


@pytest.fixture
def audio_agent_anthropic(mock_anthropic_provider):
    """Create AudioAgent with Anthropic provider."""
    return AudioAgent(provider=mock_anthropic_provider)


class TestAudioAgentInit:
    """Tests for AudioAgent initialization."""

    def test_init_with_openai_provider(self, mock_openai_provider):
        """Test initialization with OpenAI provider."""
        agent = AudioAgent(provider=mock_openai_provider)
        assert agent.provider == mock_openai_provider
        assert agent.model == "whisper-1"

    def test_init_with_anthropic_provider(self, mock_anthropic_provider):
        """Test initialization with Anthropic provider."""
        agent = AudioAgent(provider=mock_anthropic_provider)
        assert agent.provider == mock_anthropic_provider
        assert agent.model == "claude-sonnet-4-5"

    def test_init_with_custom_model(self, mock_openai_provider):
        """Test initialization with custom model."""
        agent = AudioAgent(provider=mock_openai_provider, model="whisper-large-v3")
        assert agent.model == "whisper-large-v3"

    def test_get_default_model_for_unknown_provider(self):
        """Test default model for unknown provider."""
        unknown_provider = MockProviderFactory.create_with_response("test", name="unknown_provider")
        agent = AudioAgent(provider=unknown_provider)
        assert agent.model == "whisper-1"

    def test_is_audio_capable_openai(self, mock_openai_provider):
        """Test audio capability check for OpenAI."""
        agent = AudioAgent(provider=mock_openai_provider)
        assert agent._is_audio_capable() is True

    def test_is_audio_capable_anthropic(self, mock_anthropic_provider):
        """Test audio capability check for Anthropic."""
        agent = AudioAgent(provider=mock_anthropic_provider)
        assert agent._is_audio_capable() is False


class TestAudioValidation:
    """Tests for audio file validation."""

    def test_validate_audio_file_mp3(self, audio_agent_openai, sample_audio_path):
        """Test validating MP3 file."""
        path = audio_agent_openai._validate_audio_file(sample_audio_path)
        assert path.exists()
        assert path.suffix == ".mp3"

    def test_validate_audio_file_wav(self, audio_agent_openai, sample_wav_path):
        """Test validating WAV file."""
        path = audio_agent_openai._validate_audio_file(sample_wav_path)
        assert path.exists()
        assert path.suffix == ".wav"

    def test_validate_audio_file_not_found(self, audio_agent_openai):
        """Test validating non-existent file."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            audio_agent_openai._validate_audio_file("/nonexistent/audio.mp3")

    def test_validate_audio_file_unsupported_format(self, audio_agent_openai, tmp_path):
        """Test validating unsupported file format."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not an audio file")

        with pytest.raises(ValidationError, match="Unsupported audio format"):
            audio_agent_openai._validate_audio_file(str(txt_file))


class TestTranscribeAudio:
    """Tests for audio transcription."""

    @pytest.mark.asyncio
    async def test_transcribe_audio_openai_basic(
        self, audio_agent_openai, sample_audio_path, mock_openai_provider
    ):
        """Test basic transcription with OpenAI provider."""
        # Mock the OpenAI client's audio transcriptions
        mock_transcription_response = Mock()
        mock_transcription_response.text = "Hello, this is a test transcription."
        mock_transcription_response.language = "en"
        mock_transcription_response.duration = 10.5
        mock_transcription_response.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=mock_transcription_response
        )
        mock_openai_provider.client = mock_client

        transcription = await audio_agent_openai.transcribe_audio(sample_audio_path)

        assert isinstance(transcription, Transcription)
        assert transcription.text == "Hello, this is a test transcription."
        assert transcription.language == "en"
        # Duration comes from response when timestamps are included (default)
        assert transcription.duration == 10.5
        assert transcription.confidence == 0.95

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_language(
        self, audio_agent_openai, sample_audio_path, mock_openai_provider
    ):
        """Test transcription with language specified."""
        mock_transcription_response = Mock()
        mock_transcription_response.text = "Hola, esto es una prueba."
        mock_transcription_response.language = "es"
        mock_transcription_response.duration = 8.0
        mock_transcription_response.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=mock_transcription_response
        )
        mock_openai_provider.client = mock_client

        transcription = await audio_agent_openai.transcribe_audio(sample_audio_path, language="es")

        assert transcription.language == "es"

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_timestamps(self, audio_agent_openai, sample_audio_path):
        """Test transcription with timestamps."""
        from datetime import datetime
        from victor.agent.multimodal.audio_agent import SpeakerSegment, TranscriptionResult

        # Create expected result
        expected_segment = SpeakerSegment(
            speaker_id="SPEAKER_00",
            start_time=0.0,
            end_time=2.5,
            transcript="Hello world",
            confidence=0.98,
        )

        expected_result = TranscriptionResult(
            text="Hello world",
            confidence=0.95,
            timestamp=datetime.now(),
            language="en",
            segments=[expected_segment],
            duration=2.5,
        )

        # Patch the internal method to return our expected result
        with patch.object(
            audio_agent_openai,
            "_transcribe_with_provider",
            return_value=expected_result,
        ):
            transcription = await audio_agent_openai.transcribe_audio(
                sample_audio_path, include_timestamps=True
            )

        assert len(transcription.segments) == 1
        assert transcription.segments[0].start_time == 0.0
        assert transcription.segments[0].end_time == 2.5
        assert transcription.segments[0].transcript == "Hello world"

    @pytest.mark.asyncio
    async def test_transcribe_audio_file_not_found(self, audio_agent_openai):
        """Test transcribing non-existent file."""
        with pytest.raises(FileNotFoundError):
            await audio_agent_openai.transcribe_audio("/nonexistent/audio.mp3")

    @pytest.mark.asyncio
    async def test_transcribe_audio_unsupported_format(self, audio_agent_openai, tmp_path):
        """Test transcribing unsupported format."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not an audio file")

        with pytest.raises(ValidationError):
            await audio_agent_openai.transcribe_audio(str(txt_file))

    @pytest.mark.asyncio
    async def test_transcribe_audio_non_openai_provider(
        self, audio_agent_anthropic, sample_audio_path
    ):
        """Test transcription with non-OpenAI provider."""
        # Non-OpenAI providers should raise RuntimeError when local whisper is not available
        with pytest.raises(RuntimeError, match="Audio transcription not available"):
            await audio_agent_anthropic.transcribe_audio(sample_audio_path)

    @pytest.mark.asyncio
    async def test_transcribe_audio_api_error(self, audio_agent_openai, sample_audio_path):
        """Test handling API errors during transcription."""
        from unittest.mock import patch, AsyncMock

        # Create a mock that raises an error
        mock_transcribe = AsyncMock(side_effect=RuntimeError("API Error"))

        # Patch both _is_audio_capable and _transcribe_with_provider
        with patch.object(audio_agent_openai, "_is_audio_capable", return_value=True):
            with patch.object(audio_agent_openai, "_transcribe_with_provider", mock_transcribe):
                with pytest.raises(RuntimeError):
                    await audio_agent_openai.transcribe_audio(sample_audio_path)


class TestAnalyzeAudio:
    """Tests for audio analysis."""

    @pytest.mark.asyncio
    async def test_analyze_audio_basic(self, audio_agent_openai, sample_audio_path):
        """Test basic audio analysis."""
        analysis = await audio_agent_openai.analyze_audio(sample_audio_path)

        assert isinstance(analysis, AudioAnalysis)
        assert analysis.duration >= 0
        assert analysis.sample_rate > 0
        assert analysis.channels > 0
        assert analysis.format == "mp3"
        # Check quality_metrics fields
        assert 0 <= analysis.quality_metrics.speech_ratio <= 1
        assert 0 <= analysis.quality_metrics.clarity_score <= 1

    @pytest.mark.asyncio
    async def test_analyze_audio_wav(self, audio_agent_openai, sample_wav_path):
        """Test analyzing WAV file."""
        analysis = await audio_agent_openai.analyze_audio(sample_wav_path)

        assert analysis.format == "wav"

    @pytest.mark.asyncio
    async def test_analyze_audio_file_not_found(self, audio_agent_openai):
        """Test analyzing non-existent file."""
        with pytest.raises(FileNotFoundError):
            await audio_agent_openai.analyze_audio("/nonexistent/audio.mp3")

    @pytest.mark.asyncio
    async def test_analyze_audio_unsupported_format(self, audio_agent_openai, tmp_path):
        """Test analyzing unsupported format."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not an audio file")

        with pytest.raises(ValidationError):
            await audio_agent_openai.analyze_audio(str(txt_file))


class TestExtractSpeakerDiarization:
    """Tests for speech segment extraction."""

    @pytest.mark.asyncio
    async def test_extract_speaker_diarization_basic(self, audio_agent_openai, sample_audio_path):
        """Test basic speech segment extraction."""
        from datetime import datetime
        from victor.agent.multimodal.audio_agent import SpeakerSegment, TranscriptionResult

        # Create mock segments
        mock_segment1 = SpeakerSegment(
            speaker_id="SPEAKER_00",
            start_time=0.0,
            end_time=3.0,
            transcript="First segment",
            confidence=0.95,
        )

        mock_segment2 = SpeakerSegment(
            speaker_id="SPEAKER_00",
            start_time=3.5,
            end_time=6.0,
            transcript="Second segment",
            confidence=0.92,
        )

        mock_transcription = TranscriptionResult(
            text="First segment Second segment",
            confidence=0.95,
            timestamp=datetime.now(),
            language="en",
            segments=[mock_segment1, mock_segment2],
            duration=6.0,
        )

        # Patch transcribe_audio to return our mock transcription
        with patch.object(
            audio_agent_openai,
            "transcribe_audio",
            return_value=mock_transcription,
        ):
            segments = await audio_agent_openai.extract_speaker_diarization(sample_audio_path)

        assert len(segments) == 2
        assert segments[0].transcript == "First segment"
        assert segments[1].transcript == "Second segment"

    @pytest.mark.asyncio
    async def test_extract_speaker_diarization_with_min_length(
        self, audio_agent_openai, sample_audio_path
    ):
        """Test speech segment extraction with minimum length filter."""
        from datetime import datetime
        from victor.agent.multimodal.audio_agent import SpeakerSegment, TranscriptionResult

        # Create mock segments with different lengths
        mock_segment1 = SpeakerSegment(
            speaker_id="SPEAKER_00",
            start_time=0.0,
            end_time=0.5,  # Too short
            transcript="Short",
            confidence=0.95,
        )

        mock_segment2 = SpeakerSegment(
            speaker_id="SPEAKER_00",
            start_time=1.0,
            end_time=3.0,  # Long enough
            transcript="Long segment",
            confidence=0.92,
        )

        mock_transcription = TranscriptionResult(
            text="Short Long segment",
            confidence=0.95,
            timestamp=datetime.now(),
            language="en",
            segments=[mock_segment1, mock_segment2],
            duration=3.0,
        )

        # Patch transcribe_audio to return our mock transcription
        with patch.object(
            audio_agent_openai,
            "transcribe_audio",
            return_value=mock_transcription,
        ):
            segments = await audio_agent_openai.extract_speaker_diarization(
                sample_audio_path, min_segment_length=1.0
            )

        # Should only return the long segment
        assert len(segments) == 1
        assert segments[0].transcript == "Long segment"

    @pytest.mark.asyncio
    async def test_extract_speaker_diarization_file_not_found(self, audio_agent_openai):
        """Test extracting segments from non-existent file."""
        with pytest.raises(FileNotFoundError):
            await audio_agent_openai.extract_speaker_diarization("/nonexistent/audio.mp3")


class TestDetectLanguage:
    """Tests for language detection."""

    @pytest.mark.asyncio
    async def test_detect_language_openai(
        self, audio_agent_openai, sample_audio_path, mock_openai_provider
    ):
        """Test language detection with OpenAI provider."""
        mock_transcription_response = Mock()
        mock_transcription_response.language = "en"
        mock_transcription_response.text = "Hello world"
        mock_transcription_response.duration = 2.0
        mock_transcription_response.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=mock_transcription_response
        )
        mock_openai_provider.client = mock_client

        language = await audio_agent_openai.detect_language(sample_audio_path)

        assert language == "en"

    @pytest.mark.asyncio
    async def test_detect_language_file_not_found(self, audio_agent_openai):
        """Test detecting language in non-existent file."""
        with pytest.raises(FileNotFoundError):
            await audio_agent_openai.detect_language("/nonexistent/audio.mp3")

    @pytest.mark.asyncio
    async def test_detect_language_non_openai_provider(
        self, audio_agent_anthropic, sample_audio_path
    ):
        """Test language detection with non-OpenAI provider."""
        language = await audio_agent_anthropic.detect_language(sample_audio_path)

        # Should return "unknown" for non-OpenAI providers
        assert language == "unknown"


class TestSummarizeAudio:
    """Tests for audio summarization."""

    @pytest.mark.asyncio
    async def test_summarize_audio_basic(
        self, audio_agent_openai, sample_audio_path, mock_openai_provider
    ):
        """Test basic audio summarization."""
        # Mock transcription
        mock_transcription_response = Mock()
        mock_transcription_response.text = (
            "This is a long audio transcript with lots of content that needs to be summarized."
        )
        mock_transcription_response.language = "en"
        mock_transcription_response.duration = 30.0
        mock_transcription_response.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=mock_transcription_response
        )
        mock_openai_provider.client = mock_client

        # Mock chat for summarization
        mock_openai_provider.chat.return_value = Mock(
            content="Summary: Long audio transcript summarized.",
            role="assistant",
        )

        summary = await audio_agent_openai.summarize_audio(sample_audio_path)

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Summary" in summary

    @pytest.mark.asyncio
    async def test_summarize_audio_with_custom_max_length(
        self, audio_agent_openai, sample_audio_path, mock_openai_provider
    ):
        """Test audio summarization with custom max length."""
        mock_transcription_response = Mock()
        mock_transcription_response.text = "Long transcript content here."
        mock_transcription_response.language = "en"
        mock_transcription_response.duration = 30.0
        mock_transcription_response.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=mock_transcription_response
        )
        mock_openai_provider.client = mock_client

        mock_openai_provider.chat.return_value = Mock(
            content="Brief summary.",
            role="assistant",
        )

        summary = await audio_agent_openai.summarize_audio(sample_audio_path, max_words=100)

        assert isinstance(summary, str)
        assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_summarize_audio_file_not_found(self, audio_agent_openai):
        """Test summarizing non-existent file."""
        with pytest.raises(FileNotFoundError):
            await audio_agent_openai.summarize_audio("/nonexistent/audio.mp3")

    @pytest.mark.asyncio
    async def test_summarize_audio_api_error(
        self, audio_agent_openai, sample_audio_path, mock_openai_provider
    ):
        """Test handling API errors during summarization."""
        # Transcription succeeds
        mock_transcription_response = Mock()
        mock_transcription_response.text = "Some transcript"
        mock_transcription_response.language = "en"
        mock_transcription_response.duration = 10.0
        mock_transcription_response.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=mock_transcription_response
        )
        mock_openai_provider.client = mock_client

        # But summarization fails
        mock_openai_provider.chat.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await audio_agent_openai.summarize_audio(sample_audio_path)


class TestDataClasses:
    """Tests for dataclass validation."""

    def test_speech_segment_valid(self):
        """Test creating valid SpeechSegment."""

        segment = SpeechSegment(
            speaker_id="SPEAKER_00",
            start_time=0.0,
            end_time=5.0,
            transcript="Hello world",
            confidence=0.95,
        )
        assert segment.start_time == 0.0
        assert segment.end_time == 5.0
        assert segment.duration == 5.0
        assert segment.speaker_id == "SPEAKER_00"
        assert segment.transcript == "Hello world"

    def test_speech_segment_invalid_start_time(self):
        """Test SpeechSegment with negative start time."""
        with pytest.raises(ValidationError, match="Start time must be non-negative"):
            SpeechSegment(
                speaker_id="SPEAKER_00",
                start_time=-1.0,
                end_time=5.0,
                transcript="test",
                confidence=0.9,
            )

    def test_speech_segment_invalid_end_time(self):
        """Test SpeechSegment with end before start."""
        with pytest.raises(ValidationError, match="End time.*must be greater than start time"):
            SpeechSegment(
                speaker_id="SPEAKER_00",
                start_time=5.0,
                end_time=3.0,
                transcript="test",
                confidence=0.9,
            )

    def test_speech_segment_invalid_confidence(self):
        """Test SpeechSegment with invalid confidence."""
        with pytest.raises(ValidationError, match="Confidence must be between 0 and 1"):
            SpeechSegment(
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=5.0,
                transcript="test",
                confidence=1.5,
            )

    def test_transcription_valid(self):
        """Test creating valid Transcription."""
        from datetime import datetime

        transcription = Transcription(
            text="Full transcript",
            confidence=0.92,
            timestamp=datetime.now(),
            segments=[],
            language="en",
            duration=60.0,
        )
        assert transcription.text == "Full transcript"
        assert transcription.confidence == 0.92
        assert transcription.language == "en"
        assert transcription.duration == 60.0

    def test_transcription_invalid_confidence(self):
        """Test Transcription with invalid confidence."""
        from datetime import datetime

        with pytest.raises(ValidationError, match="Confidence must be between 0 and 1"):
            Transcription(
                text="test",
                confidence=1.5,
                timestamp=datetime.now(),
                segments=[],
                language="en",
                duration=10.0,
            )

    def test_transcription_invalid_duration(self):
        """Test Transcription with negative duration."""
        from datetime import datetime

        with pytest.raises(ValidationError, match="Duration must be non-negative"):
            Transcription(
                text="test",
                confidence=0.9,
                timestamp=datetime.now(),
                segments=[],
                language="en",
                duration=-1.0,
            )

    def test_audio_analysis_valid(self):
        """Test creating valid AudioAnalysis."""
        from victor.agent.multimodal.audio_agent import AudioQualityMetrics

        analysis = AudioAnalysis(
            duration=120.0,
            sample_rate=44100,
            channels=2,
            format="mp3",
            quality_metrics=AudioQualityMetrics(
                clarity_score=0.92,
                speech_ratio=0.85,
            ),
        )
        assert analysis.duration == 120.0
        assert analysis.sample_rate == 44100
        assert analysis.channels == 2
        assert analysis.format == "mp3"

    def test_audio_analysis_invalid_duration(self):
        """Test AudioAnalysis with negative duration."""
        from victor.agent.multimodal.audio_agent import AudioQualityMetrics

        with pytest.raises(ValidationError, match="Duration must be non-negative"):
            AudioAnalysis(
                duration=-1.0,
                sample_rate=44100,
                channels=2,
                format="mp3",
                quality_metrics=AudioQualityMetrics(
                    clarity_score=0.9,
                    speech_ratio=0.8,
                ),
            )

    def test_audio_analysis_invalid_speech_ratio(self):
        """Test AudioAnalysis with invalid speech ratio."""
        from victor.agent.multimodal.audio_agent import AudioQualityMetrics

        with pytest.raises(ValidationError, match="speech_ratio must be between 0 and 1"):
            AudioAnalysis(
                duration=120.0,
                sample_rate=44100,
                channels=2,
                format="mp3",
                quality_metrics=AudioQualityMetrics(
                    clarity_score=0.9,
                    speech_ratio=1.5,  # Invalid
                ),
            )


class TestIntegrationScenarios:
    """Integration-style test scenarios."""

    @pytest.mark.asyncio
    async def test_full_audio_workflow(
        self, audio_agent_openai, sample_audio_path, mock_openai_provider
    ):
        """Test complete audio workflow."""
        # Mock transcription
        mock_transcription_response = Mock()
        mock_transcription_response.text = (
            "This is a test audio file with multiple segments of speech."
        )
        mock_transcription_response.language = "en"
        mock_transcription_response.duration = 30.0
        mock_transcription_response.segments = []

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=mock_transcription_response
        )
        mock_openai_provider.client = mock_client

        # Mock chat for summarization
        mock_openai_provider.chat.return_value = Mock(
            content="Audio summary: Test file with speech segments.",
            role="assistant",
        )

        # Transcribe
        transcription = await audio_agent_openai.transcribe_audio(sample_audio_path)
        assert "test audio file" in transcription.text.lower()

        # Detect language
        language = await audio_agent_openai.detect_language(sample_audio_path)
        assert language == "en"

        # Summarize
        summary = await audio_agent_openai.summarize_audio(sample_audio_path)
        assert "summary" in summary.lower()

    @pytest.mark.asyncio
    async def test_different_audio_formats(
        self, audio_agent_openai, sample_audio_path, sample_wav_path
    ):
        """Test processing different audio formats."""
        # Analyze MP3
        analysis_mp3 = await audio_agent_openai.analyze_audio(sample_audio_path)
        assert analysis_mp3.format == "mp3"

        # Analyze WAV
        analysis_wav = await audio_agent_openai.analyze_audio(sample_wav_path)
        assert analysis_wav.format == "wav"
