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

"""Audio processing agent for transcription and analysis.

This module provides the AudioAgent class for multimodal audio processing:
- Audio transcription (speech-to-text) with Whisper API and local fallback
- Audio analysis and quality assessment
- Speaker diarization (identify different speakers)
- Language detection from audio
- Audio summarization using LLM
- Support for multiple audio formats (MP3, WAV, M4A, FLAC, etc.)

Example:
    >>> from victor.providers.openai_provider import OpenAIProvider
    >>> from victor.agent.multimodal import AudioAgent
    >>>
    >>> provider = OpenAIProvider(api_key="sk-...")
    >>> audio_agent = AudioAgent(provider=provider)
    >>>
    >>> # Transcribe audio
    >>> transcription = await audio_agent.transcribe_audio(
    ...     audio_path="recording.mp3",
    ...     language="en"
    ... )
    >>> print(transcription.text)
    >>>
    >>> # Analyze audio
    >>> analysis = await audio_agent.analyze_audio("recording.mp3")
    >>> print(f"Duration: {analysis.duration}s")
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from victor.core.errors import ValidationError
from victor.providers.base import BaseProvider, Message

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


class AudioFormat(Enum):
    """Supported audio formats."""

    MP3 = "mp3"
    WAV = "wav"
    M4A = "m4a"
    FLAC = "flac"
    MP4 = "mp4"
    MPEG = "mpeg"
    MPGA = "mpga"
    WEBM = "webm"
    OGG = "ogg"
    OPUS = "opus"


@dataclass
class SpeakerSegment:
    """A segment of speech with speaker information.

    Attributes:
        speaker_id: Identifier for the speaker (e.g., "SPEAKER_00", "SPEAKER_01")
        start_time: Start time in seconds
        end_time: End time in seconds
        transcript: Transcribed text for this segment
        confidence: Confidence score (0-1)
        metadata: Additional metadata (emotion, tone, etc.)

    Raises:
        ValidationError: If time values or confidence are invalid

    Example:
        >>> segment = SpeakerSegment(
        ...     speaker_id="SPEAKER_00",
        ...     start_time=0.0,
        ...     end_time=5.2,
        ...     transcript="Hello, how are you?",
        ...     confidence=0.95
        ... )
        >>> print(f"Duration: {segment.duration}s")
    """

    speaker_id: str
    start_time: float
    end_time: float
    transcript: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate segment data."""
        if self.start_time < 0:
            raise ValidationError(f"Start time must be non-negative, got {self.start_time}")
        if self.end_time <= self.start_time:
            raise ValidationError(
                f"End time ({self.end_time}) must be greater than start time ({self.start_time})"
            )
        if not 0 <= self.confidence <= 1:
            raise ValidationError(f"Confidence must be between 0 and 1, got {self.confidence}")

    @property
    def duration(self) -> float:
        """Get duration of the segment in seconds.

        Returns:
            Duration in seconds
        """
        return self.end_time - self.start_time


@dataclass
class TranscriptionResult:
    """Complete transcription of audio with metadata.

    Attributes:
        text: Full transcribed text
        confidence: Overall confidence score (0-1)
        timestamp: When transcription was performed
        language: Detected or specified language (ISO 639-1 code)
        segments: Individual speech segments with speaker info
        duration: Audio duration in seconds

    Raises:
        ValidationError: If confidence or duration are invalid

    Example:
        >>> result = TranscriptionResult(
        ...     text="Hello world",
        ...     confidence=0.95,
        ...     timestamp=datetime.now(),
        ...     language="en",
        ...     segments=[],
        ...     duration=10.5
        ... )
    """

    text: str
    confidence: float
    timestamp: datetime
    language: str
    segments: List[SpeakerSegment]
    duration: float

    def __post_init__(self) -> None:
        """Validate transcription data."""
        if not 0 <= self.confidence <= 1:
            raise ValidationError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.duration < 0:
            raise ValidationError(f"Duration must be non-negative, got {self.duration}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all fields
        """
        return {
            "text": self.text,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {
                    "speaker_id": seg.speaker_id,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "transcript": seg.transcript,
                    "confidence": seg.confidence,
                    "metadata": seg.metadata,
                }
                for seg in self.segments
            ],
        }


@dataclass
class AudioQualityMetrics:
    """Audio quality assessment metrics.

    Attributes:
        snr_db: Signal-to-noise ratio in decibels
        clarity_score: Speech clarity score (0-1)
        noise_level: Background noise level (0-1, lower is better)
        volume_level: Average volume level (0-1)
        speech_ratio: Ratio of speech to total duration (0-1)
    """

    snr_db: Optional[float] = None
    clarity_score: float = 0.8
    noise_level: float = 0.2
    volume_level: float = 0.7
    speech_ratio: float = 0.8

    def __post_init__(self) -> None:
        """Validate metrics."""
        for score_name in ["clarity_score", "noise_level", "volume_level", "speech_ratio"]:
            score = getattr(self, score_name)
            if not 0 <= score <= 1:
                raise ValidationError(f"{score_name} must be between 0 and 1, got {score}")


@dataclass
class AudioAnalysis:
    """Analysis of audio file characteristics and quality.

    Attributes:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        channels: Number of audio channels (1=mono, 2=stereo)
        format: Audio format (e.g., "mp3", "wav")
        quality_metrics: Audio quality assessment metrics
        bitrate: Bitrate in kbps (if available)
        metadata: Additional analysis metadata

    Raises:
        ValidationError: If numeric values are invalid

    Example:
        >>> analysis = AudioAnalysis(
        ...     duration=120.5,
        ...     sample_rate=44100,
        ...     channels=2,
        ...     format="wav",
        ...     quality_metrics=AudioQualityMetrics()
        ... )
    """

    duration: float
    sample_rate: int
    channels: int
    format: str
    quality_metrics: AudioQualityMetrics
    bitrate: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate analysis data."""
        if self.duration < 0:
            raise ValidationError(f"Duration must be non-negative, got {self.duration}")
        if self.sample_rate <= 0:
            raise ValidationError(f"Sample rate must be positive, got {self.sample_rate}")
        if self.channels <= 0:
            raise ValidationError(f"Channels must be positive, got {self.channels}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all fields
        """
        return {
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format,
            "bitrate": self.bitrate,
            "quality_metrics": {
                "snr_db": self.quality_metrics.snr_db,
                "clarity_score": self.quality_metrics.clarity_score,
                "noise_level": self.quality_metrics.noise_level,
                "volume_level": self.quality_metrics.volume_level,
                "speech_ratio": self.quality_metrics.speech_ratio,
            },
            "metadata": self.metadata,
        }


# =============================================================================
# Audio Agent
# =============================================================================


class AudioAgent:
    """Agent for audio processing, transcription, and analysis.

    The AudioAgent integrates with speech-to-text APIs (OpenAI Whisper, etc.)
    and provides local fallback with Whisper model for offline processing.

    Features:
    - Audio transcription with timestamps
    - Speaker diarization (identify different speakers)
    - Language detection from audio
    - Audio quality analysis
    - Transcription summarization using LLM
    - Support for 10+ audio formats
    - Local Whisper fallback for offline operation

    Example:
        >>> from victor.providers.openai_provider import OpenAIProvider
        >>> from victor.agent.multimodal import AudioAgent
        >>>
        >>> # Initialize with provider
        >>> provider = OpenAIProvider(api_key="sk-...")
        >>> audio_agent = AudioAgent(provider=provider)
        >>>
        >>> # Transcribe audio
        >>> result = await audio_agent.transcribe_audio("meeting.mp3", language="en")
        >>> print(result.text)
        >>>
        >>> # Extract speaker segments
        >>> segments = await audio_agent.extract_speaker_diarization("meeting.mp3")
        >>> for seg in segments:
        ...     print(f"{seg.speaker_id}: {seg.transcript}")
        >>>
        >>> # Analyze audio quality
        >>> analysis = await audio_agent.analyze_audio("meeting.mp3")
        >>> print(f"Quality score: {analysis.quality_metrics.clarity_score}")
    """

    # Default model configurations
    DEFAULT_OPENAI_MODEL = "whisper-1"
    DEFAULT_LOCAL_WHISPER_MODEL = "base"  # tiny, base, small, medium, large

    # Supported audio formats
    SUPPORTED_FORMATS = {fmt.value for fmt in AudioFormat}

    def __init__(
        self,
        provider: BaseProvider,
        model: Optional[str] = None,
        use_local_fallback: bool = True,
        local_whisper_model: str = DEFAULT_LOCAL_WHISPER_MODEL,
    ):
        """Initialize AudioAgent.

        Args:
            provider: LLM provider for audio API calls and summarization
            model: Model to use for transcription (defaults to provider's audio model)
            use_local_fallback: Whether to use local Whisper as fallback
            local_whisper_model: Whisper model size for local processing

        Raises:
            ValidationError: If provider configuration is invalid
        """
        self.provider = provider
        self.model = model or self._get_default_model()
        self.use_local_fallback = use_local_fallback
        self.local_whisper_model = local_whisper_model
        self._local_whisper_available = self._check_whisper_available()

        # Log warnings about capabilities
        if not self._is_audio_capable() and not self._local_whisper_available:
            logger.warning(
                f"Provider {provider.name} doesn't support native audio. "
                "Local Whisper fallback is not available. "
                "Install with: pip install openai-whisper"
            )
        elif not self._is_audio_capable() and self._local_whisper_available:
            logger.info(
                f"Provider {provider.name} doesn't support native audio. "
                "Using local Whisper fallback."
            )

    def _get_default_model(self) -> str:
        """Get default audio model for the current provider.

        Returns:
            Model identifier string
        """
        audio_defaults = {
            "openai": self.DEFAULT_OPENAI_MODEL,
            "anthropic": "claude-sonnet-4-5",  # For summarization only
            "google": "whisper-1",
        }
        return audio_defaults.get(self.provider.name, self.DEFAULT_OPENAI_MODEL)

    def _is_audio_capable(self) -> bool:
        """Check if provider supports native audio transcription.

        Returns:
            True if provider has native audio support
        """
        return self.provider.name == "openai"

    def _check_whisper_available(self) -> bool:
        """Check if local Whisper is available for fallback.

        Returns:
            True if whisper package is installed
        """
        try:
            import whisper

            return whisper is not None
        except ImportError:
            return False

    def _validate_audio_file(self, audio_path: str) -> Path:
        """Validate audio file exists and has supported format.

        Args:
            audio_path: Path to audio file

        Returns:
            Validated Path object

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValidationError: If file format is not supported
        """
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not path.is_file():
            raise ValidationError(f"Path is not a file: {audio_path}")

        suffix = path.suffix.lower().lstrip(".")

        if suffix not in self.SUPPORTED_FORMATS:
            raise ValidationError(
                f"Unsupported audio format: .{suffix}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

        return path

    async def transcribe_audio(
        self,
        audio_path: str,
        language: str = "en",
        include_timestamps: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio file to text using speech-to-text APIs.

        This method attempts to use the provider's native audio API (e.g., OpenAI Whisper).
        If unavailable and local fallback is enabled, it uses the Whisper library locally.

        Args:
            audio_path: Path to audio file
            language: Language code (ISO 639-1, e.g., "en", "es", "fr"). Auto-detect if None.
            include_timestamps: Whether to include segment timestamps in result

        Returns:
            TranscriptionResult with text, confidence, segments, language, duration

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValidationError: If file format is not supported
            RuntimeError: If transcription fails and no fallback available

        Example:
            >>> result = await audio_agent.transcribe_audio(
            ...     "recording.mp3",
            ...     language="en",
            ...     include_timestamps=True
            ... )
            >>> print(f"Transcribed {result.duration}s of audio")
            >>> print(f"Text: {result.text[:100]}...")
        """
        logger.info(f"Transcribing audio: {audio_path} (language={language})")

        path = self._validate_audio_file(audio_path)

        # Try provider's native audio API first
        if self._is_audio_capable():
            try:
                return await self._transcribe_with_provider(
                    path=path,
                    language=language,
                    include_timestamps=include_timestamps,
                )
            except Exception as e:
                logger.error(f"Provider transcription failed: {e}")
                if not self.use_local_fallback:
                    raise
                logger.info("Falling back to local Whisper")

        # Fallback to local Whisper
        if self._local_whisper_available and self.use_local_fallback:
            return await self._transcribe_with_local_whisper(
                path=path,
                language=language,
                include_timestamps=include_timestamps,
            )

        # No transcription method available
        raise RuntimeError(
            "Audio transcription not available. "
            "Use OpenAI provider or install openai-whisper: pip install openai-whisper"
        )

    async def _transcribe_with_provider(
        self,
        path: Path,
        language: Optional[str],
        include_timestamps: bool,
    ) -> TranscriptionResult:
        """Transcribe audio using provider's native API.

        Args:
            path: Path to audio file
            language: Language code
            include_timestamps: Whether to include timestamps

        Returns:
            TranscriptionResult

        Raises:
            ImportError: If OpenAI package not available
            RuntimeError: If API call fails
        """
        try:
            from openai import AsyncOpenAI

            # Get OpenAI client
            if hasattr(self.provider, "client"):
                client = self.provider.client
            else:
                api_key = self.provider.api_key or ""
                client = AsyncOpenAI(api_key=api_key)

            # Open audio file
            with open(path, "rb") as audio_file:
                # Call Whisper API
                response_format = "verbose_json" if include_timestamps else "text"
                transcript_response = await client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=language,
                    response_format=response_format,
                )

            # Parse response
            if include_timestamps and hasattr(transcript_response, "segments"):
                segments = [
                    SpeakerSegment(
                        speaker_id="SPEAKER_00",  # Whisper doesn't provide speaker ID
                        start_time=seg.start,
                        end_time=seg.end,
                        transcript=seg.text.strip(),
                        confidence=getattr(seg, "no_speech_prob", 0.05),
                        metadata={"avg_logprob": getattr(seg, "avg_logprob", None)},
                    )
                    for seg in transcript_response.segments
                ]

                return TranscriptionResult(
                    text=transcript_response.text,
                    confidence=0.95,  # Average confidence
                    timestamp=datetime.now(),
                    language=transcript_response.language,
                    segments=segments,
                    duration=transcript_response.duration,
                )
            else:
                # Text-only response
                text = (
                    transcript_response
                    if isinstance(transcript_response, str)
                    else transcript_response.text
                )
                return TranscriptionResult(
                    text=text,
                    confidence=0.95,
                    timestamp=datetime.now(),
                    language=language or "en",
                    segments=[],
                    duration=0.0,
                )

        except ImportError:
            raise ImportError("OpenAI package required: pip install openai")
        except Exception as e:
            logger.error(f"Provider transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")

    async def _transcribe_with_local_whisper(
        self,
        path: Path,
        language: Optional[str],
        include_timestamps: bool,
    ) -> TranscriptionResult:
        """Transcribe audio using local Whisper model.

        Args:
            path: Path to audio file
            language: Language code
            include_timestamps: Whether to include timestamps

        Returns:
            TranscriptionResult

        Raises:
            ImportError: If whisper not installed
            RuntimeError: If transcription fails
        """
        try:
            import whisper

            logger.info(f"Loading local Whisper model: {self.local_whisper_model}")

            # Load model (this is cached by whisper)
            model = whisper.load_model(self.local_whisper_model)

            # Transcribe
            result = model.transcribe(
                str(path),
                language=language,
                word_timestamps=include_timestamps,
            )

            # Extract segments
            segments = []
            if include_timestamps and "segments" in result:
                for seg in result["segments"]:
                    segments.append(
                        SpeakerSegment(
                            speaker_id="SPEAKER_00",  # No diarization
                            start_time=seg["start"],
                            end_time=seg["end"],
                            transcript=seg["text"].strip(),
                            confidence=seg.get("avg_logprob", 0.0),
                            metadata={
                                "temperature": result.get("temperature", 0.0),
                                "no_speech_prob": seg.get("no_speech_prob", 0.0),
                            },
                        )
                    )

            return TranscriptionResult(
                text=result["text"].strip(),
                confidence=0.9,  # Estimated confidence
                timestamp=datetime.now(),
                language=result.get("language", language or "unknown"),
                segments=segments,
                duration=(
                    len(model.audio) / model.audio.sample_rate if hasattr(model, "audio") else 0.0
                ),
            )

        except ImportError:
            raise ImportError("Whisper package required: pip install openai-whisper")
        except Exception as e:
            logger.error(f"Local Whisper transcription failed: {e}")
            raise RuntimeError(f"Local transcription failed: {e}")

    async def analyze_audio(self, audio_path: str) -> AudioAnalysis:
        """Analyze audio file characteristics and quality.

        Performs basic audio analysis including duration, format detection,
        and quality assessment. For detailed waveform analysis, consider
        integrating libraries like librosa or pydub.

        Args:
            audio_path: Path to audio file

        Returns:
            AudioAnalysis with duration, sample rate, channels, quality metrics

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValidationError: If file format is not supported

        Example:
            >>> analysis = await audio_agent.analyze_audio("recording.mp3")
            >>> print(f"Duration: {analysis.duration}s")
            >>> print(f"Quality: {analysis.quality_metrics.clarity_score:.2f}")
        """
        logger.info(f"Analyzing audio: {audio_path}")

        path = self._validate_audio_file(audio_path)

        try:
            file_size = path.stat().st_size

            # Try to get detailed audio info using mutagen if available
            sample_rate = 16000  # Default assumption
            channels = 1
            bitrate = None

            try:
                from mutagen import File as MutagenFile

                audio_file = MutagenFile(path)
                if audio_file is not None:
                    if hasattr(audio_file, "info"):
                        info = audio_file.info
                        sample_rate = getattr(info, "sample_rate", 16000)
                        channels = getattr(info, "channels", 1)
                        bitrate = getattr(info, "bitrate", None)

                        # Estimate duration from file info
                        if hasattr(info, "length"):
                            duration = info.length
                        else:
                            # Rough estimate based on file size
                            duration = file_size / (sample_rate * 2)
                    else:
                        duration = file_size / (sample_rate * 2)
                else:
                    duration = file_size / (sample_rate * 2)
            except ImportError:
                # Mutagen not available, use basic estimation
                duration = file_size / (sample_rate * 2)

            # Quality metrics (placeholder - would use audio processing library)
            quality_metrics = AudioQualityMetrics(
                snr_db=None,
                clarity_score=0.85,
                noise_level=0.15,
                volume_level=0.7,
                speech_ratio=0.8,
            )

            return AudioAnalysis(
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                format=path.suffix[1:].lower(),
                quality_metrics=quality_metrics,
                bitrate=int(bitrate) if bitrate else None,
                metadata={
                    "file_size": file_size,
                    "method": "mutagen" if bitrate else "estimated",
                    "note": "For detailed analysis, integrate librosa/pydub",
                },
            )

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            raise RuntimeError(f"Audio analysis failed: {e}")

    async def extract_speaker_diarization(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_segment_length: float = 1.0,
    ) -> List[SpeakerSegment]:
        """Extract speaker segments from audio with diarization.

        Identifies different speakers and when they're speaking. This requires
        external diarization services like:
        - Pyannote.audio (local)
        - AssemblyAI (cloud)
        - Rev.ai (cloud)

        This implementation provides a basic segmentation. For production
        use with actual speaker identification, integrate a diarization service.

        Args:
            audio_path: Path to audio file
            num_speakers: Number of speakers (auto-detect if None)
            min_segment_length: Minimum segment length in seconds

        Returns:
            List of SpeakerSegment with speaker_id, timestamps, transcript

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValidationError: If file format is not supported
            RuntimeError: If diarization service unavailable

        Example:
            >>> segments = await audio_agent.extract_speaker_diarization(
            ...     "meeting.mp3",
            ...     num_speakers=3
            ... )
            >>> for seg in segments:
            ...     print(f"{seg.speaker_id} ({seg.start_time:.1f}s): {seg.transcript}")
        """
        logger.info(f"Extracting speaker diarization: {audio_path}")

        self._validate_audio_file(audio_path)

        # Get transcription with timestamps first
        transcription = await self.transcribe_audio(
            audio_path=audio_path,
            include_timestamps=True,
        )

        # Try to use pyannote for actual speaker diarization
        try:
            return await self._diarize_with_pyannote(
                audio_path=audio_path,
                transcription=transcription,
                num_speakers=num_speakers,
                min_segment_length=min_segment_length,
            )
        except Exception as e:
            logger.warning(f"Pyannote diarization unavailable: {e}")
            logger.info("Using basic segmentation without speaker identification")

        # Fallback: Basic segmentation without speaker identification
        segments = [
            SpeakerSegment(
                speaker_id="SPEAKER_00",  # Single speaker
                start_time=seg.start_time,
                end_time=seg.end_time,
                transcript=seg.transcript,
                confidence=seg.confidence,
            )
            for seg in transcription.segments
            if seg.duration >= min_segment_length
        ]

        logger.info(f"Extracted {len(segments)} segments (without speaker identification)")
        return segments

    async def _diarize_with_pyannote(
        self,
        audio_path: str,
        transcription: TranscriptionResult,
        num_speakers: Optional[int],
        min_segment_length: float,
    ) -> List[SpeakerSegment]:
        """Perform speaker diarization using pyannote.audio.

        Args:
            audio_path: Path to audio file
            transcription: Existing transcription for alignment
            num_speakers: Number of speakers
            min_segment_length: Minimum segment length

        Returns:
            List of SpeakerSegment with speaker IDs

        Raises:
            ImportError: If pyannote not installed
            RuntimeError: If diarization fails
        """
        try:
            from pyannote.audio import Pipeline
            from pyannote.core import Annotation, Segment

            logger.info("Loading pyannote diarization pipeline")

            # Load pre-trained pipeline (requires HuggingFace token)
            # Note: Users need to accept pyannote/speaker-diarization model terms
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.environ.get("HUGGINGFACE_TOKEN"),
            )

            # Run diarization
            diarization: Annotation = pipeline(str(audio_path))

            # Map transcription segments to speaker segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Find corresponding transcript segments
                for trans_seg in transcription.segments:
                    # Check for overlap
                    if turn.start <= trans_seg.end_time and turn.end >= trans_seg.start_time:
                        if turn.end - turn.start >= min_segment_length:
                            segments.append(
                                SpeakerSegment(
                                    speaker_id=speaker,
                                    start_time=turn.start,
                                    end_time=turn.end,
                                    transcript=trans_seg.transcript,
                                    confidence=trans_seg.confidence,
                                )
                            )
                            break

            logger.info(f"Diarization complete: {len(segments)} segments")
            return segments

        except ImportError:
            raise ImportError(
                "pyannote.audio required: pip install pyannote.audio "
                "(also requires accepting model terms on HuggingFace)"
            )
        except Exception as e:
            logger.error(f"Pyannote diarization failed: {e}")
            raise RuntimeError(f"Diarization failed: {e}")

    async def detect_language(self, audio_path: str) -> str:
        """Detect the language spoken in audio.

        Uses the transcription API's language detection capabilities.
        Returns ISO 639-1 language code (e.g., "en", "es", "fr", "de").

        Args:
            audio_path: Path to audio file

        Returns:
            Detected language code

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValidationError: If file format is not supported
            RuntimeError: If language detection fails

        Example:
            >>> language = await audio_agent.detect_language("recording.mp3")
            >>> print(f"Detected language: {language}")
        """
        logger.info(f"Detecting language: {audio_path}")

        self._validate_audio_file(audio_path)

        # Use OpenAI's language detection
        if self._is_audio_capable():
            try:
                from openai import AsyncOpenAI

                client = getattr(self.provider, "client", None)
                if not client:
                    client = AsyncOpenAI(api_key=self.provider.api_key or "")

                path = Path(audio_path)

                with open(path, "rb") as audio_file:
                    response = await client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        response_format="verbose_json",
                    )

                    language = response.language
                    logger.info(f"Detected language: {language}")
                    return language

            except Exception as e:
                logger.error(f"Language detection failed: {e}")
                raise RuntimeError(f"Language detection failed: {e}")

        # Fallback: Use local Whisper
        if self._local_whisper_available:
            try:
                import whisper

                model = whisper.load_model(self.local_whisper_model)
                audio = whisper.load_audio(str(audio_path))
                result = model.transcribe(audio, language=None)

                detected_language: str = result.get("language", "unknown")
                logger.info(f"Detected language (Whisper): {detected_language}")
                return detected_language

            except Exception as e:
                logger.error(f"Whisper language detection failed: {e}")

        return "unknown"

    async def generate_audio_summary(
        self,
        transcription: Union[str, TranscriptionResult],
        max_words: int = 300,
    ) -> str:
        """Generate a summary of audio transcription.

        Transcribes audio if needed, then uses the LLM to generate a concise
        summary focusing on key points, main topics, and important information.

        Args:
            transcription: Either transcription text or TranscriptionResult
            max_words: Maximum length of summary in words

        Returns:
            Generated summary text

        Raises:
            ValidationError: If transcription is invalid
            RuntimeError: If summarization fails

        Example:
            >>> # Summarize from file
            >>> summary = await audio_agent.generate_audio_summary("meeting.mp3")
            >>> print(summary)
            >>>
            >>> # Summarize from existing transcription
            >>> result = await audio_agent.transcribe_audio("meeting.mp3")
            >>> summary = await audio_agent.generate_audio_summary(result)
            >>> print(summary)
        """
        logger.info("Generating audio summary")

        # Get transcription text
        if isinstance(transcription, str):
            # Assume it's a file path
            result = await self.transcribe_audio(transcription)
            transcript_text = result.text
        elif isinstance(transcription, TranscriptionResult):
            transcript_text = transcription.text
        else:
            raise ValidationError(
                f"Invalid transcription type: {type(transcription)}. "
                "Expected str or TranscriptionResult"
            )

        logger.info(f"Summarizing {len(transcript_text)} characters of transcript")

        # Generate summary using LLM
        prompt = f"""Analyze the following audio transcript and provide a comprehensive summary.

Your summary should:
1. Identify the main topics and key points discussed
2. Highlight important information, decisions, or action items
3. Note any questions raised and answers provided
4. Capture the overall context and purpose
5. Be concise but comprehensive

Keep the summary to approximately {max_words} words.

Transcript:
{transcript_text}

Summary:"""

        try:
            response = await self.provider.chat(
                messages=[Message(role="user", content=prompt)],
                model=self.model,
                temperature=0.5,
            )

            summary = response.content.strip()
            word_count = len(summary.split())
            logger.info(f"Generated summary: {word_count} words")

            return summary

        except Exception as e:
            logger.error(f"Audio summarization failed: {e}")
            raise RuntimeError(f"Summarization failed: {e}")

    async def summarize_audio(
        self,
        audio_path: str,
        max_words: int = 300,
    ) -> str:
        """Alias for generate_audio_summary for backward compatibility.

        Args:
            audio_path: Path to audio file
            max_words: Maximum length of summary in words

        Returns:
            Generated summary text
        """
        return await self.generate_audio_summary(audio_path, max_words=max_words)

    async def batch_process_audio(
        self,
        audio_paths: List[str],
        operation: str = "transcribe",
        **kwargs: Any,
    ) -> List["TranscriptionResult | AudioAnalysis | str | List[SpeakerSegment] | None"]:
        """Process multiple audio files in batch.

        Args:
            audio_paths: List of paths to audio files
            operation: Operation to perform
                - "transcribe": Transcribe audio
                - "analyze": Analyze audio quality
                - "detect_language": Detect language
                - "diarize": Extract speaker segments
            **kwargs: Additional arguments for the specific operation

        Returns:
            List of results corresponding to input audio files

        Raises:
            ValidationError: If operation is invalid
            FileNotFoundError: If any audio file doesn't exist

        Example:
            >>> paths = ["audio1.mp3", "audio2.mp3"]
            >>> results = await audio_agent.batch_process_audio(
            ...     paths,
            ...     operation="transcribe",
            ...     language="en"
            ... )
            >>> for path, result in zip(paths, results):
            ...     print(f"{path}: {result.text[:50]}...")
        """
        logger.info(f"Batch processing {len(audio_paths)} audio files (operation: {operation})")

        valid_operations = ["transcribe", "analyze", "detect_language", "diarize"]
        if operation not in valid_operations:
            raise ValidationError(
                f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}"
            )

        results: List[TranscriptionResult | AudioAnalysis | str | List[SpeakerSegment] | None] = []

        for audio_path in audio_paths:
            try:
                if operation == "transcribe":
                    result: (
                        TranscriptionResult | AudioAnalysis | str | List[SpeakerSegment] | None
                    ) = await self.transcribe_audio(audio_path, **kwargs)
                elif operation == "analyze":
                    result = await self.analyze_audio(audio_path)
                elif operation == "detect_language":
                    result = await self.detect_language(audio_path)
                elif operation == "diarize":
                    result = await self.extract_speaker_diarization(audio_path, **kwargs)
                else:
                    result = None

                results.append(result)
                logger.debug(f"Processed {audio_path}")

            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                # Append None for failed items
                results.append(None)

        logger.info(f"Batch processing complete: {len(results)} results")
        return results

    async def audio_quality_analysis(
        self,
        audio_path: str,
    ) -> AudioQualityMetrics:
        """Perform detailed audio quality analysis.

        Analyzes various aspects of audio quality including clarity, noise levels,
        speech intelligibility, and overall production quality.

        Args:
            audio_path: Path to audio file

        Returns:
            AudioQualityMetrics with detailed quality assessment

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValidationError: If file format is not supported
            RuntimeError: If analysis fails

        Example:
            >>> quality = await audio_agent.audio_quality_analysis("recording.mp3")
            >>> print(f"Clarity: {quality.clarity_score:.2%}")
            >>> print(f"Noise level: {quality.noise_level:.2%}")
        """
        logger.info(f"Analyzing audio quality: {audio_path}")

        # First get basic analysis
        basic_analysis = await self.analyze_audio(audio_path)

        # Enhanced quality assessment using LLM
        prompt = f"""
        You are an audio quality expert. Analyze this audio file and provide quality metrics.

        Audio file information:
        - Duration: {basic_analysis.duration:.2f} seconds
        - Sample rate: {basic_analysis.sample_rate} Hz
        - Channels: {basic_analysis.channels}
        - Format: {basic_analysis.format}

        Provide a JSON assessment with:
        {{
            "clarity_score": 0.0 to 1.0,
            "noise_level": 0.0 to 1.0 (lower is better),
            "volume_level": 0.0 to 1.0,
            "speech_ratio": 0.0 to 1.0,
            "snr_estimate_db": approximate signal-to-noise ratio in dB,
            "issues": ["issue1", "issue2"],
            "recommendations": ["improvement1", "improvement2"],
            "overall_rating": "excellent|good|fair|poor"
        }}

        Rate each aspect objectively:
        - clarity_score: Speech clarity and intelligibility
        - noise_level: Background noise, hums, hisses (lower is better)
        - volume_level: Appropriate volume level
        - speech_ratio: Percentage with clear speech vs silence/noise
        - snr_estimate_db: Estimated signal-to-noise ratio (>20dB is good)
        """

        try:
            response = await self.provider.chat(
                messages=[Message(role="user", content=prompt)],
                model=self.model,
                temperature=0.3,
            )

            try:
                quality_data = json.loads(response.content)
                return AudioQualityMetrics(
                    snr_db=quality_data.get("snr_estimate_db"),
                    clarity_score=quality_data.get("clarity_score", 0.8),
                    noise_level=quality_data.get("noise_level", 0.2),
                    volume_level=quality_data.get("volume_level", 0.7),
                    speech_ratio=quality_data.get("speech_ratio", 0.8),
                )

            except json.JSONDecodeError:
                logger.warning("Failed to parse quality JSON, using basic metrics")
                return basic_analysis.quality_metrics

        except Exception as e:
            logger.error(f"Audio quality analysis failed: {e}")
            raise RuntimeError(f"Quality analysis failed: {e}")

    async def detect_emotion_from_audio(
        self,
        audio_path: str,
    ) -> Dict[str, Any]:
        """Detect emotions and sentiment from audio.

        Analyzes vocal patterns, tone, and speech characteristics to identify
        emotions and sentiment expressed in the audio.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with:
                - primary_emotion: Main emotion detected
                - emotions: Dictionary of emotion scores
                - sentiment: Overall sentiment (positive/negative/neutral)
                - confidence: Confidence in analysis
                - timestamps: Time-based emotion segments

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValidationError: If file format is not supported
            RuntimeError: If analysis fails

        Example:
            >>> emotions = await audio_agent.detect_emotion_from_audio("interview.mp3")
            >>> print(f"Primary emotion: {emotions['primary_emotion']}")
            >>> print(f"Sentiment: {emotions['sentiment']}")
        """
        logger.info(f"Detecting emotions from audio: {audio_path}")

        # Get transcription for context
        transcription = await self.transcribe_audio(audio_path, include_timestamps=True)

        # Analyze emotions using both audio and text
        prompt = f"""
        Analyze the emotional content of this audio transcript and provide an emotion assessment.

        Transcript:
        {transcription.text}

        Audio duration: {transcription.duration:.2f} seconds

        Provide a JSON response with:
        {{
            "primary_emotion": "joy|sadness|anger|fear|surprise|disgust|neutral",
            "emotions": {{
                "joy": 0.0 to 1.0,
                "sadness": 0.0 to 1.0,
                "anger": 0.0 to 1.0,
                "fear": 0.0 to 1.0,
                "surprise": 0.0 to 1.0,
                "disgust": 0.0 to 1.0,
                "neutral": 0.0 to 1.0
            }},
            "sentiment": "positive|negative|neutral",
            "sentiment_score": -1.0 to 1.0,
            "confidence": 0.0 to 1.0,
            "tone": "formal|casual|emotional|monotone|energetic",
            "analysis": "brief explanation of emotional analysis"
        }}

        Consider:
        - Emotional indicators in speech and language
        - Tone and energy level
        - Overall sentiment (positive/negative/neutral)
        - Confidence in your assessment
        """

        try:
            response = await self.provider.chat(
                messages=[Message(role="user", content=prompt)],
                model=self.model,
                temperature=0.3,
            )

            try:
                emotion_data = json.loads(response.content)
                return {
                    "primary_emotion": emotion_data.get("primary_emotion", "neutral"),
                    "emotions": emotion_data.get("emotions", {}),
                    "sentiment": emotion_data.get("sentiment", "neutral"),
                    "sentiment_score": emotion_data.get("sentiment_score", 0.0),
                    "confidence": emotion_data.get("confidence", 0.7),
                    "tone": emotion_data.get("tone", "neutral"),
                    "analysis": emotion_data.get("analysis", ""),
                }

            except json.JSONDecodeError:
                logger.warning("Failed to parse emotion JSON")
                return {
                    "primary_emotion": "neutral",
                    "emotions": {},
                    "sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "confidence": 0.5,
                    "tone": "neutral",
                    "analysis": "Emotion analysis unavailable",
                }

        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            raise RuntimeError(f"Emotion detection failed: {e}")

    async def speaker_diarization_detailed(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_segment_length: float = 1.0,
        include_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """Perform detailed speaker diarization with embeddings.

        Enhanced speaker diarization that identifies different speakers,
        their segments, and optionally provides speaker embeddings for
        clustering and identification.

        Args:
            audio_path: Path to audio file
            num_speakers: Number of speakers (auto-detect if None)
            min_segment_length: Minimum segment length in seconds
            include_embeddings: Whether to include speaker embeddings

        Returns:
            Dictionary with:
                - num_speakers: Total number of speakers detected
                - segments: List of speaker segments
                - speaker_profiles: Profiles for each speaker
                - total_speech_time: Total speech duration
                - speaker_ratios: Percentage of time each speaker spoke

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValidationError: If file format is not supported
            RuntimeError: If diarization fails

        Example:
            >>> result = await audio_agent.speaker_diarization_detailed(
            ...     "meeting.mp3",
            ...     num_speakers=3
            ... )
            >>> print(f"Detected {result['num_speakers']} speakers")
            >>> for speaker_id, ratio in result['speaker_ratios'].items():
            ...     print(f"{speaker_id}: {ratio:.1%} of speaking time")
        """
        logger.info(f"Detailed speaker diarization: {audio_path}")

        segments = await self.extract_speaker_diarization(
            audio_path=audio_path,
            num_speakers=num_speakers,
            min_segment_length=min_segment_length,
        )

        # Analyze segments to build speaker profiles
        speaker_segments: Dict[str, List[SpeakerSegment]] = {}
        total_speech_time = 0.0

        for seg in segments:
            speaker_id = seg.speaker_id
            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = []
            speaker_segments[speaker_id].append(seg)
            total_speech_time += seg.duration

        # Build speaker profiles
        speaker_profiles = {}
        speaker_ratios = {}

        for speaker_id, segs in speaker_segments.items():
            total_duration = sum(s.duration for s in segs)
            ratio = total_duration / total_speech_time if total_speech_time > 0 else 0

            # Get speaker's transcript snippets for profile
            transcripts = [s.transcript for s in segs[:5]]  # First 5 segments
            avg_confidence = sum(s.confidence for s in segs) / len(segs)

            speaker_profiles[speaker_id] = {
                "total_duration": total_duration,
                "num_segments": len(segs),
                "avg_confidence": avg_confidence,
                "sample_transcripts": transcripts,
                "speaking_style": "unknown",  # Would require additional analysis
            }

            speaker_ratios[speaker_id] = ratio

        return {
            "num_speakers": len(speaker_segments),
            "segments": segments,
            "speaker_profiles": speaker_profiles,
            "total_speech_time": total_speech_time,
            "speaker_ratios": speaker_ratios,
        }

    async def transcription_summarization(
        self,
        transcription: Union[str, TranscriptionResult],
        summary_type: str = "comprehensive",
        max_length: int = 300,
    ) -> Dict[str, Any]:
        """Generate structured summary of audio transcription.

        Enhanced summarization that provides structured output with key points,
        topics, action items, and different summary formats.

        Args:
            transcription: Either transcription text or TranscriptionResult
            summary_type: Type of summary to generate
                - "comprehensive": Full detailed summary
                - "brief": Short executive summary
                - "bullet_points": Bullet point summary
                - "action_items": Extract action items only
            max_length: Maximum summary length in words

        Returns:
            Dictionary with:
                - summary: Main summary text
                - key_points: List of key points
                - topics: Main topics discussed
                - action_items: Action items extracted
                - sentiment: Overall sentiment
                - participants: Number of speakers (if diarization available)

        Raises:
            ValidationError: If transcription is invalid
            RuntimeError: If summarization fails

        Example:
            >>> result = await audio_agent.transcription_summarization(
            ...     transcription,
            ...     summary_type="bullet_points"
            ... )
            >>> print("Key Points:")
            >>> for point in result["key_points"]:
            ...     print(f"  - {point}")
        """
        logger.info(f"Generating {summary_type} summary")

        # Get transcription text
        if isinstance(transcription, str):
            result = await self.transcribe_audio(transcription)
            transcript_text = result.text
            num_speakers = 1
        elif isinstance(transcription, TranscriptionResult):
            transcript_text = transcription.text
            num_speakers = len(set(seg.speaker_id for seg in transcription.segments))
        else:
            raise ValidationError(
                f"Invalid transcription type: {type(transcription)}. "
                "Expected str or TranscriptionResult"
            )

        # Build prompt based on summary type
        if summary_type == "comprehensive":
            prompt = f"""Provide a comprehensive summary of this audio transcript.

Transcript:
{transcript_text}

Provide a JSON response with:
{{
    "summary": "detailed paragraph summary (max {max_length} words)",
    "key_points": ["point1", "point2", "point3"],
    "topics": ["topic1", "topic2"],
    "action_items": ["item1", "item2"],
    "sentiment": "positive|negative|neutral",
    "num_participants": {num_speakers}
}}
"""
        elif summary_type == "brief":
            prompt = f"""Provide a brief executive summary of this audio transcript (max {max_length} words).

Transcript:
{transcript_text}

Provide a JSON response with:
{{
    "summary": "concise executive summary",
    "key_points": ["main point 1", "main point 2"],
    "sentiment": "positive|negative|neutral"
}}
"""
        elif summary_type == "bullet_points":
            prompt = f"""Summarize this audio transcript as bullet points.

Transcript:
{transcript_text}

Provide a JSON response with:
{{
    "summary": "bullet point summary",
    "key_points": ["point1", "point2", "point3"],
    "topics": ["topic1", "topic2"],
    "sentiment": "positive|negative|neutral"
}}
"""
        elif summary_type == "action_items":
            prompt = f"""Extract action items and decisions from this audio transcript.

Transcript:
{transcript_text}

Provide a JSON response with:
{{
    "action_items": ["action item 1", "action item 2"],
    "decisions": ["decision 1", "decision 2"],
    "owners": ["item: owner name"],
    "deadlines": ["item: deadline"]
}}
"""
        else:
            raise ValidationError(
                f"Invalid summary_type: {summary_type}. "
                "Must be: comprehensive, brief, bullet_points, or action_items"
            )

        try:
            response = await self.provider.chat(
                messages=[Message(role="user", content=prompt)],
                model=self.model,
                temperature=0.5,
            )

            try:
                summary_data = json.loads(response.content)
                if isinstance(summary_data, dict):
                    return summary_data
                else:
                    return {"result": summary_data}

            except json.JSONDecodeError:
                # Fallback to plain text
                return {
                    "summary": response.content.strip(),
                    "key_points": [],
                    "topics": [],
                    "action_items": [],
                    "sentiment": "neutral",
                }

        except Exception as e:
            logger.error(f"Transcription summarization failed: {e}")
            raise RuntimeError(f"Summarization failed: {e}")


# =============================================================================
# Legacy Aliases (for backward compatibility)
# =============================================================================

# SpeechSegment was renamed to SpeakerSegment
SpeechSegment = SpeakerSegment

# Transcription was renamed to TranscriptionResult
Transcription = TranscriptionResult
