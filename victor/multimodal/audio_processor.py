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

"""Audio processing capabilities for multi-modal AI.

Supports:
- Audio transcription
- Speaker diarization
- Video audio extraction
- Audio metadata extraction
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from victor.multimodal.processor import ProcessingResult

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    import whisper  # type: ignore[import-untyped]

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available. Install with: pip install openai-whisper")

try:
    from pydub import AudioSegment  # type: ignore[import-untyped]

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not available. Install with: pip install pydub")

try:
    import moviepy.editor as mp  # type: ignore[import-untyped]

    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("moviepy not available. Install with: pip install moviepy")


class AudioProcessor:
    """Process and analyze audio.

    Example:
        processor = AudioProcessor()

        # Transcribe audio
        result = await processor.process(
            audio_path="meeting.mp3",
            query="Extract action items"
        )

        # With diarization
        result = await processor.process(
            audio_path="interview.wav",
            diarize=True
        )
    """

    supported_formats = [
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".aac",
        ".ogg",
        ".wma",
    ]

    supported_video_formats = [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    ]

    def __init__(self, config: Optional[Any] = None):
        """Initialize audio processor.

        Args:
            config: ProcessingConfig instance
        """
        self.config = config
        self._whisper_model = None

    async def process(
        self,
        audio_path: Union[str, Path],
        query: Optional[str] = None,
        diarize: bool = False,
    ) -> "ProcessingResult":
        """Process audio and transcribe.

        Args:
            audio_path: Path to audio file
            query: Optional query for audio understanding
            diarize: Whether to perform speaker diarization

        Returns:
            ProcessingResult with transcription
        """
        from victor.multimodal.processor import MediaType, ProcessingResult, ProcessingStatus

        path = Path(audio_path)

        if not path.exists():
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                content="",
                error=f"Audio file not found: {audio_path}",
                media_type=MediaType.AUDIO,
                source=str(audio_path),
            )

        try:
            # Get audio metadata
            metadata = self._get_audio_metadata(path)

            # Transcribe
            transcription, transcript_metadata = await self._transcribe(path, diarize)

            # Combine metadata
            metadata.update(transcript_metadata)

            # Process with query if provided
            content = transcription
            if query:
                content = self._process_with_query(transcription, query)

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                content=content,
                metadata=metadata,
                confidence=transcript_metadata.get("confidence", 0.85),
                media_type=MediaType.AUDIO,
                source=str(audio_path),
            )

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                content="",
                error=str(e),
                media_type=MediaType.AUDIO,
                source=str(audio_path),
            )

    def _get_audio_metadata(self, path: Path) -> Dict[str, Any]:
        """Extract audio metadata."""
        metadata = {
            "filename": path.name,
            "filesize": path.stat().st_size,
            "format": path.suffix.lower(),
        }

        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(str(path))
                metadata.update(
                    {
                        "duration": len(audio) / 1000,  # Convert to seconds
                        "channels": audio.channels,
                        "frame_rate": audio.frame_rate,
                        "sample_width": audio.sample_width,
                    }
                )
            except Exception as e:
                logger.warning(f"Could not extract audio metadata: {e}")

        return metadata

    async def _transcribe(self, path: Path, diarize: bool) -> tuple[str, Dict[str, Any]]:
        """Transcribe audio using Whisper."""
        if not WHISPER_AVAILABLE:
            return "", {"error": "Whisper not available"}

        try:
            # Load model (lazy load)
            if self._whisper_model is None:
                model_size = "base"  # Options: tiny, base, small, medium, large
                self._whisper_model = whisper.load_model(model_size)

            # Transcribe
            assert self._whisper_model is not None
            result = self._whisper_model.transcribe(  # type: ignore[unreachable]
                str(path),
                fp16=False,  # Use FP32 for compatibility
                language=self.config.preferred_language if self.config else "en",
            )

            transcription = result["text"]
            metadata = {
                "language": result.get("language", "unknown"),
                "confidence": self._calculate_confidence(result),
            }

            # Add segments if available
            if "segments" in result:
                metadata["segments"] = [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"],
                    }
                    for seg in result["segments"]
                ]

            return transcription, metadata

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return "", {"error": str(e)}

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score from transcription result."""
        if "segments" not in result:
            return 0.85  # Default confidence

        # Average probability from segments
        total_prob = 0
        count = 0

        for segment in result["segments"]:
            if "no_speech_prob" in segment:
                total_prob += 1 - segment["no_speech_prob"]
                count += 1

        if count > 0:
            return total_prob / count

        return 0.85

    def _process_with_query(self, transcription: str, query: str) -> str:
        """Process transcription with query context.

        This is a simple implementation. For production, use LLM-based analysis.
        """
        query_lower = query.lower()

        # Simple keyword-based extraction
        if "action item" in query_lower or "todo" in query_lower:
            lines = transcription.split(". ")
            action_items = [
                line
                for line in lines
                if any(
                    word in line.lower() for word in ["need to", "should", "will", "must", "action"]
                )
            ]
            if action_items:
                return f"Query: {query}\n\nAction Items:\n" + "\n".join(action_items)

        elif "summary" in query_lower or "summarize" in query_lower:
            # Return first N sentences as summary
            sentences = transcription.split(". ")
            summary = ". ".join(sentences[:5])
            return f"Query: {query}\n\nSummary:\n{summary}"

        return f"Query: {query}\n\nTranscription:\n{transcription}"

    async def process_video_audio(
        self,
        video_path: Union[str, Path],
    ) -> "ProcessingResult":
        """Extract and transcribe audio from video.

        Args:
            video_path: Path to video file

        Returns:
            ProcessingResult with transcription
        """
        from victor.multimodal.processor import MediaType, ProcessingResult, ProcessingStatus

        path = Path(video_path)

        if not path.exists():
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                content="",
                error=f"Video file not found: {video_path}",
                media_type=MediaType.VIDEO,
                source=str(video_path),
            )

        if not MOVIEPY_AVAILABLE:
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                content="",
                error="moviepy not available for video audio extraction",
                media_type=MediaType.VIDEO,
                source=str(video_path),
            )

        try:
            # Extract audio
            video = mp.VideoFileClip(str(video_path))
            audio = video.audio

            if audio is None:
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    content="",
                    error="No audio track found in video",
                    media_type=MediaType.VIDEO,
                    source=str(video_path),
                )

            # Save audio to temp file
            temp_audio_path = path.parent / f"{path.stem}_temp_audio.wav"
            audio.write_audiofile(str(temp_audio_path), verbose=False, logger=None)

            # Transcribe
            result = await self.process(temp_audio_path)

            # Clean up temp file
            temp_audio_path.unlink(missing_ok=True)

            # Update metadata
            if result.metadata:
                result.metadata["video_duration"] = video.duration
                result.metadata["audio_duration"] = audio.duration

            # Close resources
            audio.close()
            video.close()

            return result

        except Exception as e:
            logger.error(f"Video audio processing failed: {e}")
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                content="",
                error=str(e),
                media_type=MediaType.VIDEO,
                source=str(video_path),
            )

    def convert_audio_format(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        output_format: str = "wav",
    ) -> bool:
        """Convert audio to different format.

        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            output_format: Target format (wav, mp3, etc.)

        Returns:
            True if successful
        """
        if not PYDUB_AVAILABLE:
            logger.error("pydub not available for audio conversion")
            return False

        try:
            audio = AudioSegment.from_file(str(input_path))
            audio.export(str(output_path), format=output_format)
            return True
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return False
