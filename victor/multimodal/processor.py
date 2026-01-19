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

"""Multi-Modal Content Processor.

Unified processor for handling multiple content types:
- Images (PNG, JPG, etc.)
- Documents (PDF, DOCX, PPTX, etc.)
- Audio/Video (MP3, MP4, WAV, etc.)
- Multi-modal prompt construction

Design Principles:
- Protocol-based architecture for extensibility
- Graceful degradation when dependencies unavailable
- Async/await for non-blocking I/O
- Comprehensive error handling
"""

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from victor.multimodal.audio_processor import AudioProcessor
    from victor.multimodal.document_processor import DocumentProcessor
    from victor.multimodal.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class MediaType(str, Enum):
    """Supported media types."""

    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


class ProcessingStatus(str, Enum):
    """Processing status codes."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


@dataclass
class ProcessingResult:
    """Result from multi-modal processing.

    Attributes:
        status: Processing status
        content: Processed content (text, transcription, etc.)
        metadata: Additional metadata (dimensions, duration, etc.)
        confidence: Confidence score (0-1)
        error: Error message if failed
        media_type: Type of media processed
        source: Source file or URL
    """

    status: ProcessingStatus
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    error: Optional[str] = None
    media_type: Optional[MediaType] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "content": self.content,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "error": self.error,
            "media_type": self.media_type.value if self.media_type else None,
            "source": self.source,
        }


@dataclass
class ProcessingConfig:
    """Configuration for multi-modal processing.

    Attributes:
        max_image_size: Maximum image dimension (width/height)
        max_audio_duration: Maximum audio duration in seconds
        max_document_size: Maximum document size in MB
        enable_ocr: Enable OCR for images
        enable_transcription: Enable audio transcription
        preferred_language: Preferred language for transcription
        timeout: Processing timeout in seconds
    """

    max_image_size: int = 4096
    max_audio_duration: int = 600  # 10 minutes
    max_document_size: int = 50  # 50 MB
    enable_ocr: bool = True
    enable_transcription: bool = True
    preferred_language: str = "en"
    timeout: int = 300


class MultiModalProcessor:
    """Unified multi-modal content processor.

    This class provides a single interface for processing multiple types
    of media content. It delegates to specialized processors while providing
    a consistent API.

    Example:
        processor = MultiModalProcessor()

        # Process image
        result = await processor.process_image(
            image_path="screenshot.png",
            query="What does this show?"
        )

        # Process document
        result = await processor.process_document(
            document_path="report.pdf",
            query="Summarize key points"
        )

        # Process audio
        result = await processor.process_audio(
            audio_path="meeting.mp3",
            query="Extract action items"
        )
    """

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
    ):
        """Initialize multi-modal processor.

        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        self._image_processor: Optional["ImageProcessor"] = None
        self._document_processor: Optional["DocumentProcessor"] = None
        self._audio_processor: Optional["AudioProcessor"] = None

    @property
    def image_processor(self) -> "ImageProcessor":
        """Get or create image processor."""
        if self._image_processor is None:
            from victor.multimodal.image_processor import ImageProcessor

            self._image_processor = ImageProcessor(config=self.config)
        return self._image_processor

    @property
    def document_processor(self) -> "DocumentProcessor":
        """Get or create document processor."""
        if self._document_processor is None:
            from victor.multimodal.document_processor import DocumentProcessor

            self._document_processor = DocumentProcessor(config=self.config)
        return self._document_processor

    @property
    def audio_processor(self) -> "AudioProcessor":
        """Get or create audio processor."""
        if self._audio_processor is None:
            from victor.multimodal.audio_processor import AudioProcessor

            self._audio_processor = AudioProcessor(config=self.config)
        return self._audio_processor

    async def process_image(
        self,
        image_path: Union[str, Path],
        query: Optional[str] = None,
        encode_base64: bool = False,
    ) -> ProcessingResult:
        """Process image content.

        Args:
            image_path: Path to image file
            query: Optional query for image understanding
            encode_base64: Whether to encode as base64

        Returns:
            ProcessingResult with extracted information
        """
        try:
            return await self.image_processor.process(
                image_path=image_path,
                query=query,
                encode_base64=encode_base64,
            )
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                content="",
                error=str(e),
                media_type=MediaType.IMAGE,
                source=str(image_path),
            )

    async def process_document(
        self,
        document_path: Union[str, Path],
        query: Optional[str] = None,
        extract_images: bool = False,
    ) -> ProcessingResult:
        """Process document content.

        Args:
            document_path: Path to document file
            query: Optional query for document understanding
            extract_images: Whether to extract images from document

        Returns:
            ProcessingResult with extracted text
        """
        try:
            return await self.document_processor.process(
                document_path=document_path,
                query=query,
                extract_images=extract_images,
            )
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                content="",
                error=str(e),
                media_type=MediaType.DOCUMENT,
                source=str(document_path),
            )

    async def process_audio(
        self,
        audio_path: Union[str, Path],
        query: Optional[str] = None,
        diarize: bool = False,
    ) -> ProcessingResult:
        """Process audio content.

        Args:
            audio_path: Path to audio file
            query: Optional query for audio understanding
            diarize: Whether to perform speaker diarization

        Returns:
            ProcessingResult with transcription
        """
        try:
            return await self.audio_processor.process(
                audio_path=audio_path,
                query=query,
                diarize=diarize,
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

    async def process_video(
        self,
        video_path: Union[str, Path],
        query: Optional[str] = None,
        extract_frames: int = 5,
    ) -> ProcessingResult:
        """Process video content.

        Args:
            video_path: Path to video file
            query: Optional query for video understanding
            extract_frames: Number of frames to extract

        Returns:
            ProcessingResult with transcription and frame analysis
        """
        try:
            # Extract audio and transcribe
            audio_result = await self.audio_processor.process_video_audio(
                video_path=video_path,
            )

            if audio_result.status != ProcessingStatus.SUCCESS:
                return audio_result

            # Extract and analyze frames
            frames_result = await self.image_processor.extract_video_frames(
                video_path=video_path,
                num_frames=extract_frames,
            )

            # Combine results
            combined_content = f"Transcription:\n{audio_result.content}\n\n"

            if frames_result.status == ProcessingStatus.SUCCESS:
                combined_content += f"Visual Analysis:\n{frames_result.content}"

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                content=combined_content,
                metadata={
                    **audio_result.metadata,
                    **frames_result.metadata,
                },
                confidence=(audio_result.confidence + frames_result.confidence) / 2,
                media_type=MediaType.VIDEO,
                source=str(video_path),
            )

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                content="",
                error=str(e),
                media_type=MediaType.VIDEO,
                source=str(video_path),
            )

    async def batch_process(
        self,
        items: List[Dict[str, Any]],
        max_concurrency: int = 5,
    ) -> List[ProcessingResult]:
        """Process multiple items concurrently.

        Args:
            items: List of processing specs with 'path', 'type', 'query'
            max_concurrency: Maximum concurrent operations

        Returns:
            List of ProcessingResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_with_semaphore(item: Dict[str, Any]) -> ProcessingResult:
            async with semaphore:
                media_type = item.get("type", "").lower()
                path = item["path"]
                query = item.get("query")

                if media_type == "image":
                    return await self.process_image(path, query)
                elif media_type == "document":
                    return await self.process_document(path, query)
                elif media_type == "audio":
                    return await self.process_audio(path, query)
                elif media_type == "video":
                    return await self.process_video(path, query)
                else:
                    return ProcessingResult(
                        status=ProcessingStatus.UNSUPPORTED,
                        content="",
                        error=f"Unsupported media type: {media_type}",
                        source=str(path),
                    )

        tasks = [process_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def get_supported_formats(self) -> Dict[MediaType, List[str]]:
        """Get supported file formats by media type.

        Returns:
            Dict mapping media types to supported extensions
        """
        return {
            MediaType.IMAGE: self.image_processor.supported_formats,
            MediaType.DOCUMENT: self.document_processor.supported_formats,
            MediaType.AUDIO: self.audio_processor.supported_formats,
            MediaType.VIDEO: self.audio_processor.supported_video_formats,
        }

    def is_supported(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is supported.

        Args:
            file_path: Path to file

        Returns:
            True if format is supported
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        for formats in self.get_supported_formats().values():
            if ext in formats:
                return True

        return False
