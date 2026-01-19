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

"""Multi-Modal processing capabilities for Victor AI.

This package provides comprehensive multi-modal content processing:
- Image understanding and analysis
- Document processing (PDF, Office, etc.)
- Audio/video transcription
- Multi-modal prompt building

Example:
    from victor.multimodal import MultiModalProcessor

    processor = MultiModalProcessor()

    # Process image
    result = await processor.process_image(
        image_path="screenshot.png",
        query="What does this image show?"
    )

    # Process document
    result = await processor.process_document(
        document_path="report.pdf",
        query="Summarize the key findings"
    )

    # Process audio
    result = await processor.process_audio(
        audio_path="meeting.mp3",
        query="Extract action items"
    )
"""

from victor.multimodal.processor import (
    MultiModalProcessor,
    ProcessingResult,
    ProcessingConfig,
    MediaType,
    ProcessingStatus,
)
from victor.multimodal.image_processor import ImageProcessor
from victor.multimodal.document_processor import DocumentProcessor
from victor.multimodal.audio_processor import AudioProcessor
from victor.multimodal.prompt_builder import MultiModalPromptBuilder

__all__ = [
    "MultiModalProcessor",
    "ProcessingResult",
    "ProcessingConfig",
    "MediaType",
    "ProcessingStatus",
    "ImageProcessor",
    "DocumentProcessor",
    "AudioProcessor",
    "MultiModalPromptBuilder",
]
