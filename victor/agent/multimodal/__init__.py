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

"""Multimodal capabilities for Victor AI agent.

This module provides vision and audio processing capabilities:
- VisionAgent: Image analysis, object detection, OCR, captioning
- AudioAgent: Audio transcription, analysis, speaker diarization, summarization

Audio Features:
- Audio transcription with Whisper API and local fallback
- Speaker diarization (identify different speakers)
- Language detection from audio
- Audio quality analysis with quality metrics
- Transcription summarization using LLM
- Support for 10+ audio formats (MP3, WAV, M4A, FLAC, etc.)

Vision Features:
- Image analysis and understanding
- Object detection and classification
- OCR (text extraction from images)
- Plot and chart data extraction
- Image comparison and similarity
"""

from victor.agent.multimodal.audio_agent import (
    AudioAnalysis,
    AudioAgent,
    AudioFormat,
    AudioQualityMetrics,
    SpeakerSegment,
    SpeechSegment,  # Legacy alias
    TranscriptionResult,
)
from victor.agent.multimodal.vision_agent import (
    BoundingBox,
    ChartType,
    ColorInfo,
    ComparisonResult,
    DataPoint,
    DetectedObject,
    FaceDetection,
    ImageFormat,
    ImageMetadata,
    PlotData,
    VisionAgent,
    VisionAnalysisResult,
    # Legacy aliases
    ImageAnalysis,
    ImageComparison,
)

__all__ = [
    # Vision
    "VisionAgent",
    "VisionAnalysisResult",
    "ImageAnalysis",  # Legacy alias
    "DetectedObject",
    "PlotData",
    "ComparisonResult",
    "ImageComparison",  # Legacy alias
    "BoundingBox",
    "ColorInfo",
    "FaceDetection",
    "DataPoint",
    "ImageMetadata",
    "ChartType",
    "ImageFormat",
    # Audio
    "AudioAgent",
    "AudioFormat",
    "TranscriptionResult",
    "AudioAnalysis",
    "AudioQualityMetrics",
    "SpeakerSegment",
    "SpeechSegment",  # Legacy alias
]
