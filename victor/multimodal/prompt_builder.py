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

"""Multi-modal prompt builder for constructing rich prompts.

This module provides utilities for building prompts that include multiple
types of content (text, images, audio, documents) in a format suitable
for multi-modal LLMs.
"""

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MediaContent:
    """Content item for multi-modal prompt.

    Attributes:
        type: Content type (text, image, audio, document)
        content: Text content or base64 encoded data
        path: Optional file path
        metadata: Additional metadata
    """

    type: str
    content: str
    path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuiltPrompt:
    """Built multi-modal prompt.

    Attributes:
        text: Main prompt text
        media: List of media content items
        metadata: Additional metadata
    """

    text: str
    media: list[MediaContent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "media": [
                {
                    "type": m.type,
                    "content": m.content,
                    "path": m.path,
                    "metadata": m.metadata,
                }
                for m in self.media
            ],
            "metadata": self.metadata,
        }


class MultiModalPromptBuilder:
    """Build prompts with multi-modal content.

    Example:
        builder = MultiModalPromptBuilder()

        # Add text
        builder.add_text("Analyze this image:")

        # Add image
        builder.add_image("screenshot.png")

        # Add document
        builder.add_document("report.pdf", query="Summarize key points")

        # Build prompt
        prompt = builder.build()

        # Use with LLM
        response = await llm.generate(
            prompt.text,
            images=prompt.media,
        )
    """

    def __init__(self, max_text_length: int = 10000):
        """Initialize prompt builder.

        Args:
            max_text_length: Maximum total text length
        """
        self.max_text_length = max_text_length
        self._text_parts: list[str] = []
        self._media_items: list[MediaContent] = []
        self._metadata: dict[str, Any] = {}

    def add_text(self, text: str, truncate: bool = True) -> "MultiModalPromptBuilder":
        """Add text content.

        Args:
            text: Text to add
            truncate: Whether to truncate if too long

        Returns:
            Self for chaining
        """
        if truncate and len(text) > self.max_text_length:
            text = text[: self.max_text_length - 3] + "..."

        self._text_parts.append(text)
        return self

    def add_image(
        self,
        image_path: str | Path,
        encode_base64: bool = True,
        description: Optional[str] = None,
    ) -> "MultiModalPromptBuilder":
        """Add image content.

        Args:
            image_path: Path to image file
            encode_base64: Whether to encode as base64
            description: Optional image description

        Returns:
            Self for chaining
        """
        path = Path(image_path)

        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return self

        content = ""
        if encode_base64:
            with open(path, "rb") as f:
                content = base64.b64encode(f.read()).decode("utf-8")

        self._media_items.append(
            MediaContent(
                type="image",
                content=content,
                path=str(path),
                metadata={"description": description} if description else {},
            )
        )

        if description:
            self._text_parts.append(f"[Image: {description}]")

        return self

    def add_document(
        self,
        document_path: str | Path,
        query: Optional[str] = None,
        max_length: int = 2000,
    ) -> "MultiModalPromptBuilder":
        """Add document content.

        Args:
            document_path: Path to document file
            query: Optional query for document understanding
            max_length: Maximum document length to include

        Returns:
            Self for chaining
        """
        path = Path(document_path)

        if not path.exists():
            logger.warning(f"Document file not found: {document_path}")
            return self

        # Read document text
        try:
            if path.suffix.lower() in [".txt", ".md", ".markdown"]:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

                # Truncate if needed
                if len(text) > max_length:
                    text = text[:max_length] + "\n\n[Document truncated...]"

                self._text_parts.append(f"\n[Document: {path.name}]\n{text}\n")

                self._media_items.append(
                    MediaContent(
                        type="document",
                        content=text[:max_length],
                        path=str(path),
                        metadata={"query": query} if query else {},
                    )
                )

            else:
                # For binary formats, just add reference
                self._text_parts.append(f"\n[Document: {path.name}]")

                self._media_items.append(
                    MediaContent(
                        type="document",
                        content="",
                        path=str(path),
                        metadata={"query": query} if query else {},
                    )
                )

        except Exception as e:
            logger.error(f"Failed to add document: {e}")

        return self

    def add_audio(
        self,
        audio_path: str | Path,
        transcription: Optional[str] = None,
    ) -> "MultiModalPromptBuilder":
        """Add audio content.

        Args:
            audio_path: Path to audio file
            transcription: Optional transcription text

        Returns:
            Self for chaining
        """
        path = Path(audio_path)

        if not path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            return self

        self._media_items.append(
            MediaContent(
                type="audio",
                content="",
                path=str(path),
                metadata={"transcription": transcription} if transcription else {},
            )
        )

        if transcription:
            self._text_parts.append(f"\n[Audio transcription:]\n{transcription}\n")
        else:
            self._text_parts.append(f"\n[Audio: {path.name}]\n")

        return self

    def add_video(
        self,
        video_path: str | Path,
        transcription: Optional[str] = None,
        num_frames: int = 3,
    ) -> "MultiModalPromptBuilder":
        """Add video content.

        Args:
            video_path: Path to video file
            transcription: Optional transcription text
            num_frames: Number of frames to extract

        Returns:
            Self for chaining
        """
        path = Path(video_path)

        if not path.exists():
            logger.warning(f"Video file not found: {video_path}")
            return self

        self._media_items.append(
            MediaContent(
                type="video",
                content="",
                path=str(path),
                metadata=(
                    {
                        "transcription": transcription,
                        "num_frames": num_frames,
                    }
                    if transcription or num_frames
                    else {}
                ),
            )
        )

        if transcription:
            self._text_parts.append(f"\n[Video transcription:]\n{transcription}\n")
        else:
            self._text_parts.append(f"\n[Video: {path.name}]\n")

        return self

    def add_metadata(self, key: str, value: Any) -> "MultiModalPromptBuilder":
        """Add metadata.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for chaining
        """
        self._metadata[key] = value
        return self

    def build(self) -> BuiltPrompt:
        """Build the final prompt.

        Returns:
            BuiltPrompt with text and media
        """
        text = "\n".join(self._text_parts)

        return BuiltPrompt(
            text=text,
            media=self._media_items.copy(),
            metadata=self._metadata.copy(),
        )

    def clear(self) -> None:
        """Clear all content."""
        self._text_parts.clear()
        self._media_items.clear()
        self._metadata.clear()

    def get_text_length(self) -> int:
        """Get current text length."""
        return sum(len(part) for part in self._text_parts)

    def get_media_count(self) -> int:
        """Get number of media items."""
        return len(self._media_items)


def create_vision_prompt(
    query: str,
    image_path: str | Path,
    context: Optional[str] = None,
) -> BuiltPrompt:
    """Create a vision prompt for image analysis.

    Args:
        query: Query about the image
        image_path: Path to image
        context: Optional additional context

    Returns:
        BuiltPrompt
    """
    builder = MultiModalPromptBuilder()

    if context:
        builder.add_text(f"Context: {context}\n")

    builder.add_text(query)
    builder.add_image(image_path)

    return builder.build()


def create_document_prompt(
    query: str,
    document_path: str | Path,
    excerpt_length: int = 2000,
) -> BuiltPrompt:
    """Create a prompt for document analysis.

    Args:
        query: Query about the document
        document_path: Path to document
        excerpt_length: Maximum excerpt length

    Returns:
        BuiltPrompt
    """
    builder = MultiModalPromptBuilder()

    builder.add_text(f"Query: {query}")
    builder.add_document(document_path, max_length=excerpt_length)

    return builder.build()


def create_multimodal_prompt(
    query: str,
    items: list[dict[str, Any]],
) -> BuiltPrompt:
    """Create a prompt with multiple media types.

    Args:
        query: Main query
        items: List of items with 'path' and 'type' keys

    Returns:
        BuiltPrompt
    """
    builder = MultiModalPromptBuilder()

    builder.add_text(f"Query: {query}\n")

    for item in items:
        item_type = item.get("type", "").lower()
        path = item["path"]

        if item_type == "image":
            builder.add_image(path, description=item.get("description"))
        elif item_type == "document":
            builder.add_document(path, query=item.get("query"))
        elif item_type == "audio":
            builder.add_audio(path, transcription=item.get("transcription"))
        elif item_type == "video":
            builder.add_video(
                path,
                transcription=item.get("transcription"),
                num_frames=item.get("num_frames", 3),
            )

    return builder.build()
