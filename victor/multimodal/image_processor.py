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

"""Image processing capabilities for multi-modal AI.

Supports:
- Image analysis with vision models
- OCR text extraction
- Image metadata extraction
- Base64 encoding for API calls
- Video frame extraction
"""

import base64
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from victor.multimodal.processor import ProcessingResult

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    from PIL import Image, ImageOps

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available. Install with: pip install Pillow")

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available. Install with: pip install opencv-python")

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available. Install with: pip install pytesseract")


class ImageProcessor:
    """Process and analyze images.

    Example:
        processor = ImageProcessor()

        # Analyze image
        result = await processor.process(
            image_path="screenshot.png",
            query="What does this image show?"
        )

        # Extract text with OCR
        result = await processor.extract_text("document.png")

        # Encode for API
        base64_data = processor.encode_base64("image.png")
    """

    supported_formats = [
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".webp",
        ".tiff",
    ]

    def __init__(self, config: Optional[Any] = None):
        """Initialize image processor.

        Args:
            config: ProcessingConfig instance
        """
        self.config = config

    async def process(
        self,
        image_path: str | Path,
        query: Optional[str] = None,
        encode_base64: bool = False,
        extract_ocr: bool = False,
    ) -> "ProcessingResult":
        """Process image and extract information.

        Args:
            image_path: Path to image file
            query: Optional query for image understanding
            encode_base64: Whether to return base64 encoded data
            extract_ocr: Whether to extract text with OCR

        Returns:
            ProcessingResult with image information
        """
        from victor.multimodal.processor import MediaType, ProcessingResult, ProcessingStatus

        path = Path(image_path)

        if not path.exists():
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                content="",
                error=f"Image file not found: {image_path}",
                media_type=MediaType.IMAGE,
                source=str(image_path),
            )

        try:
            # Load image
            if not PIL_AVAILABLE:
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    content="",
                    error="PIL/Pillow not available",
                    media_type=MediaType.IMAGE,
                    source=str(image_path),
                )

            image = Image.open(path)

            # Extract metadata
            metadata = self._extract_metadata(image, path)

            content_parts = []

            # OCR if requested
            if extract_ocr and TESSERACT_AVAILABLE:
                ocr_text = await self._extract_ocr_text(image)
                if ocr_text:
                    content_parts.append(f"Extracted Text:\n{ocr_text}")

            # Basic image description
            description = self._describe_image(image)
            content_parts.append(f"Image Description:\n{description}")

            # Encode if requested
            if encode_base64:
                base64_data = self.encode_base64(path)
                metadata["base64"] = base64_data

            content = "\n\n".join(content_parts)

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                content=content,
                metadata=metadata,
                confidence=0.9,
                media_type=MediaType.IMAGE,
                source=str(image_path),
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

    def _extract_metadata(self, image: "Image.Image", path: Path) -> dict[str, Any]:
        """Extract image metadata."""
        metadata: dict[str, Any] = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height,
            "filename": path.name,
            "filesize": path.stat().st_size,
        }

        # Extract EXIF data if available
        if hasattr(image, "info") and isinstance(image.info, dict):
            metadata["exif"] = image.info

        return metadata

    def _describe_image(self, image: "Image.Image") -> str:
        """Generate basic image description."""
        width, height = image.size
        mode = image.mode

        description = f"Image dimensions: {width}x{height}, Color mode: {mode}"

        # Detect if image is mostly dark/light
        if PIL_AVAILABLE:
            grayscale = image.convert("L")
            pixels = list(grayscale.getdata())
            avg_brightness = sum(pixels) / len(pixels)

            if avg_brightness < 85:
                description += ", predominantly dark"
            elif avg_brightness > 170:
                description += ", predominantly light"
            else:
                description += ", balanced brightness"

        return description

    async def _extract_ocr_text(self, image: "Image.Image") -> str:
        """Extract text from image using OCR."""
        if not TESSERACT_AVAILABLE:
            return ""

        try:
            # Convert to grayscale and apply threshold for better OCR
            image_gray = image.convert("L")

            # Use pytesseract
            text = pytesseract.image_to_string(image_gray)

            return str(text).strip()

        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return ""

    def encode_base64(self, image_path: str | Path) -> str:
        """Encode image as base64 string.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded string
        """
        path = Path(image_path)

        with open(path, "rb") as f:
            image_data = f.read()
            return base64.b64encode(image_data).decode("utf-8")

    def encode_base64_from_pil(self, image: "Image.Image", format: str = "PNG") -> str:
        """Encode PIL image as base64.

        Args:
            image: PIL Image object
            format: Image format (PNG, JPEG, etc.)

        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode("utf-8")

    async def extract_video_frames(
        self,
        video_path: str | Path,
        num_frames: int = 5,
    ) -> "ProcessingResult":
        """Extract frames from video for analysis.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract

        Returns:
            ProcessingResult with frame information
        """
        from victor.multimodal.processor import MediaType, ProcessingResult, ProcessingStatus

        if not OPENCV_AVAILABLE:
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                content="",
                error="OpenCV not available",
                media_type=MediaType.VIDEO,
                source=str(video_path),
            )

        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    content="",
                    error="Could not open video file",
                    media_type=MediaType.VIDEO,
                    source=str(video_path),
                )

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            # Extract frames at regular intervals
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

            frames_info = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if ret:
                    # Get frame properties
                    height, width = frame.shape[:2]
                    frames_info.append(
                        {
                            "frame_number": idx,
                            "timestamp": idx / fps if fps > 0 else 0,
                            "resolution": (width, height),
                        }
                    )

            cap.release()

            content = f"Extracted {len(frames_info)} frames from {duration:.2f}s video\n"
            content += f"Resolution: {frames_info[0]['resolution'] if frames_info else 'N/A'}\n"
            content += f"FPS: {fps:.2f}\n"

            for i, frame_info in enumerate(frames_info):
                content += f"\nFrame {i + 1}: {frame_info['timestamp']:.2f}s"

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                content=content,
                metadata={
                    "total_frames": total_frames,
                    "fps": fps,
                    "duration": duration,
                    "extracted_frames": frames_info,
                },
                confidence=0.95,
                media_type=MediaType.VIDEO,
                source=str(video_path),
            )

        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                content="",
                error=str(e),
                media_type=MediaType.VIDEO,
                source=str(video_path),
            )

    def resize_image(
        self,
        image_path: str | Path,
        max_size: int = 1024,
        output_path: Optional[str | Path] = None,
    ) -> Optional[str]:
        """Resize image to max dimension while maintaining aspect ratio.

        Args:
            image_path: Path to input image
            max_size: Maximum dimension (width or height)
            output_path: Optional output path (default: overwrite input)

        Returns:
            Path to resized image or None if failed
        """
        if not PIL_AVAILABLE:
            logger.error("PIL not available for image resizing")
            return None

        try:
            img: Any = Image.open(image_path)

            # Calculate new size
            width, height = img.size
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save
            output = output_path or image_path
            img.save(output, optimize=True, quality=95)

            return str(output)

        except Exception as e:
            logger.error(f"Image resize failed: {e}")
            return None
