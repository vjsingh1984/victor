# Multimodal Assistant

A sophisticated demonstration of Victor AI's multimodal capabilities, featuring image analysis, audio processing, OCR (Optical Character Recognition), and cross-modal understanding.

## Features

- **Image Analysis**: Understand and describe images, objects, scenes
- **Audio Processing**: Transcribe speech, analyze audio content
- **OCR**: Extract text from images and scanned documents
- **Cross-Modal Queries**: Ask questions about images and audio
- **Video Analysis**: Process video frames and extract insights
- **Document Processing**: Convert scanned docs to searchable text
- **Interactive Web UI**: Easy-to-use interface for all capabilities

## Screenshots

The web interface provides:
- Image upload and analysis
- Audio recording and transcription
- OCR text extraction
- Cross-modal query interface
- Result visualization

## Installation

```bash
# Navigate to demo directory
cd examples/multimodal_assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install system dependencies for audio/video
# On Ubuntu/Debian:
sudo apt-get install ffmpeg libsndfile1

# On macOS:
brew install ffmpeg libsndfile
```

## Usage

### Web Interface

Start the web server:

```bash
python app.py
```

Open browser to `http://localhost:5000`

### CLI Usage

Analyze an image:

```bash
python main.py analyze-image path/to/image.jpg
```

With specific questions:

```bash
python main.py analyze-image photo.jpg --question "What objects are in this image?"
```

Transcribe audio:

```bash
python main.py transcribe audio.wav
```

Extract text from image (OCR):

```bash
python main.py ocr document.png --output extracted_text.txt
```

Process video:

```bash
python main.py analyze-video video.mp4 --frames 10
```

Cross-modal query:

```bash
python main.py query --image photo.jpg --audio voice.wav \
    "What is the relationship between what you see and hear?"
```

### Python API

```python
from multimodal_assistant import MultimodalAssistant

# Initialize assistant
assistant = MultimodalAssistant()

# Analyze image
image_result = assistant.analyze_image(
    image_path="photo.jpg",
    question="Describe what you see"
)

# Transcribe audio
audio_result = assistant.transcribe_audio(
    audio_path="interview.wav"
)

# OCR
text = assistant.extract_text(
    image_path="scanned_doc.png"
)

# Cross-modal analysis
result = assistant.cross_modal_query(
    image="chart.png",
    question="What data is shown in this chart?"
)
```

## Capabilities in Detail

### 1. Image Analysis

```python
from victor.tools.vision import ImageAnalyzer

analyzer = ImageAnalyzer()

# Describe image
description = analyzer.describe("photo.jpg")
# Output: "A sunny beach with palm trees and people swimming"

# Detect objects
objects = analyzer.detect_objects("photo.jpg")
# Output: ["person", "palm tree", "beach umbrella", "ocean"]

# Answer questions
answer = analyzer.query("photo.jpg", "What color are the umbrellas?")
# Output: "The umbrellas are blue and yellow"
```

### 2. Audio Processing

```python
from victor.tools.audio import AudioProcessor

processor = AudioProcessor()

# Transcribe speech
transcript = processor.transcribe("speech.wav")
# Output: "Hello, this is a test recording."

# Detect language
language = processor.detect_language("audio.mp3")
# Output: "en"

# Analyze sentiment
sentiment = processor.sentiment("conversation.wav")
# Output: {"positive": 0.8, "negative": 0.1, "neutral": 0.1}
```

### 3. OCR (Text Extraction)

```python
from victor.tools.vision import OCRProcessor

ocr = OCRProcessor()

# Extract text
text = ocr.extract("document.png")
# Output: "INVOICE #1234\nDate: 2024-01-15\n..."

# Detect text regions
regions = ocr.detect_regions("page.jpg")
# Output: [{"bbox": [10, 20, 100, 50], "text": "Header"}, ...]

# Structured extraction
invoice = ocr.extract_structured("invoice.png", template="invoice")
# Output: {"invoice_number": "1234", "date": "2024-01-15", ...}
```

### 4. Video Analysis

```python
from victor.tools.vision import VideoAnalyzer

analyzer = VideoAnalyzer()

# Extract key frames
frames = analyzer.extract_frames("video.mp4", count=10)

# Analyze video content
summary = analyzer.summarize("video.mp4")
# Output: "Video shows a person cooking in a kitchen"

# Detect scenes
scenes = analyzer.detect_scenes("movie.mp4")
# Output: [{"start": 0, "end": 120, "description": "Opening scene"}, ...]
```

### 5. Cross-Modal Queries

```python
from victor.multimodal import CrossModalEngine

engine = CrossModalEngine()

# Question about image
result = engine.query(
    image="chart.png",
    question="What does this graph tell us about sales trends?"
)

# Question combining image and text
result = engine.query(
    image="xray.jpg",
    context="Patient is a 45-year-old male",
    question="What abnormalities do you see?"
)
```

## Use Cases

### 1. Document Digitization

Convert scanned documents to searchable text:

```bash
python main.py ocr --input scanned_docs/ --output digital_text/
```

Features:
- Batch processing
- Multiple language support
- Preserve layout information
- Export to searchable PDF

### 2. Image Captioning

Generate captions for images:

```bash
python main.py caption-image photos/ --output captions.json
```

Features:
- Automatic alt text generation
- SEO-friendly descriptions
- Multi-language captions

### 3. Meeting Transcription

Transcribe audio recordings:

```bash
python main.py transcribe meeting.mp3 \
    --speakers 2 \
    --output transcript.txt
```

Features:
- Speaker diarization
- Timestamp generation
- Punctuation and formatting

### 4. Content Moderation

Analyze images and text for policy compliance:

```bash
python main.py moderate --image user_upload.jpg
```

Features:
- Detect inappropriate content
- Check for watermarks
- Analyze text in images

### 5. Medical Imaging

Analyze medical images (research only):

```bash
python main.py analyze-medical xray.png \
    --question "What potential issues do you observe?"
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              User Interface                          │
│          (Web, CLI, Python API)                     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│          Multimodal Orchestrator                     │
│  - Routes requests to appropriate processors         │
│  - Combines multi-modal results                      │
│  - Manages cross-modal queries                       │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│    Vision   │ │    Audio    │ │   Cross-    │
│  Processor  │ │  Processor  │ │   Modal     │
└─────────────┘ └─────────────┘ └─────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              ┌─────────────────┐
              │  Victor AI      │
              │  Orchestrator   │
              └─────────────────┘
```

## Configuration

Create a `multimodal_config.yaml`:

```yaml
# Vision configuration
vision:
  model: gpt-4-vision-preview
  max_image_size: 20MB
  supported_formats: [jpg, png, gif, webp]

# Audio configuration
audio:
  transcription_model: whisper-large
  sample_rate: 16000
  supported_formats: [wav, mp3, m4a, flac]

# OCR configuration
ocr:
  engine: tesseract
  languages: [en, es, fr, de]
  preprocessing: true

# Cross-modal configuration
cross_modal:
  context_window: 5
  fusion_strategy: attention
```

## Integration with Victor AI

This demo showcases:

### Vision Tools
```python
from victor.tools.vision import (
    ImageAnalyzer,
    OCRProcessor,
    VideoAnalyzer
)
```

### Audio Tools
```python
from victor.tools.audio import (
    AudioTranscriber,
    AudioAnalyzer
)
```

### Multimodal Processing
```python
from victor.multimodal import (
    CrossModalEngine,
    MultiModalOrchestrator
)
```

### Provider Support

Works with providers that support multimodal inputs:
- Anthropic (Claude with vision)
- OpenAI (GPT-4V)
- Google (Gemini Pro Vision)

## Performance Tips

- **Images**: Resize large images before upload for faster processing
- **Audio**: Use compressed formats (MP3, AAC) for faster uploads
- **OCR**: Pre-process images (deskew, denoise) for better accuracy
- **Video**: Extract key frames instead of processing every frame

## Testing

```bash
# Run tests
pytest tests/

# Test specific modality
pytest tests/test_vision.py -v

# Integration test
pytest tests/integration/test_multimodal.py
```

## Sample Data

The `sample_data/` directory includes:
- `sample_images/` - Example images for analysis
- `sample_audio/` - Sample audio files
- `sample_documents/` - Scanned documents for OCR

## Contributing

This is a demo for Victor AI. Contributions welcome!

## License

MIT License

## Support

- **Documentation**: https://victor-ai.readthedocs.io
- **Issues**: https://github.com/your-org/victor-ai/issues
