# Multimodal Capabilities Guide

## Overview

Victor AI's multimodal capabilities enable processing and understanding of visual and audio content alongside text. This guide explains how to use vision and audio processing effectively.

## Table of Contents

- [What are Multimodal Capabilities?](#what-are-multimodal-capabilities)
- [Vision Processing](#vision-processing)
- [Audio Processing](#audio-processing)
- [Multimodal Workflows](#multimodal-workflows)
- [Provider Requirements](#provider-requirements)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## What are Multimodal Capabilities?

Multimodal AI extends Victor beyond text-only interactions:

### Vision Capabilities
- **Image Analysis**: Understand and describe images
- **Chart Extraction**: Extract data from charts and graphs
- **OCR**: Extract text from images
- **Diagram Understanding**: Interpret technical diagrams
- **Screenshot Analysis**: Analyze UI screenshots

### Audio Capabilities
- **Speech Transcription**: Convert speech to text
- **Audio Analysis**: Analyze audio content
- **Meeting Transcription**: Transcribe meetings and calls
- **Voice Commands**: Process voice inputs

### When to Use Multimodal Capabilities

**Ideal for:**
- Analyzing screenshots or diagrams
- Extracting data from charts
- Transcribing audio content
- Document processing
- Accessibility features

**Not ideal for:**
- Pure text tasks (use regular chat)
- Simple data extraction (use specialized tools)
- Real-time processing (latency considerations)

## Vision Processing

Process and understand visual content.

### Basic Image Analysis

```python
from victor.agent.multimodal import VisionAgent
from victor.agent import AgentOrchestrator

orchestrator = AgentOrchestrator(...)

# Create vision agent
vision = VisionAgent(orchestrator=orchestrator)

# Analyze an image
analysis = await vision.analyze_image(
    image_path="screenshot.png",
    prompt="Describe what's shown in this screenshot"
)

print(analysis.description)
print(analysis.details)
```

### Chart Data Extraction

```python
# Extract data from charts
chart_data = await vision.extract_chart_data(
    image_path="revenue_chart.png",
    chart_type="bar"  # or "line", "pie", "scatter"
)

print(f"Chart type: {chart_data.type}")
print(f"Data points: {len(chart_data.data)}")
print(f"X-axis: {chart_data.x_axis_label}")
print(f"Y-axis: {chart_data.y_axis_label}")

for point in chart_data.data:
    print(f"  {point.x}: {point.y}")
```

### OCR - Text Extraction

```python
# Extract text from images
text = await vision.extract_text(
    image_path="document.png",
    language="auto"  # or "en", "es", "fr", etc.
)

print(f"Extracted text:\n{text}")

# Extract structured text (tables, forms)
structured = await vision.extract_structured_text(
    image_path="form.png",
    structure="table"  # or "form", "key_value"
)

for row in structured.rows:
    print(row)
```

### Diagram Understanding

```python
# Understand technical diagrams
diagram_analysis = await vision.analyze_diagram(
    image_path="architecture.png",
    diagram_type="flowchart"  # or "sequence", "architecture", "network"
)

print(f"Diagram type: {diagram_analysis.type}")
print(f"Components: {len(diagram_analysis.components)}")

for component in diagram_analysis.components:
    print(f"  - {component.name}: {component.type}")
    print(f"    Connections: {component.connections}")

# Get diagram description
description = await vision.describe_diagram(
    image_path="architecture.png",
    detail_level="detailed"  # or "brief", "comprehensive"
)

print(description)
```

### Screenshot Analysis

```python
# Analyze UI screenshots
ui_analysis = await vision.analyze_ui(
    image_path="app_screenshot.png",
    analysis_type="accessibility"  # or "layout", "usability", "comprehensive"
)

print(f"Accessibility score: {ui_analysis.accessibility_score}")
print(f"Issues found: {len(ui_analysis.issues)}")

for issue in ui_analysis.issues:
    print(f"  - {issue.severity}: {issue.description}")
    print(f"    Location: {issue.location}")
    print(f"    Suggestion: {issue.suggestion}")
```

### Batch Image Processing

```python
# Process multiple images
images = ["chart1.png", "chart2.png", "chart3.png"]

results = await vision.batch_analyze(
    image_paths=images,
    prompt="Extract key insights from these charts"
)

for image_path, result in zip(images, results):
    print(f"{image_path}: {result.summary}")
```

## Audio Processing

Process and understand audio content.

### Basic Transcription

```python
from victor.agent.multimodal import AudioAgent

# Create audio agent
audio = AudioAgent(orchestrator=orchestrator)

# Transcribe audio file
transcript = await audio.transcribe_audio(
    audio_path="meeting.mp3",
    language="en"  # or "auto" for auto-detection
)

print(f"Transcript:\n{transcript.text}")
print(f"Duration: {transcript.duration}s")
print(f"Confidence: {transcript.confidence:.2%}")
```

### Speaker Diarization

```python
# Identify different speakers
diarization = await audio.identify_speakers(
    audio_path="interview.mp3",
    num_speakers=2  # or "auto" for auto-detection
)

print(f"Speakers identified: {len(diarization.speakers)}")

for segment in diarization.segments:
    print(f"[{segment.start_time:.1f}s - {segment.end_time:.1f}s]")
    print(f"Speaker {segment.speaker_id}: {segment.text}")
```

### Meeting Transcription

```python
# Transcribe meeting with summary
meeting = await audio.transcribe_meeting(
    audio_path="team_meeting.mp3",
    include_summary=True,
    include_action_items=True
)

print(f"=== Transcript ===\n{meeting.transcript}")
print(f"\n=== Summary ===\n{meeting.summary}")
print(f"\n=== Action Items ===")
for item in meeting.action_items:
    print(f"- {item.description} (assigned to: {item.assignee})")
```

### Audio Analysis

```python
# Analyze audio content
analysis = await audio.analyze_audio(
    audio_path="podcast.mp3",
    analysis_type="sentiment"  # or "topic", "keyword", "comprehensive"
)

print(f"Sentiment: {analysis.sentiment}")
print(f"Topics: {analysis.topics}")
print(f"Keywords: {analysis.keywords}")
```

### Voice Commands

```python
# Process voice commands
command = await audio.process_command(
    audio_path="command.wav",
    command_type="general"  # or "coding", "navigation", "system"
)

print(f"Command: {command.intent}")
print(f"Parameters: {command.parameters}")

# Execute command
if command.intent == "open_file":
    await orchestrator.execute_tool("read_file", {"path": command.parameters["file"]})
```

## Multimodal Workflows

Combine vision and audio processing with text.

### Video Analysis

```python
# Analyze video content
async def analyze_video(video_path):
    # Extract audio track
    audio_path = await extract_audio(video_path)

    # Extract frames
    frames = await extract_frames(video_path, count=10)

    # Transcribe audio
    transcript = await audio.transcribe_audio(audio_path)

    # Analyze frames
    frame_descriptions = []
    for frame in frames:
        desc = await vision.analyze_image(frame, "Describe this frame")
        frame_descriptions.append(desc.description)

    # Combine analysis
    summary = await orchestrator.chat(
        f"Summarize this video based on:\n"
        f"Transcript: {transcript.text}\n"
        f"Visuals: {' '.join(frame_descriptions)}"
    )

    return {
        "transcript": transcript,
        "visual_summary": frame_descriptions,
        "overall_summary": summary
    }
```

### Document Processing

```python
# Process document with images
async def process_document(doc_path):
    # Extract text
    text = await extract_text_from_pdf(doc_path)

    # Extract images
    images = await extract_images_from_pdf(doc_path)

    # Analyze images
    image_contexts = []
    for img in images:
        analysis = await vision.analyze_image(img, "Describe this image")
        image_contexts.append(analysis.description)

    # Combine
    document_summary = await orchestrator.chat(
        f"Document text: {text}\n"
        f"Images: {' '.join(image_contexts)}\n"
        f"Provide a comprehensive summary."
    )

    return document_summary
```

### Presentation Creation

```python
# Create presentation from assets
async def create_presentation(images, audio_note):
    # Analyze images
    image_analyses = []
    for img in images:
        analysis = await vision.analyze_image(img, "Describe this image")
        image_analyses.append(analysis)

    # Transcribe audio note
    transcript = await audio.transcribe_audio(audio_note)

    # Generate slide content
    slides = []
    for i, analysis in enumerate(image_analyses):
        slide = await orchestrator.chat(
            f"Create slide {i+1} based on:\n"
            f"Image: {analysis.description}\n"
            f"Context: {transcript.text}"
        )
        slides.append(slide)

    return slides
```

## Provider Requirements

Multimodal capabilities require specific provider support.

### Vision Capabilities

| Provider | Image Analysis | Chart Extraction | OCR | Diagram Understanding |
|----------|---------------|------------------|-----|----------------------|
| Anthropic | ✅ | ✅ | ✅ | ✅ |
| OpenAI (GPT-4V) | ✅ | ✅ | ✅ | ✅ |
| Google Gemini | ✅ | ✅ | ✅ | ✅ |
| Azure OpenAI | ✅ | ✅ | ✅ | Partial |

### Audio Capabilities

| Provider | Transcription | Diarization | Analysis |
|----------|--------------|-------------|----------|
| OpenAI (Whisper) | ✅ | ❌ | ❌ |
| Google Speech-to-Text | ✅ | ✅ | ✅ |
| AWS Transcribe | ✅ | ✅ | Partial |
| Azure Speech | ✅ | ✅ | Partial |

### Configuring Providers

```python
# For vision
settings = Settings()
settings.vision_provider = "anthropic"  # or "openai", "google"
settings.vision_model = "claude-3-opus"  # or "gpt-4-vision", "gemini-pro-vision"

# For audio
settings.audio_provider = "openai"  # or "google", "aws"
settings.audio_model = "whisper-1"  # or "chirp", "transcribe-best"

orchestrator = AgentOrchestrator(settings=settings)
```

## Best Practices

### 1. Optimize Image Quality

```python
# Preprocess images for better results
from PIL import Image

def preprocess_image(image_path):
    img = Image.open(image_path)

    # Resize if too large
    if img.width > 2000:
        ratio = 2000 / img.width
        img = img.resize((2000, int(img.height * ratio)))

    # Enhance contrast
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)

    # Save
    processed_path = image_path.replace(".png", "_processed.png")
    img.save(processed_path)

    return processed_path

# Use preprocessed image
processed = preprocess_image("screenshot.png")
analysis = await vision.analyze_image(processed, "Describe this image")
```

### 2. Use Specific Prompts

```python
# Good: Specific prompt
analysis = await vision.analyze_image(
    image_path="code.png",
    prompt="Identify the programming language and describe the algorithm implemented"
)

# Bad: Vague prompt
analysis = await vision.analyze_image(
    image_path="code.png",
    prompt="What is this?"
)
```

### 3. Handle Large Audio Files

```python
# Split large audio files
async def transcribe_long_audio(audio_path, chunk_duration=300):
    # Split into chunks
    chunks = await split_audio(audio_path, chunk_duration)

    # Transcribe each chunk
    transcripts = []
    for chunk in chunks:
        transcript = await audio.transcribe_audio(chunk)
        transcripts.append(transcript)

    # Combine
    full_transcript = " ".join([t.text for t in transcripts])
    return full_transcript
```

### 4. Validate Results

```python
# Validate OCR results
text = await vision.extract_text("document.png")

# Check confidence
if text.confidence < 0.8:
    print("Low confidence, consider manual review")

# Validate with checksum if available
if "checksum" in metadata:
    computed = compute_checksum(text.text)
    if computed != metadata["checksum"]:
        print("Checksum mismatch, possible OCR error")
```

### 5. Cache Results

```python
from functools import lru_cache

@lru_cache(maxsize=100)
async def cached_analyze_image(image_path, prompt):
    # Compute hash of image
    image_hash = compute_image_hash(image_path)

    # Check cache
    cached = get_from_cache(image_hash, prompt)
    if cached:
        return cached

    # Analyze
    result = await vision.analyze_image(image_path, prompt)

    # Cache result
    save_to_cache(image_hash, prompt, result)

    return result
```

## Troubleshooting

### Vision Issues

**Problem**: Poor image analysis results.

**Solutions**:
1. **Improve image quality**: Higher resolution, better contrast
2. **Use specific prompts**: More detailed prompts
3. **Check provider**: Ensure provider supports vision
4. **Preprocess images**: Resize, enhance, crop

```python
# Improve results
processed = preprocess_image(image_path)
analysis = await vision.analyze_image(
    processed,
    "Provide detailed analysis focusing on [specific aspects]"
)
```

### Audio Issues

**Problem**: Poor transcription quality.

**Solutions**:
1. **Improve audio quality**: Remove background noise
2. **Specify language**: Use correct language code
3. **Split long files**: Process in chunks
4. **Check provider**: Ensure provider supports audio

```python
# Improve transcription
clean_audio = remove_noise(audio_path)
transcript = await audio.transcribe_audio(
    clean_audio,
    language="en",
    quality="high"  # If provider supports quality setting
)
```

### Performance Issues

**Problem**: Slow processing.

**Solutions**:
1. **Compress images**: Reduce file size
2. **Process in batches**: Batch multiple operations
3. **Use caching**: Cache frequent analyses
4. **Parallel processing**: Process multiple files in parallel

```python
# Batch processing
results = await vision.batch_analyze(
    image_paths=images,
    prompt="Analyze these images",
    max_parallel=5
)
```

## Examples

### Example 1: Analyzing Architecture Diagrams

```python
async def analyze_architecture(diagram_path):
    # Understand diagram
    diagram = await vision.analyze_diagram(
        diagram_path,
        diagram_type="architecture"
    )

    # Extract components
    components = {}
    for component in diagram.components:
        components[component.name] = {
            "type": component.type,
            "connections": component.connections,
            "description": component.description
        }

    # Generate documentation
    docs = await orchestrator.chat(
        f"Generate architecture documentation based on:\n"
        f"Components: {components}\n"
        f"Flow: {diagram.flow_description}"
    )

    return {
        "components": components,
        "documentation": docs
    }
```

### Example 2: Meeting Intelligence

```python
async def meeting_intelligence(audio_path):
    # Transcribe meeting
    meeting = await audio.transcribe_meeting(
        audio_path,
        include_summary=True,
        include_action_items=True
    )

    # Extract topics
    topics = await audio.analyze_audio(
        audio_path,
        analysis_type="topic"
    )

    # Generate report
    report = await orchestrator.chat(
        f"Generate meeting report:\n"
        f"Summary: {meeting.summary}\n"
        f"Topics: {topics.topics}\n"
        f"Action Items: {meeting.action_items}"
    )

    return report
```

### Example 3: Chart Analysis

```python
async def analyze_chart(chart_path):
    # Extract data
    data = await vision.extract_chart_data(chart_path)

    # Get description
    description = await vision.analyze_image(
        chart_path,
        "Describe the trends and patterns in this chart"
    )

    # Generate insights
    insights = await orchestrator.chat(
        f"Provide insights based on:\n"
        f"Data: {data.data}\n"
        f"Description: {description}"
    )

    return {
        "data": data.data,
        "description": description,
        "insights": insights
    }
```

### Example 4: Document Digitization

```python
async def digitize_document(image_path):
    # Extract text with OCR
    text = await vision.extract_text(image_path)

    # Extract structure
    structure = await vision.extract_structured_text(
        image_path,
        structure="auto"
    )

    # Validate extraction
    confidence = text.confidence
    if confidence < 0.8:
        print(f"Warning: Low confidence ({confidence:.1%})")

    # Generate digital version
    digital_doc = await orchestrator.chat(
        f"Create a well-formatted digital document from:\n"
        f"Text: {text.text}\n"
        f"Structure: {structure}"
    )

    return digital_doc
```

## Additional Resources

- [API Reference](../api/NEW_CAPABILITIES_API.md)
- [User Guide](../USER_GUIDE.md)
- [Provider Documentation](../providers/README.md)
- [Vision Agent API](../api/multimodal/vision.md)
- [Audio Agent API](../api/multimodal/audio.md)
