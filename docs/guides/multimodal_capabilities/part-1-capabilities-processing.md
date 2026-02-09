# Multimodal Capabilities Guide - Part 1

**Part 1 of 2:** Overview, Vision Processing, Audio Processing, Multimodal Workflows, and Provider Requirements

---

## Navigation

- **[Part 1: Capabilities & Processing](#)** (Current)
- [Part 2: Best Practices](part-2-best-practices.md)
- [**Complete Guide](../MULTIMODAL_CAPABILITIES.md)**

---
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

