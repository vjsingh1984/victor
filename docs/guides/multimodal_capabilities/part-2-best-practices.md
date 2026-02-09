# Multimodal Capabilities Guide - Part 2

**Part 2 of 2:** Best Practices, Troubleshooting, Examples, and Additional Resources

---

## Navigation

- [Part 1: Capabilities & Processing](part-1-capabilities-processing.md)
- **[Part 2: Best Practices](#)** (Current)
- [**Complete Guide](../MULTIMODAL_CAPABILITIES.md)**

---
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
- [User Guide](../user-guide/index.md)
- [Provider Documentation](../reference/providers/index.md)
- [Vision Agent API](../reference/api/index.md)
- [Audio Agent API](../reference/api/index.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
