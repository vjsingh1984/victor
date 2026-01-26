#!/usr/bin/env python3
"""
Victor AI Multimodal Assistant - CLI Interface

Command-line interface for multimodal AI capabilities.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from victor.agent.orchestrator_factory import create_orchestrator
from victor.config.settings import Settings

from src.vision_processor import VisionProcessor
from src.audio_processor import AudioProcessor
from src.multimodal_engine import MultimodalEngine


console = Console()


@click.group()
@click.version_option(version="0.5.1")
def cli():
    """Victor AI Multimodal Assistant - Image, audio, and OCR processing."""
    pass


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--question", "-q", help="Specific question about the image")
@click.option("--detect-objects", is_flag=True, help="Detect objects in image")
@click.option("--output", "-o", type=click.Path(), help="Save output to file")
def analyze_image(
    image_path: str, question: Optional[str], detect_objects: bool, output: Optional[str]
):
    """Analyze an image with AI.

    Examples:
        victor-multimodal analyze-image photo.jpg
        victor-multimodal analyze-image photo.jpg --question "What colors are present?"
        victor-multimodal analyze-image photo.jpg --detect-objects
    """
    # Initialize
    settings = Settings()
    orchestrator = create_orchestrator(settings)

    console.print(
        Panel.fit(
            "[bold blue]Victor AI Image Analysis[/bold blue]\n" f"Analyzing: {image_path}",
            border_style="blue",
        )
    )

    # Process image
    processor = VisionProcessor(orchestrator)

    if detect_objects:
        console.print("\n[cyan]Detecting objects...[/cyan]")
        objects = processor.detect_objects(image_path)

        table = Table(title="Detected Objects")
        table.add_column("Object", style="cyan")
        table.add_column("Confidence", style="green")
        table.add_column("Location", style="yellow")

        for obj in objects:
            table.add_row(obj["label"], f"{obj['confidence']:.1%}", f"{obj['bbox']}")

        console.print(table)

    else:
        console.print("\n[cyan]Analyzing image...[/cyan]")

        if question:
            result = processor.query(image_path, question)
            console.print(f"\n[bold]Question:[/bold] {question}")
            console.print(f"[bold]Answer:[/bold] {result}")
        else:
            description = processor.describe(image_path)
            console.print(f"\n[bold]Description:[/bold]")
            console.print(description)

    # Save output if requested
    if output:
        _save_output(result if question else description, output)


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--language", "-l", default="en", help="Language code")
@click.option("--timestamps", is_flag=True, help="Include timestamps")
@click.option("--output", "-o", type=click.Path(), help="Save transcript to file")
def transcribe(audio_path: str, language: str, timestamps: bool, output: Optional[str]):
    """Transcribe audio to text.

    Examples:
        victor-multimodal transcribe recording.wav
        victor-multimodal transcribe interview.mp3 --timestamps
        victor-multimodal transcribe speech.wav --language es
    """
    settings = Settings()
    orchestrator = create_orchestrator(settings)

    console.print(
        Panel.fit(
            "[bold blue]Victor AI Audio Transcription[/bold blue]\n" f"Processing: {audio_path}",
            border_style="blue",
        )
    )

    console.print(f"\n[cyan]Transcribing audio (language: {language})...[/cyan]")

    # Process audio
    processor = AudioProcessor(orchestrator)
    transcript = processor.transcribe(audio_path, language=language)

    console.print(f"\n[bold]Transcript:[/bold]")
    console.print(transcript)

    # Save if requested
    if output:
        _save_output(transcript, output)


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--language", "-l", default="en", help="Language for OCR")
@click.option("--preprocess", is_flag=True, help="Preprocess image for better accuracy")
@click.option("--output", "-o", type=click.Path(), help="Save extracted text to file")
def ocr(image_path: str, language: str, preprocess: bool, output: Optional[str]):
    """Extract text from image using OCR.

    Examples:
        victor-multimodal ocr document.png
        victor-multimodal ocr scanned.jpg --preprocess
        victor-multimodal ocr invoice.png --language de
    """
    settings = Settings()
    orchestrator = create_orchestrator(settings)

    console.print(
        Panel.fit(
            "[bold blue]Victor AI OCR (Text Extraction)[/bold blue]\n" f"Processing: {image_path}",
            border_style="blue",
        )
    )

    console.print(f"\n[cyan]Extracting text...[/cyan]")

    # Process with OCR
    processor = VisionProcessor(orchestrator)
    text = processor.extract_text(image_path, language=language, preprocess=preprocess)

    console.print(f"\n[bold]Extracted Text:[/bold]")
    console.print(text)

    # Show statistics
    lines = text.split("\n")
    words = text.split()

    stats_table = Table(title="Extraction Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Lines", str(len(lines)))
    stats_table.add_row("Words", str(len(words)))
    stats_table.add_row("Characters", str(len(text)))

    console.print("\n")
    console.print(stats_table)

    # Save if requested
    if output:
        _save_output(text, output)


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--frames", "-f", type=int, default=5, help="Number of key frames to extract")
@click.option("--analyze-frames", is_flag=True, help="Analyze each extracted frame")
def analyze_video(video_path: str, frames: int, analyze_frames: bool):
    """Analyze video content.

    Examples:
        victor-multimodal analyze-video clip.mp4
        victor-multimodal analyze-video clip.mp4 --frames 10 --analyze-frames
    """
    settings = Settings()
    orchestrator = create_orchestrator(settings)

    console.print(
        Panel.fit(
            "[bold blue]Victor AI Video Analysis[/bold blue]\n" f"Processing: {video_path}",
            border_style="blue",
        )
    )

    console.print(f"\n[cyan]Extracting {frames} key frames...[/cyan]")

    # Process video
    processor = VisionProcessor(orchestrator)
    key_frames = processor.extract_key_frames(video_path, count=frames)

    console.print(f"\n[bold]Extracted {len(key_frames)} key frames[/bold]")

    if analyze_frames:
        console.print("\n[cyan]Analyzing frames...[/cyan]\n")

        for i, frame in enumerate(key_frames, 1):
            description = processor.describe_image_data(frame)
            console.print(f"[bold]Frame {i}:[/bold] {description}")

    # Generate video summary
    console.print("\n[cyan]Generating video summary...[/cyan]")
    summary = processor.summarize_video(video_path)

    console.print(f"\n[bold]Video Summary:[/bold]")
    console.print(summary)


@cli.command()
@click.option("--image", "-i", type=click.Path(exists=True), help="Image to analyze")
@click.option("--audio", "-a", type=click.Path(exists=True), help="Audio to transcribe")
@click.option("--text", "-t", help="Text context")
@click.argument("question")
def query(image: Optional[str], audio: Optional[str], text: Optional[str], question: str):
    """Cross-modal query combining multiple inputs.

    Examples:
        victor-multimodal query --image chart.png "What does this graph show?"
        victor-multimodal query --image photo.jpg --text "This is Paris" "Where was this taken?"
        victor-multimodal query --audio lecture.wav "What are the main topics?"
    """
    settings = Settings()
    orchestrator = create_orchestrator(settings)

    console.print(
        Panel.fit(
            "[bold blue]Victor AI Cross-Modal Query[/bold blue]\n" f"Question: {question}",
            border_style="blue",
        )
    )

    # Build multimodal context
    context = {}
    modalities = []

    if image:
        console.print(f"\n[cyan]Processing image: {image}[/cyan]")
        processor = VisionProcessor(orchestrator)
        context["image"] = image
        modalities.append("image")

    if audio:
        console.print(f"\n[cyan]Processing audio: {audio}[/cyan]")
        processor = AudioProcessor(orchestrator)
        transcript = processor.transcribe(audio)
        context["audio_transcript"] = transcript
        modalities.append("audio")

    if text:
        context["text"] = text
        modalities.append("text")

    # Execute cross-modal query
    console.print(f"\n[cyan]Analyzing {', '.join(modalities)}...[/cyan]")

    engine = MultimodalEngine(orchestrator)
    answer = engine.query(question, context)

    console.print(f"\n[bold]Answer:[/bold]")
    console.print(answer)


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output report file")
def batch_process(directory: str, output: Optional[str]):
    """Batch process all images in a directory.

    Examples:
        victor-multimodal batch-process photos/
        victor-multimodal batch-process docs/ --output report.json
    """
    settings = Settings()
    orchestrator = create_orchestrator(settings)

    console.print(
        Panel.fit(
            "[bold blue]Victor AI Batch Processing[/bold blue]\n" f"Directory: {directory}",
            border_style="blue",
        )
    )

    # Find all images
    dir_path = Path(directory)
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
        image_files.extend(dir_path.glob(f"*{ext}"))
        image_files.extend(dir_path.glob(f"*{ext.upper()}"))

    console.print(f"\n[cyan]Found {len(image_files)} images[/cyan]")

    # Process each image
    processor = VisionProcessor(orchestrator)
    results = []

    with console.status("[bold green]Processing images...") as status:
        for i, image_file in enumerate(image_files, 1):
            console.print(f"[{i}/{len(image_files)}] {image_file.name}")
            description = processor.describe(str(image_file))
            results.append({"file": str(image_file), "description": description})

    # Display summary
    console.print(f"\n[bold green]Processed {len(results)} images[/bold]")

    # Save report if requested
    if output:
        import json

        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Report saved to: {output}[/green]")


def _save_output(content: str, output_path: str):
    """Save content to file."""
    with open(output_path, "w") as f:
        f.write(content)
    console.print(f"\n[green]Output saved to: {output_path}[/green]")


if __name__ == "__main__":
    cli()
