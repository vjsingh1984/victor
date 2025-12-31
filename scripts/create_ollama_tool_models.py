#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Create Ollama model variants with tool support.

This script identifies models that are trained for tool support but lack
tool templates in Ollama, then creates "-tools" variants with proper templates.

The underlying model weights are shared - only the Modelfile template changes.

Usage:
    # Check which models need tool-enabled variants
    python scripts/create_ollama_tool_models.py --check-only

    # Create tool-enabled variants on localhost
    python scripts/create_ollama_tool_models.py

    # Create on specific Ollama server
    python scripts/create_ollama_tool_models.py --host http://192.168.1.20:11434

    # Create variants for specific models only
    python scripts/create_ollama_tool_models.py --models qwen3-coder:30b,mistral:7b
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Models known to have tool calling capability from training
# These can do tools when served by LMStudio/vLLM but may lack
# Ollama template support
TOOL_CAPABLE_MODELS = {
    # Mistral family - trained with tool support but no Ollama template
    "mistral": {
        "chat_format": "chatml",  # Uses ChatML format
        "tool_format": "json",
    },
    "mixtral": {
        "chat_format": "chatml",
        "tool_format": "json",
    },
    "ministral": {
        "chat_format": "chatml",
        "tool_format": "json",
    },
    "devstral": {
        "chat_format": "chatml",
        "tool_format": "json",
    },
    # DeepSeek family - trained with tool support
    "deepseek-coder": {
        "chat_format": "deepseek",
        "tool_format": "json",
    },
    "deepseek-coder-v2": {
        "chat_format": "deepseek",
        "tool_format": "json",
    },
    "deepseek-r1": {
        "chat_format": "deepseek",
        "tool_format": "json",
    },
    # Qwen3-coder - has tool training but missing Ollama template
    "qwen3-coder": {
        "chat_format": "chatml",
        "tool_format": "xml",  # Qwen uses XML format
        "thinking_mode": True,
    },
    # Gemma3 - Google's model has tool capability
    "gemma3": {
        "chat_format": "gemma",
        "tool_format": "json",
    },
    # Phi models - some tool capability
    "phi": {
        "chat_format": "chatml",
        "tool_format": "json",
    },
    # Yi models
    "yi": {
        "chat_format": "chatml",
        "tool_format": "json",
    },
}

# Templates for different chat formats
TOOL_TEMPLATES = {
    "chatml": """{{- if .System }}
<|im_start|>system
{{ .System }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ . }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}
<|im_end|>
{{- end }}
{{- range .Messages }}
<|im_start|>{{ .Role }}
{{ .Content }}
{{- if .ToolCalls }}
{{- range .ToolCalls }}
<tool_call>
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
</tool_call>
{{- end }}
{{- end }}
<|im_end|>
{{- end }}
<|im_start|>assistant
""",
    "chatml_xml": """{{- if .System }}
<|im_start|>system
{{ .System }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{{ . }}
{{- end }}
</tools>

For each function call, return the call within <tool_call></tool_call> XML tags:
<tool_call>
<name>function-name</name>
<arguments>{"arg1": "value1"}</arguments>
</tool_call>
{{- end }}
<|im_end|>
{{- end }}
{{- range .Messages }}
<|im_start|>{{ .Role }}
{{ .Content }}
{{- if .ToolCalls }}
{{- range .ToolCalls }}
<tool_call>
<name>{{ .Function.Name }}</name>
<arguments>{{ .Function.Arguments }}</arguments>
</tool_call>
{{- end }}
{{- end }}
<|im_end|>
{{- end }}
<|im_start|>assistant
""",
    "deepseek": """{{- if .System }}
<|begin▁of▁sentence|>{{ .System }}
{{- if .Tools }}

### Available Tools

You have access to the following tools:
{{- range .Tools }}
{{ . }}
{{- end }}

To call a tool, respond with JSON:
{"tool_call": {"name": "function_name", "arguments": {...}}}
{{- end }}
{{- end }}
{{- range .Messages }}
{{- if eq .Role "user" }}
User: {{ .Content }}
{{- else if eq .Role "assistant" }}
Assistant: {{ .Content }}
{{- if .ToolCalls }}
{{- range .ToolCalls }}
{"tool_call": {"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}}
{{- end }}
{{- end }}
{{- end }}
{{- end }}
Assistant:""",
    "gemma": """{{- if .System }}
<start_of_turn>system
{{ .System }}
{{- if .Tools }}

# Tools

You have access to the following tools:
{{- range .Tools }}
{{ . }}
{{- end }}

To use a tool, output JSON: {"name": "tool_name", "arguments": {...}}
{{- end }}
<end_of_turn>
{{- end }}
{{- range .Messages }}
<start_of_turn>{{ .Role }}
{{ .Content }}
{{- if .ToolCalls }}
{{- range .ToolCalls }}
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}
{{- end }}
<end_of_turn>
{{- end }}
<start_of_turn>model
""",
}


@dataclass
class ModelInfo:
    """Information about an Ollama model."""

    name: str
    has_tools: bool
    template: str
    base_name: str  # Without tag
    tag: str
    tool_capable: bool = False
    needs_variant: bool = False


def get_ollama_models(host: str) -> List[str]:
    """Get list of available models from Ollama server."""
    try:
        response = httpx.get(f"{host}/api/tags", timeout=30)
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"Error fetching models from {host}: {e}", file=sys.stderr)
        return []


def get_model_info(host: str, model: str) -> Optional[Dict[str, Any]]:
    """Get model information including template from Ollama."""
    try:
        response = httpx.post(
            f"{host}/api/show",
            json={"name": model},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching model info for {model}: {e}", file=sys.stderr)
        return None


def has_tool_support(template: str) -> bool:
    """Check if template has tool support."""
    if not template:
        return False
    patterns = [
        r"\{\{-?\s*if\s+\.Tools\s*\}\}",
        r"\{\{-?\s*range\s+\.Tools\s*\}\}",
    ]
    return any(re.search(p, template) for p in patterns)


def get_base_name(model_name: str) -> tuple:
    """Extract base name and tag from model name."""
    if ":" in model_name:
        parts = model_name.split(":", 1)
        return parts[0], parts[1]
    return model_name, "latest"


def is_tool_capable(base_name: str) -> Optional[Dict]:
    """Check if model family is known to have tool capability."""
    for family, config in TOOL_CAPABLE_MODELS.items():
        if base_name.startswith(family):
            return config
    return None


def analyze_models(host: str, specific_models: Optional[List[str]] = None) -> List[ModelInfo]:
    """Analyze models and identify which need tool-enabled variants."""
    if specific_models:
        models = specific_models
    else:
        models = get_ollama_models(host)

    results = []
    for model in models:
        info = get_model_info(host, model)
        if not info:
            continue

        template = info.get("template", "")
        base_name, tag = get_base_name(model)
        tool_support = has_tool_support(template)
        tool_config = is_tool_capable(base_name)

        model_info = ModelInfo(
            name=model,
            has_tools=tool_support,
            template=template,
            base_name=base_name,
            tag=tag,
            tool_capable=tool_config is not None,
            needs_variant=tool_config is not None and not tool_support,
        )
        results.append(model_info)

    return results


def generate_modelfile(base_model: str, tool_config: Dict) -> str:
    """Generate a Modelfile with tool support for a model."""
    chat_format = tool_config.get("chat_format", "chatml")
    tool_format = tool_config.get("tool_format", "json")

    # Select template based on format
    if tool_format == "xml" and chat_format == "chatml":
        template = TOOL_TEMPLATES["chatml_xml"]
    else:
        template = TOOL_TEMPLATES.get(chat_format, TOOL_TEMPLATES["chatml"])

    modelfile = f'''FROM {base_model}

TEMPLATE """
{template}
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
'''

    return modelfile


def create_tool_variant(
    host: str,
    model: str,
    tool_config: Dict,
    dry_run: bool = False,
) -> bool:
    """Create a tool-enabled variant of a model."""
    base_name, tag = get_base_name(model)

    # Create variant name with -tools suffix
    variant_name = f"{base_name}-tools:{tag}"

    print(f"Creating tool-enabled variant: {variant_name}")

    modelfile_content = generate_modelfile(model, tool_config)

    if dry_run:
        print(f"  Would create Modelfile:\n{modelfile_content[:500]}...")
        return True

    # Write Modelfile to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False) as f:
        f.write(modelfile_content)
        modelfile_path = f.name

    try:
        # Use ollama CLI to create the model
        env = os.environ.copy()
        env["OLLAMA_HOST"] = host

        result = subprocess.run(
            ["ollama", "create", variant_name, "-f", modelfile_path],
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode == 0:
            print(f"  Created: {variant_name}")
            return True
        else:
            print(f"  Error: {result.stderr}", file=sys.stderr)
            return False

    finally:
        os.unlink(modelfile_path)


def print_analysis(models: List[ModelInfo]) -> None:
    """Print analysis of models."""
    print("=" * 70)
    print("OLLAMA MODEL TOOL SUPPORT ANALYSIS")
    print("=" * 70)

    # Group models
    has_tools = [m for m in models if m.has_tools]
    needs_variant = [m for m in models if m.needs_variant]
    not_capable = [m for m in models if not m.tool_capable and not m.has_tools]

    print(f"\nModels WITH native tool support: {len(has_tools)}")
    for m in sorted(has_tools, key=lambda x: x.name):
        print(f"  ✓ {m.name}")

    print(f"\nModels NEEDING -tools variant: {len(needs_variant)}")
    for m in sorted(needs_variant, key=lambda x: x.name):
        print(f"  → {m.name} (tool-capable but no template)")

    print(f"\nModels without tool capability: {len(not_capable)}")
    for m in sorted(not_capable, key=lambda x: x.name):
        print(f"  ✗ {m.name}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Create Ollama model variants with tool support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama server URL (default: $OLLAMA_HOST or localhost:11434)",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated list of specific models to process",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check which models need variants, don't create them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually creating models",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create variants even if -tools version already exists",
    )

    args = parser.parse_args()

    # Ensure host has protocol
    host = args.host
    if not host.startswith("http"):
        host = f"http://{host}"

    print(f"Analyzing Ollama server: {host}\n")

    # Parse specific models if provided
    specific_models = None
    if args.models:
        specific_models = [m.strip() for m in args.models.split(",")]

    # Analyze models
    models = analyze_models(host, specific_models)

    if not models:
        print("No models found or error connecting to Ollama server.")
        sys.exit(1)

    print_analysis(models)

    if args.check_only:
        return

    # Create variants for models that need them
    needs_variant = [m for m in models if m.needs_variant]

    if not needs_variant:
        print("No models need tool-enabled variants.")
        return

    print("=" * 70)
    print("CREATING TOOL-ENABLED VARIANTS")
    print("=" * 70)

    created = 0
    failed = 0

    for model_info in needs_variant:
        tool_config = is_tool_capable(model_info.base_name)
        if not tool_config:
            continue

        # Check if variant already exists
        variant_name = f"{model_info.base_name}-tools:{model_info.tag}"
        existing_models = get_ollama_models(host)
        if variant_name in existing_models and not args.force:
            print(f"Skipping {model_info.name}: variant {variant_name} already exists")
            continue

        if create_tool_variant(host, model_info.name, tool_config, args.dry_run):
            created += 1
        else:
            failed += 1

    print()
    print(f"Created: {created}, Failed: {failed}")

    if created > 0 and not args.dry_run:
        print("\nTo use the new tool-enabled models:")
        for model_info in needs_variant:
            variant_name = f"{model_info.base_name}-tools:{model_info.tag}"
            print(f"  victor chat --model {variant_name}")


if __name__ == "__main__":
    main()
