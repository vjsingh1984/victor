#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Check Ollama model templates for native tool calling support.

This script queries Ollama's /api/show endpoint to examine model templates
and authoritatively determine if they support native tool calling.

Tool support is detected by looking for template patterns like:
- `{{ if .Tools }}` or `{{- if .Tools }}`
- `{{ range .Tools }}`
- `{{ .ToolCalls }}`

Usage:
    # Check all models on localhost
    python scripts/check_ollama_tool_support.py

    # Check specific Ollama server
    python scripts/check_ollama_tool_support.py --host http://192.168.1.20:11434

    # Check specific model
    python scripts/check_ollama_tool_support.py --model qwen2.5-coder:14b

    # Output as JSON
    python scripts/check_ollama_tool_support.py --json

    # Update model_capabilities.yaml
    python scripts/check_ollama_tool_support.py --update-config
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx
import yaml


@dataclass
class ModelToolSupport:
    """Tool support information for a model."""

    name: str
    supports_tools: bool
    template_has_tools: bool
    template_has_tool_calls: bool
    tool_response_format: str = "unknown"  # xml, json, or native
    detection_method: str = "template"
    template_snippet: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "supports_tools": self.supports_tools,
            "template_has_tools": self.template_has_tools,
            "template_has_tool_calls": self.template_has_tool_calls,
            "tool_response_format": self.tool_response_format,
            "detection_method": self.detection_method,
            "template_snippet": self.template_snippet[:200] if self.template_snippet else "",
            "error": self.error,
        }


# Patterns that indicate native tool support in Ollama templates
TOOL_SUPPORT_PATTERNS = [
    (r"\{\{-?\s*if\s+\.Tools\s*\}\}", "tools_conditional"),
    (r"\{\{-?\s*if\s+or\s+\.System\s+\.Tools\s*\}\}", "tools_or_system"),
    (r"\{\{-?\s*range\s+\.Tools\s*\}\}", "tools_range"),
    (r"\{\{-?\s*\.ToolCalls\s*\}\}", "tool_calls"),
    (r"\{\{\s*range\s+\$\.Tools\s*\}\}", "tools_range_dollar"),
]

# Patterns for detecting tool response format
TOOL_FORMAT_PATTERNS = [
    (r"<tool_call>", "xml"),
    (r"<function_call>", "xml"),
    (r'"name":\s*"?<function-name>"?', "json"),
    (r'"parameters":\s*\{', "json"),
    (r'{"name":', "json"),
]


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
        return {"error": str(e)}


def detect_tool_support(template: str) -> ModelToolSupport:
    """Analyze template to detect tool support patterns."""
    result = ModelToolSupport(
        name="",
        supports_tools=False,
        template_has_tools=False,
        template_has_tool_calls=False,
    )

    if not template:
        return result

    # Check for tool support patterns
    for pattern, pattern_name in TOOL_SUPPORT_PATTERNS:
        if re.search(pattern, template):
            if "tools" in pattern_name.lower():
                result.template_has_tools = True
            if "tool_calls" in pattern_name.lower():
                result.template_has_tool_calls = True
            result.supports_tools = True
            result.detection_method = f"template:{pattern_name}"

    # Detect tool response format
    for pattern, format_name in TOOL_FORMAT_PATTERNS:
        if re.search(pattern, template):
            result.tool_response_format = format_name
            break

    # Extract relevant snippet
    if result.supports_tools:
        match = re.search(
            r"\{\{-?\s*if\s+\.?Tools[^}]*\}\}.*?\{\{-?\s*end\s*\}\}", template, re.DOTALL
        )
        if match:
            result.template_snippet = match.group(0)[:300]

    return result


def check_model_tool_support(host: str, model: str) -> ModelToolSupport:
    """Check if a specific model supports tool calling."""
    info = get_model_info(host, model)

    if info is None or "error" in info:
        return ModelToolSupport(
            name=model,
            supports_tools=False,
            template_has_tools=False,
            template_has_tool_calls=False,
            error=info.get("error") if info else "Unknown error",
        )

    template = info.get("template", "")
    result = detect_tool_support(template)
    result.name = model

    return result


def check_all_models(host: str) -> List[ModelToolSupport]:
    """Check tool support for all available models."""
    models = get_ollama_models(host)
    results = []

    for model in models:
        result = check_model_tool_support(host, model)
        results.append(result)

    return results


def update_model_capabilities(results: List[ModelToolSupport], config_path: Path) -> None:
    """Update model_capabilities.yaml with detected tool support."""
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    models_section = config.setdefault("models", {})

    # Track what we updated
    updated = []

    for result in results:
        if result.supports_tools and not result.error:
            # Extract base model name (without tag)
            base_name = result.name.split(":")[0]
            pattern = f"{base_name}*"

            # Check if pattern already exists
            if pattern not in models_section:
                models_section[pattern] = {
                    "native_tool_calls": True,
                    "streaming_tool_calls": False,
                    "parallel_tool_calls": True,
                    "requires_strict_prompting": False,
                    "recommended_tool_budget": 15,
                }
                updated.append(pattern)

    if updated:
        # Write updated config
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Updated model_capabilities.yaml with {len(updated)} new patterns:")
        for p in updated:
            print(f"  - {p}")
    else:
        print("No new model patterns to add.")


def print_report(results: List[ModelToolSupport], json_output: bool = False) -> None:
    """Print a report of tool support detection results."""
    if json_output:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    # Group by support status
    supported = [r for r in results if r.supports_tools and not r.error]
    unsupported = [r for r in results if not r.supports_tools and not r.error]
    errors = [r for r in results if r.error]

    print("=" * 70)
    print("OLLAMA MODEL TOOL SUPPORT DETECTION REPORT")
    print("=" * 70)
    print()

    print(f"Total models checked: {len(results)}")
    print(f"Models with tool support: {len(supported)}")
    print(f"Models without tool support: {len(unsupported)}")
    print(f"Errors: {len(errors)}")
    print()

    print("-" * 70)
    print("MODELS WITH NATIVE TOOL SUPPORT (Template has {{ if .Tools }})")
    print("-" * 70)
    for r in sorted(supported, key=lambda x: x.name):
        format_str = f" [{r.tool_response_format}]" if r.tool_response_format != "unknown" else ""
        print(f"  ✓ {r.name}{format_str}")
    print()

    print("-" * 70)
    print("MODELS WITHOUT TOOL SUPPORT (Require fallback parsing)")
    print("-" * 70)
    for r in sorted(unsupported, key=lambda x: x.name):
        print(f"  ✗ {r.name}")
    print()

    if errors:
        print("-" * 70)
        print("ERRORS")
        print("-" * 70)
        for r in errors:
            print(f"  ! {r.name}: {r.error}")
        print()

    print("=" * 70)
    print("DIAGNOSIS GUIDE")
    print("=" * 70)
    print(
        """
To check if a model supports tools:

1. Run: ollama show <model_name>
   Look for '{{ if .Tools }}' in the TEMPLATE section

2. Or use this script:
   python scripts/check_ollama_tool_support.py --model <model_name>

3. If template has '{{ if .Tools }}' block:
   - Model supports native Ollama tool calling
   - Victor can pass tools directly to the API

4. If template lacks '{{ if .Tools }}' block:
   - Model does NOT support native tool calling
   - Victor will use fallback parsing (JSON/XML from content)
   - May work via prompt engineering, but less reliable

5. To enable tool support for a model without it:
   - Create a custom Modelfile with tool template
   - Use: ollama create <new_name> -f Modelfile
   - See: https://ollama.com/blog/tool-support
"""
    )


def main():
    parser = argparse.ArgumentParser(
        description="Check Ollama models for native tool calling support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama server URL (default: $OLLAMA_HOST or localhost:11434)",
    )
    parser.add_argument(
        "--model",
        help="Check specific model instead of all models",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update victor/config/model_capabilities.yaml with detected patterns",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(__file__).parent.parent / "victor" / "config" / "model_capabilities.yaml",
        help="Path to model_capabilities.yaml",
    )

    args = parser.parse_args()

    # Ensure host has protocol
    host = args.host
    if not host.startswith("http"):
        host = f"http://{host}"

    print(f"Checking Ollama server: {host}", file=sys.stderr)

    if args.model:
        results = [check_model_tool_support(host, args.model)]
    else:
        results = check_all_models(host)

    if not results:
        print("No models found or error connecting to Ollama server.", file=sys.stderr)
        sys.exit(1)

    print_report(results, json_output=args.json)

    if args.update_config:
        update_model_capabilities(results, args.config_path)


if __name__ == "__main__":
    main()
