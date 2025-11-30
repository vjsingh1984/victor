#!/usr/bin/env python3
"""
Quick connectivity check for LMStudio tiered endpoints.

Prints the first reachable URL (from settings.lmstudio_base_urls) and lists
available models from /v1/models. Useful for verifying which host will be
picked as the default provider.
"""

from __future__ import annotations

import sys
from typing import List

import httpx

from victor.config.settings import Settings


def fetch_models(url: str) -> List[str]:
    """Return model IDs from an LMStudio /v1/models response."""
    try:
        resp = httpx.get(f"{url.rstrip('/')}/v1/models", timeout=1.5)
        if resp.status_code != 200:
            return []
        data = resp.json() or {}
        models = []
        for item in data.get("data", []):
            if isinstance(item, dict):
                model_id = item.get("id") or item.get("model")
                if model_id:
                    models.append(model_id)
        return models
    except Exception:
        return []


def main() -> int:
    settings = Settings()
    urls = settings.lmstudio_base_urls

    settings_instance = Settings()
    detected_vram = settings_instance._detect_vram_gb()
    max_vram = getattr(settings_instance, "lmstudio_max_vram_gb", None)
    effective_vram = None
    if detected_vram and max_vram:
        effective_vram = min(detected_vram, max_vram)
    else:
        effective_vram = detected_vram or max_vram

    if detected_vram:
        print(f"Detected GPU VRAM (best-effort): ~{detected_vram:.1f} GB")
    else:
        print("GPU VRAM detection unavailable (best-effort).")
    if max_vram:
        print(f"Config VRAM cap: {max_vram:.1f} GB")
    if effective_vram:
        print(f"Effective VRAM budget: {effective_vram:.1f} GB\n")
    else:
        print("Proceeding without VRAM hints.\n")

    print("Checking LMStudio endpoints (in order):")
    for url in urls:
        models = fetch_models(url)
        reachable = "reachable" if models else "unreachable"
        print(f"- {url}: {reachable}")
        if models:
            print(f"  models: {', '.join(models)}")
            if effective_vram:
                viable = []
                for mid in models:
                    req = Settings._estimate_model_vram_gb(mid)
                    if req and req <= effective_vram:
                        viable.append((req, "coder" in mid.lower(), mid))
                if viable:
                    # Recommend largest option within VRAM; coder preferred
                    viable.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
                    recommendation = viable[0]
                    print(
                        f"  recommended (fits VRAM, max capability): {recommendation[2]} (~{recommendation[0]} GB)"
                    )

            print(f"\nFirst reachable endpoint: {url}")
            return 0

    print("\nNo LMStudio endpoint reachable. Is the server running on any of these?")
    return 1


if __name__ == "__main__":
    sys.exit(main())
