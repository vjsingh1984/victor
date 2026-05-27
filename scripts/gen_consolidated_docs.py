#!/usr/bin/env python3
"""Generate consolidated Victor documentation.

Creates/updates the canonical doc suite with consistent Mermaid diagrams,
grounded module references, and consolidated content.
"""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def w(rel_path, content):
    full = os.path.join(BASE, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 