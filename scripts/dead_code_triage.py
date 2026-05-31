#!/usr/bin/env python3
"""
Triage dead code candidates with DI-aware heuristics.
"""

from __future__ import annotations

import json
import ast
import tokenize
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPORT_DIR = ROOT / "docs" / "analysis_reports"
INPUT = REPORT_DIR / "06_dead_code.json"

FALSE_POSITIVE_PATH_HINTS = [
    "/vscode-victor/",
    "/protocol",
    "/protocols",
    "/registry",
    "/adapters",
    "/providers",
    "/plugins/",
    "/languages/plugins/",
    "/integrations/api/",
    "/ui/commands/",
    "/ui/slash/commands/",
    "/escape_hatches",
    "/workflows/",
]

HOOK_METHOD_NAMES = {
    "__post_init__",
    "_create",
    "_create_config",
    "_create_capabilities",
    "_create_tree_sitter_queries",
    "_get_base_capabilities",
    "_get_escape_hatches_module",
    "_register_default_templates",
    "_compute_reward",
    "_ensure_tables",
    "_save_to_db",
    "_do_start",
    "_do_stop",
    "_do_cleanup",
    "_get_sync_wrapper",
    "_run_command_in_service",
    "_get_capability_provider_module",
}


def build_self_cls_usage_counts() -> Counter:
    counts: Counter[str] = Counter()
    for path in ROOT.rglob("*.py"):
        try:
            with path.open("rb") as handle:
                prev = None
                prev_prev = None
                for tok in tokenize.tokenize(handle.readline):
                    if tok.type == tokenize.NAME:
                        if prev_prev and prev and prev.type == tokenize.OP and prev.string == ".":
                            if prev_prev.type == tokenize.NAME and prev_prev.string in {
                                "self",
                                "cls",
                            }:
                                counts[tok.string] += 1
                        prev_prev = prev
                        prev = tok
                    elif tok.type != tokenize.NL:
                        prev_prev = prev
                        prev = tok
        except Exception:
            continue
    return counts


def nested_function_lines(path: Path) -> set[int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    nested: set[int] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[ast.AST] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if self.stack and isinstance(self.stack[-1], (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested.add(node.lineno)
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if self.stack and isinstance(self.stack[-1], (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested.add(node.lineno)
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

    Visitor().visit(tree)
    return nested


def is_false_positive(entry: dict, reasons: list[str], nested_lines: set[int]) -> bool:
    name = entry["name"]
    file_path = entry.get("file") or ""
    lower = file_path.lower()

    if name in HOOK_METHOD_NAMES:
        reasons.append("hook_override")
    if name == "__post_init__":
        reasons.append("dataclass_hook")
    if any(hint in lower for hint in FALSE_POSITIVE_PATH_HINTS):
        reasons.append("dynamic_entrypoint")
    if entry.get("type") == "function" and entry.get("line") in nested_lines:
        reasons.append("nested_function")

    return bool(reasons)


def confidence_score(entry: dict, self_cls_counts: Counter) -> float:
    score = 0.4
    if entry.get("severity") == "medium":
        score -= 0.2
    if entry.get("is_private"):
        score += 0.1
    if self_cls_counts.get(entry["name"], 0) == 0:
        score += 0.2
    if "debug" in (entry.get("file") or ""):
        score += 0.05
    if "benchmark" in (entry.get("file") or ""):
        score += 0.05
    return max(0.0, min(0.9, round(score, 2)))


def main() -> int:
    if not INPUT.exists():
        print(f"Missing input: {INPUT}")
        return 1

    data = json.loads(INPUT.read_text())
    candidates = data.get("candidates", [])
    self_cls_counts = build_self_cls_usage_counts()

    false_positives = []
    prune_shortlist = []

    nested_cache: dict[str, set[int]] = {}

    for entry in candidates:
        file_path = Path(entry.get("file") or "")
        if file_path.suffix == ".py":
            nested_lines = nested_cache.get(str(file_path))
            if nested_lines is None:
                nested_lines = nested_function_lines(ROOT / file_path)
                nested_cache[str(file_path)] = nested_lines
        else:
            nested_lines = set()

        reasons: list[str] = []
        if is_false_positive(entry, reasons, nested_lines):
            false_positives.append(
                {
                    **entry,
                    "reasons": reasons,
                }
            )
            continue

        if (
            entry.get("severity") == "low"
            and entry.get("is_private")
            and not entry.get("is_dunder")
        ):
            if not (entry.get("file") or "").endswith(".py"):
                continue
            confidence = confidence_score(entry, self_cls_counts)
            if confidence >= 0.6:
                prune_shortlist.append(
                    {
                        **entry,
                        "confidence": confidence,
                    }
                )

    false_positives.sort(key=lambda x: (x.get("module", ""), x.get("name", "")))
    prune_shortlist.sort(key=lambda x: x["confidence"], reverse=True)

    (REPORT_DIR / "06_dead_code_false_positives.json").write_text(
        json.dumps(false_positives, indent=2)
    )
    (REPORT_DIR / "06_dead_code_prune_shortlist.json").write_text(
        json.dumps(prune_shortlist, indent=2)
    )

    summary = {
        "false_positive_count": len(false_positives),
        "prune_shortlist_count": len(prune_shortlist),
        "prune_shortlist_top": prune_shortlist[:20],
    }
    (REPORT_DIR / "07_dead_code_triage_summary.json").write_text(json.dumps(summary, indent=2))

    print("Dead code triage written:")
    print(f"- {REPORT_DIR / '06_dead_code_false_positives.json'}")
    print(f"- {REPORT_DIR / '06_dead_code_prune_shortlist.json'}")
    print(f"- {REPORT_DIR / '07_dead_code_triage_summary.json'}")
    print(f"False positives: {len(false_positives)}")
    print(f"Prune shortlist: {len(prune_shortlist)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
