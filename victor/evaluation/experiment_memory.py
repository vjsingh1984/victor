from __future__ import annotations

"""Structured experiment-memory records and a lightweight persistent store."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tokenize_text(value: Any) -> set[str]:
    text = _normalize_text(value)
    if not text:
        return set()
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if token}


@dataclass(frozen=True)
class ExperimentScope:
    """Stable scope for experiment-memory reuse."""

    benchmark: str
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_candidate_hash: Optional[str] = None
    section_name: Optional[str] = None
    dataset_name: Optional[str] = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "provider": self.provider,
            "model": self.model,
            "prompt_candidate_hash": self.prompt_candidate_hash,
            "section_name": self.section_name,
            "dataset_name": self.dataset_name,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentScope":
        return cls(
            benchmark=str(payload.get("benchmark") or "unknown"),
            provider=_normalize_text(payload.get("provider")),
            model=_normalize_text(payload.get("model")),
            prompt_candidate_hash=_normalize_text(payload.get("prompt_candidate_hash")),
            section_name=_normalize_text(payload.get("section_name")),
            dataset_name=_normalize_text(payload.get("dataset_name")),
            tags=tuple(str(tag) for tag in list(payload.get("tags") or []) if str(tag).strip()),
        )


@dataclass(frozen=True)
class ExperimentInsight:
    """Distilled reusable insight from a benchmark or runtime experiment."""

    kind: str
    summary: str
    confidence: float = 0.5
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "confidence": round(float(self.confidence), 4),
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentInsight":
        return cls(
            kind=str(payload.get("kind") or "note"),
            summary=str(payload.get("summary") or ""),
            confidence=float(payload.get("confidence", 0.5) or 0.5),
            evidence=dict(payload.get("evidence") or {}),
        )


@dataclass(frozen=True)
class ExperimentTaskSummary:
    """Compact task-level trace retained inside experiment memory."""

    task_id: str
    status: str
    completion_score: float
    failure_category: Optional[str] = None
    failure_taxonomy: Optional[str] = None
    topology: Optional[dict[str, Any]] = None
    optimization: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "completion_score": round(float(self.completion_score), 4),
            "failure_category": self.failure_category,
            "failure_taxonomy": self.failure_taxonomy,
            "topology": dict(self.topology or {}),
            "optimization": dict(self.optimization or {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentTaskSummary":
        return cls(
            task_id=str(payload.get("task_id") or ""),
            status=str(payload.get("status") or "unknown"),
            completion_score=float(payload.get("completion_score", 0.0) or 0.0),
            failure_category=_normalize_text(payload.get("failure_category")),
            failure_taxonomy=_normalize_text(payload.get("failure_taxonomy")),
            topology=dict(payload.get("topology") or {}) or None,
            optimization=dict(payload.get("optimization") or {}) or None,
        )


@dataclass(frozen=True)
class ExperimentMemoryRecord:
    """Persisted reusable summary of an evaluation experiment."""

    record_id: str
    created_at: float
    scope: ExperimentScope
    summary_metrics: dict[str, Any]
    task_summaries: list[ExperimentTaskSummary] = field(default_factory=list)
    insights: list[ExperimentInsight] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    source_result_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "created_at": round(float(self.created_at), 6),
            "scope": self.scope.to_dict(),
            "summary_metrics": dict(self.summary_metrics),
            "task_summaries": [task.to_dict() for task in self.task_summaries],
            "insights": [insight.to_dict() for insight in self.insights],
            "keywords": list(self.keywords),
            "source_result_path": self.source_result_path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentMemoryRecord":
        return cls(
            record_id=str(payload.get("record_id") or ""),
            created_at=float(payload.get("created_at", 0.0) or 0.0),
            scope=ExperimentScope.from_dict(dict(payload.get("scope") or {})),
            summary_metrics=dict(payload.get("summary_metrics") or {}),
            task_summaries=[
                ExperimentTaskSummary.from_dict(item)
                for item in list(payload.get("task_summaries") or [])
                if isinstance(item, Mapping)
            ],
            insights=[
                ExperimentInsight.from_dict(item)
                for item in list(payload.get("insights") or [])
                if isinstance(item, Mapping)
            ],
            keywords=[str(keyword) for keyword in list(payload.get("keywords") or []) if keyword],
            source_result_path=_normalize_text(payload.get("source_result_path")),
            metadata=dict(payload.get("metadata") or {}),
        )

    def searchable_text(self) -> str:
        parts = [
            self.record_id,
            self.scope.benchmark,
            self.scope.provider or "",
            self.scope.model or "",
            self.scope.prompt_candidate_hash or "",
            self.scope.section_name or "",
            self.scope.dataset_name or "",
            " ".join(self.scope.tags),
            " ".join(self.keywords),
            " ".join(insight.summary for insight in self.insights),
        ]
        return " ".join(part for part in parts if part)


class ExperimentMemoryStore:
    """Persistent JSON store for structured experiment-memory records."""

    def __init__(
        self,
        *,
        max_records: int = 500,
        persist_path: Optional[Path] = None,
    ) -> None:
        self._records: list[ExperimentMemoryRecord] = []
        self._max_records = max(1, max_records)
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load()

    def record(self, record: ExperimentMemoryRecord) -> None:
        replaced = False
        for index, existing in enumerate(self._records):
            if existing.record_id == record.record_id:
                self._records[index] = record
                replaced = True
                break
        if not replaced:
            self._records.append(record)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]
        self._save()

    def get_recent(self, limit: int = 10) -> list[ExperimentMemoryRecord]:
        if limit <= 0:
            return []
        return sorted(self._records, key=lambda record: record.created_at, reverse=True)[:limit]

    def search(
        self,
        query: str = "",
        *,
        benchmark: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        prompt_candidate_hash: Optional[str] = None,
        section_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[ExperimentMemoryRecord]:
        if limit <= 0:
            return []

        query_tokens = _tokenize_text(query)
        query_text = (query or "").strip().lower()
        matches: list[tuple[float, float, ExperimentMemoryRecord]] = []
        for record in self._records:
            if benchmark and record.scope.benchmark != benchmark:
                continue
            if provider and record.scope.provider != provider:
                continue
            if model and record.scope.model != model:
                continue
            if (
                prompt_candidate_hash
                and record.scope.prompt_candidate_hash != prompt_candidate_hash
            ):
                continue
            if section_name and record.scope.section_name != section_name:
                continue

            score = 0.0
            searchable_text = record.searchable_text().lower()
            if query_tokens:
                record_tokens = _tokenize_text(searchable_text)
                overlap = len(query_tokens & record_tokens)
                if overlap == 0:
                    continue
                score = overlap / max(1, len(query_tokens))
                if query_text and query_text in searchable_text:
                    score += 0.25
            matches.append((score, record.created_at, record))

        matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in matches[:limit]]

    def _save(self) -> None:
        if self._persist_path is None:
            return
        try:
            payload = {"records": [record.to_dict() for record in self._records]}
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.warning("Failed to persist experiment memory store: %s", exc)

    def _load(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            payload = json.loads(self._persist_path.read_text())
            records = []
            for item in list(payload.get("records") or []):
                if isinstance(item, Mapping):
                    records.append(ExperimentMemoryRecord.from_dict(item))
            self._records = records[-self._max_records :]
        except Exception as exc:
            logger.warning("Failed to load experiment memory store: %s", exc)

    def __len__(self) -> int:
        return len(self._records)


__all__ = [
    "ExperimentInsight",
    "ExperimentMemoryRecord",
    "ExperimentMemoryStore",
    "ExperimentScope",
    "ExperimentTaskSummary",
]
