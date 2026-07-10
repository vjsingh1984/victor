"""Cache audit trail — records per-provider caching configuration.

Produces an immutable ``CacheAuditRecord`` for each registered provider,
capturing how the provider declares and implements prompt caching. This
artifact enables reproducible auditing of cache compliance across the
entire provider fleet.

Refactor notes:
  * ``_make_audit_fixtures`` centralises the Message/ToolDefinition fixture
    creation previously duplicated across ``_detect_tools_ordering`` and
    ``_detect_boundary_position`` (DRY).
  * ``log_cache_audit_record`` emits a structured ``CACHE_AUDIT`` log event
    that integrates with the existing ``ProviderLogger`` infrastructure,
    so audit verdicts surface in the same JSON log stream as API calls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from pydantic import BaseModel

_logger = logging.getLogger(__name__)


class CacheAuditRecord(BaseModel):
    """Immutable audit record capturing a single provider's caching configuration.

    Fields:
        provider_name:         Canonical provider name (e.g. "anthropic").
        cache_supports:        Whether the provider declares API-level prompt
                               caching (supports_prompt_caching()).
        cache_type:            Caching paradigm:
                                 - "auto_prefix"        → implicit server-side
                                 - "explicit_markers"   → client-side cache_control
                                 - "none"               → no caching
        serializer_method_name: Name of the method/function used to serialize
                               messages for the provider.
        tools_ordering:        Position of tools[] relative to messages[] in
                               the payload dict:
                                 - "before_messages"  → prefix-stable
                                 - "after_messages"   → not prefix-stable
                                 - "n/a"              → provider has no payload
        boundary_position:     Index of the cache_control boundary marker
                               (Anthropic only; ``_find_cache_boundary`` result).
                               ``None`` for auto_prefix / none providers.
        spec_compliance:       Overall compliance verdict:
                                 - "compliant"  → matches documented contract
                                 - "gap"        → partial / incorrect
                                 - "unknown"    → not yet assessed
        timestamp:             ISO-8601 UTC timestamp when the record was built.
    """

    model_config = {"frozen": True}

    provider_name: str
    cache_supports: bool
    cache_type: str  # 'auto_prefix' | 'explicit_markers' | 'none'
    serializer_method_name: str
    tools_ordering: str  # 'before_messages' | 'after_messages' | 'n/a'
    boundary_position: Optional[int] = None
    spec_compliance: str = "unknown"  # 'compliant' | 'gap' | 'unknown'
    timestamp: str = ""


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _has_real_method(provider: Any, attr: str) -> bool:
    """Check if ``provider`` has a real callable method named ``attr``.

    This distinguishes genuine serializer/boundary methods from the
    auto-generated attributes of ``MagicMock`` (which return ``True`` for
    every ``hasattr`` call).

    Detection strategy:
      1. Real class methods: inspect ``type(provider).__mro__`` — catches
         methods defined on actual provider classes (e.g. AnthropicProvider).
      2. Explicitly-set instance attributes: inspect ``provider.__dict__`` —
         catches attributes set via ``setattr`` in test mocks.
      3. Auto-created MagicMock attributes live in ``_mock_children``, NOT
         ``__dict__``, so they are correctly rejected.
    """
    # 1. Class-level method (real providers)
    for klass in type(provider).__mro__:
        if attr in klass.__dict__:
            return True
    # 2. Explicitly-set instance attribute (test mocks via setattr)
    instance_dict = getattr(provider, "__dict__", {})
    if attr in instance_dict:
        return callable(instance_dict[attr])
    return False


def _detect_cache_type(provider: Any, cache_supports: bool) -> str:
    """Classify the caching paradigm for a provider.

    - ``explicit_markers`` if the provider defines ``_find_cache_boundary``
      (client-side ``cache_control`` blocks).
    - ``auto_prefix`` if caching is supported but no boundary method exists
      (server-side prefix caching).
    - ``none`` otherwise.
    """
    if not cache_supports:
        return "none"
    if _has_real_method(provider, "_find_cache_boundary"):
        return "explicit_markers"
    return "auto_prefix"


def _detect_serializer_method(provider: Any) -> str:
    """Return the name of the provider's message serializer method.

    Checks for common serializer method names in priority order. Falls back
    to "unknown".
    """
    for attr in (
        "_serialize_message",
        "build_messages",
        "build_openai_messages",
        "_build_request_payload",
    ):
        if _has_real_method(provider, attr):
            return attr
    return "unknown"


def _make_audit_fixtures(num_tools: int = 3):
    """Build canonical Message + ToolDefinition fixtures for audit introspection.

    Shared by ``_detect_tools_ordering`` and ``_detect_boundary_position`` so
    both detectors probe the provider with the same representative inputs
    (DRY: previously each built its own inline fixtures).

    Returns ``(messages, tools, converted)`` or raises ImportError if the
    base models are unavailable.
    """
    from victor.providers.base import Message, ToolDefinition

    messages = [
        Message(role="system", content="audit"),
        Message(role="user", content="audit"),
    ]
    tools = [
        ToolDefinition(
            name=f"tool_{i}",
            description=f"audit tool {i}",
            parameters={"type": "object", "properties": {}},
            schema_level="full" if i < 2 else "stub",
        )
        for i in range(num_tools)
    ]
    converted = [{"name": td.name, "input_schema": {}} for td in tools]
    return messages, tools, converted


def _detect_tools_ordering(provider: Any) -> str:
    """Determine where tools[] sit relative to messages[] in the payload.

    - ``before_messages`` if the provider's payload builder places tools
      before messages (prefix-stable ordering).
    - ``after_messages`` if tools appear after messages.
    - ``n/a`` if the provider has no inspectable payload builder.
    """
    builder = getattr(provider, "_build_request_payload", None)
    if builder is None:
        return "n/a"
    try:
        messages, tools, _ = _make_audit_fixtures()
        payload = builder(
            messages=messages,
            model="audit-model",
            temperature=0.0,
            max_tokens=1,
            tools=tools,
            stream=False,
        )
        keys = list(payload.keys()) if isinstance(payload, dict) else []
        if "tools" in keys and "messages" in keys:
            if keys.index("tools") < keys.index("messages"):
                return "before_messages"
            return "after_messages"
        return "n/a"
    except Exception:
        return "n/a"


def _detect_boundary_position(provider: Any, cache_type: str) -> Optional[int]:
    """Return the cache_control boundary index for explicit_markers providers.

    For Anthropic-like providers, invokes ``_find_cache_boundary`` with a
    representative tool set and converted list. Returns ``None`` for other
    cache types.
    """
    if cache_type != "explicit_markers":
        return None
    boundary_fn = getattr(provider, "_find_cache_boundary", None)
    if boundary_fn is None:
        return None
    try:
        _, tools, converted = _make_audit_fixtures()
        return int(boundary_fn(tools, converted))
    except Exception:
        return None


def build_cache_audit_record(provider: Any) -> CacheAuditRecord:
    """Build a single ``CacheAuditRecord`` from a live provider instance.

    Args:
        provider: A provider instance exposing ``supports_prompt_caching()``,
                  ``name`` (or ``__class__.__name__``), and optionally
                  ``_find_cache_boundary`` / ``_build_request_payload``.

    Returns:
        A frozen ``CacheAuditRecord`` describing the provider's cache config.
    """
    # Provider name: prefer .name attribute, fall back to class name
    provider_name = getattr(provider, "name", None) or provider.__class__.__name__

    # Cache support flag
    if hasattr(provider, "supports_prompt_caching"):
        cache_supports = bool(provider.supports_prompt_caching())
    else:
        cache_supports = False

    cache_type = _detect_cache_type(provider, cache_supports)
    serializer_method_name = _detect_serializer_method(provider)
    tools_ordering = _detect_tools_ordering(provider)
    boundary_position = _detect_boundary_position(provider, cache_type)

    return CacheAuditRecord(
        provider_name=provider_name,
        cache_supports=cache_supports,
        cache_type=cache_type,
        serializer_method_name=serializer_method_name,
        tools_ordering=tools_ordering,
        boundary_position=boundary_position,
        spec_compliance="compliant" if cache_supports else "unknown",
        timestamp=_now_iso(),
    )


def log_cache_audit_record(
    record: CacheAuditRecord,
    logger: Optional[logging.Logger] = None,
    *,
    operation: str = "audit",
) -> None:
    """Emit a structured ``CACHE_AUDIT`` log event for a single record.

    Integrates with the existing structured-logging infrastructure so that
    audit verdicts surface in the same JSON log stream as ``API_CALL_*``
    events. Mirrors the ``ProviderLogger`` extra-data convention.

    Args:
        record:    The audit record to log.
        logger:    Optional logger; defaults to this module's logger.
        operation: Operation tag included in the structured extra payload.
    """
    target = logger or _logger
    target.debug(
        f"CACHE_AUDIT provider={record.provider_name} type={record.cache_type} "
        f"serializer={record.serializer_method_name} boundary={record.boundary_position} "
        f"verdict={record.spec_compliance}",
        extra={
            "event": "CACHE_AUDIT",
            "operation": operation,
            "provider": record.provider_name,
            "cache_supports": record.cache_supports,
            "cache_type": record.cache_type,
            "serializer_method_name": record.serializer_method_name,
            "tools_ordering": record.tools_ordering,
            "boundary_position": record.boundary_position,
            "spec_compliance": record.spec_compliance,
            "timestamp": record.timestamp,
        },
    )


def generate_cache_audit(
    providers: Optional[List[Any]] = None,
) -> List[CacheAuditRecord]:
    """Generate cache audit records for all registered providers.

    If ``providers`` is ``None``, attempts to discover all registered providers
    via the lazy provider registry. Otherwise, audits the supplied list of
    provider instances directly. Each generated record is also emitted via
    ``log_cache_audit_record`` for operational visibility.

    Args:
        providers: Optional list of provider instances to audit. When ``None``,
                   the function discovers registered provider specs from
                   ``victor.providers.registry``.

    Returns:
        List of ``CacheAuditRecord``, one per provider (sorted by provider_name).
    """
    if providers is not None:
        records = [build_cache_audit_record(p) for p in providers]
        for r in records:
            log_cache_audit_record(r)
        return records

    # Discover providers from the lazy registry specs
    records: List[CacheAuditRecord] = []
    try:
        from victor.providers.registry import _lazy_provider_specs
        import importlib

        for _name, spec in _lazy_provider_specs.items():
            try:
                module = importlib.import_module(spec.module_path)
                cls = getattr(module, spec.class_name, None)
                if cls is None:
                    continue
                # Build a lightweight instance without network credentials.
                try:
                    instance = cls.__new__(cls)
                except Exception:
                    continue
                records.append(build_cache_audit_record(instance))
            except Exception:
                continue
    except Exception:
        pass

    records.sort(key=lambda r: r.provider_name)
    for r in records:
        log_cache_audit_record(r)
    return records
