"""Flag-gated in-process sandhi transport pilot (FEP-0020 Phase 4b / ADR-0047 D10 step 4).

Routes the WIRE layer of selected OpenAI-compat providers through the ``sandhi_gateway``
binding (Rust reqwest + decorator stack) while every keep-in-victor concern — prompt
assembly, tool-format translation, SSE→StreamChunk parsing, usage parsing, resilience,
logging — continues to run in the unmodified native adapter code. The sandhi variant of a
provider is the native class with only its two wire seams overridden.

Default OFF and byte-identical when off: :func:`resolve_transport_class` returns the native
class unless the provider is named in ``VICTOR_SANDHI_TRANSPORT_PROVIDERS`` (or the
programmatic override) AND the binding is importable AND a variant exists for that class.
It never raises.

Resilience layering (cross-design contract, pinned by the no-double-request parity test):
victor's ``ResilientProvider``/circuit breaker stay the only retry owner — the binding is
always called with ``max_retries=0``; sandhi owns the socket-level timeouts
(``timeout_secs``/``stream_idle_timeout_secs``). Sandhi's circuit-open maps to
``ProviderConnectionError`` and is never a demotion cause.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, AsyncIterator, Dict, FrozenSet, Iterable, Optional, Tuple, Type

from victor.providers.base import (
    BaseProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from victor.providers.deepseek_provider import DeepSeekProvider
from victor.providers.httpx_openai_compat import HttpxOpenAICompatProvider
from victor.providers.xai_provider import XAIProvider
from victor.providers.zai_provider import ZAIProvider

logger = logging.getLogger(__name__)

try:  # optional dependency (victor[sandhi])
    import sandhi_gateway as _sg  # type: ignore[import-untyped]
except Exception:  # pragma: no cover — absent without the extra
    _sg = None

_warned_binding_missing = False
_enabled_override: Optional[FrozenSet[str]] = None

_ENV_VAR = "VICTOR_SANDHI_TRANSPORT_PROVIDERS"


def sandhi_transport_available() -> bool:
    """True when the sandhi_gateway binding is importable."""
    return _sg is not None


def set_sandhi_transport_providers(names: Optional[Iterable[str]]) -> None:
    """Programmatic override of the enabled set (``None`` falls back to the env var).

    Called by settings-aware bootstrap code (e.g. the agent factory) so
    ``Settings.sandhi_transport_providers`` wins over the raw environment.
    """
    global _enabled_override
    _enabled_override = (
        None
        if names is None
        else frozenset(str(n).strip().lower() for n in names if str(n).strip())
    )


def configure_from_settings(settings: Any) -> None:
    """Bridge ``Settings.sandhi_transport_providers`` into the resolver (bootstrap seam).

    Called by the agent factory so YAML/profile-configured values reach provider creation
    (the resolver otherwise only sees the env var). An empty list means "not configured" —
    the env fallback stays active. Never raises.
    """
    try:
        names = getattr(settings, "sandhi_transport_providers", None) if settings else None
        set_sandhi_transport_providers(names or None)
    except Exception as exc:  # never let the pilot break bootstrap
        logger.debug("sandhi transport settings bridge failed (ignored): %s", exc)


def _enabled_providers() -> FrozenSet[str]:
    if _enabled_override is not None:
        return _enabled_override
    raw = os.environ.get(_ENV_VAR, "")
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def resolve_transport_class(
    name: str, native_cls: Type[BaseProvider], kwargs: Dict[str, Any]
) -> Type[BaseProvider]:
    """The class ``registry.create`` should instantiate: sandhi variant or native.

    Exception-free by contract: any internal failure logs at debug and returns the
    native class. Alias names resolve through the registry alias map so e.g. ``grok``
    enables the ``xai`` variant.
    """
    global _warned_binding_missing
    try:
        enabled = _enabled_providers()
        if not enabled:
            return native_cls
        candidates = {name.strip().lower()}
        try:
            from victor.providers.registry import ProviderRegistry

            candidates.add(ProviderRegistry.get_aliases().get(name, name).strip().lower())
        except Exception:  # registry unavailable — match on the raw name only
            pass
        if not (candidates & enabled):
            return native_cls
        if _sg is None:
            if not _warned_binding_missing:
                _warned_binding_missing = True
                logger.warning(
                    "sandhi transport enabled for %s but the sandhi-gateway binding is not "
                    "installed (pip install 'victor-ai[sandhi]'); using native transport",
                    sorted(enabled),
                )
            return native_cls
        if kwargs.get("auth_mode") == "oauth":
            return native_cls  # binding sends x-api-key only — OAuth stays native
        variant = _SANDHI_VARIANTS.get(native_cls)
        return variant if variant is not None else native_cls
    except Exception as exc:  # never let the pilot break provider creation
        logger.debug("sandhi transport resolution failed (native fallback): %s", exc)
        return native_cls


class SandhiTransportUnavailable(Exception):
    """Internal demotion marker: a binding-level failure (not an upstream-semantic error).

    Never escapes the mixin — it triggers one-way demotion to the native wire path.
    """


_STATUS_RE = re.compile(r"upstream status (\d{3})")


def map_sandhi_error(
    exc: BaseException, provider_name: str, timeout: float
) -> Optional[ProviderError]:
    """Map a binding error to victor's typed taxonomy; ``None`` means binding-internal.

    The binding raises ``RuntimeError`` with deterministic ``Display`` prefixes, and (post
    sandhi PR-S3) builtin ``TimeoutError`` for decorator timeouts.
    """
    if isinstance(exc, TimeoutError):
        return ProviderTimeoutError(
            f"sandhi transport timed out: {exc}", provider=provider_name, timeout=timeout
        )
    if not isinstance(exc, RuntimeError):
        return None
    msg = str(exc)
    if "rate limited (429)" in msg:
        return ProviderRateLimitError(msg, provider=provider_name, status_code=429)
    if "auth failed (401/403)" in msg:
        return ProviderAuthError(msg, provider=provider_name, status_code=401)
    if "timed out after" in msg:
        return ProviderTimeoutError(msg, provider=provider_name, timeout=timeout)
    if "circuit open" in msg:
        # Sandhi's shared breaker opened: upstream failing. Connection-class error;
        # explicitly NOT a demotion cause (the native path would be failing too).
        return ProviderConnectionError(msg, provider=provider_name)
    match = _STATUS_RE.search(msg)
    if match:
        status = int(match.group(1))
        if status == 429:
            return ProviderRateLimitError(msg, provider=provider_name, status_code=status)
        if status in (401, 403):
            return ProviderAuthError(msg, provider=provider_name, status_code=status)
        return ProviderError(msg, provider=provider_name, status_code=status)
    if "transport error:" in msg:
        lowered = msg.lower()
        if "timed out" in lowered or "timeout" in lowered:
            return ProviderTimeoutError(msg, provider=provider_name, timeout=timeout)
        return ProviderConnectionError(msg, provider=provider_name)
    return None  # unknown shape — binding-internal, demotion path


async def sse_lines(byte_items: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[str]:
    """Bridge the binding's raw-byte stream items to decoded SSE lines.

    Chunk boundaries are arbitrary (NOT line- or UTF-8-aligned): buffer at the byte level,
    split on ``\\n``, decode only complete lines, strip a trailing ``\\r``, and flush any
    unterminated tail at end of stream.
    """
    buffer = b""
    async for item in byte_items:
        data = item.get("data") or b""
        if not data:
            continue
        buffer += bytes(data)
        while True:
            newline_at = buffer.find(b"\n")
            if newline_at < 0:
                break
            line, buffer = buffer[:newline_at], buffer[newline_at + 1 :]
            yield line.rstrip(b"\r").decode("utf-8", errors="replace")
    if buffer:
        yield buffer.rstrip(b"\r").decode("utf-8", errors="replace")


class SandhiHttpxTransportMixin:
    """Overrides the two wire seams of :class:`HttpxOpenAICompatProvider` with sandhi calls.

    Everything else — payload build, SSE chunk parsing, tool translation, usage parsing,
    victor-side resilience — is inherited native code. MRO puts this mixin first so its
    seam overrides win and ``super()`` reaches the native implementation for fallback.

    Failure semantics: upstream-semantic errors raise victor's typed errors (no fallback
    re-execution — the native path would fail identically and re-executing would double-hit
    the provider). Binding-internal failures demote this instance one-way to the native
    wire path (transparently re-executing the current call), except mid-stream after the
    first yielded chunk, where replay would duplicate content.
    """

    _sandhi_demoted: bool = False

    # A small grace on top of sandhi's own socket-level timeout, as an FFI-hang backstop.
    _SANDHI_WAIT_GRACE_SECS = 5.0

    def _sandhi_slug(self) -> str:
        return str(getattr(self, "name", "openai"))

    def _sandhi_timeout(self) -> float:
        timeout = getattr(self, "timeout", None)
        try:
            return float(timeout) if timeout else 120.0
        except (TypeError, ValueError):
            return 120.0

    def _demote(self, cause: BaseException) -> None:
        if not self._sandhi_demoted:
            self._sandhi_demoted = True
            logger.warning(
                "sandhi transport demoted to native for %s (%s: %s)",
                self._sandhi_slug(),
                type(cause).__name__,
                cause,
            )

    async def _sandhi_complete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """One non-streaming wire call through the binding; returns the parsed body dict."""
        timeout = self._sandhi_timeout()
        try:
            out = await asyncio.wait_for(
                _sg.complete(  # type: ignore[union-attr]
                    self._sandhi_slug(),
                    str(payload.get("model", "")),
                    str(getattr(self, "base_url", "")),
                    str(getattr(self, "_api_key", None) or getattr(self, "api_key", "") or ""),
                    json.dumps(payload),
                    None,
                    timeout_secs=timeout,
                    max_retries=0,  # victor's ResilientProvider is the sole retry owner
                ),
                timeout=timeout + self._SANDHI_WAIT_GRACE_SECS,
            )
            return json.loads(out["body"])
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except asyncio.TimeoutError as exc:
            raise ProviderTimeoutError(
                f"sandhi transport FFI-level timeout after {timeout}s",
                provider=self._sandhi_slug(),
                timeout=timeout,
            ) from exc
        except BaseException as exc:  # noqa: BLE001 — pyo3 panics subclass BaseException
            mapped = map_sandhi_error(exc, self._sandhi_slug(), timeout)
            if mapped is not None:
                raise mapped from exc
            raise SandhiTransportUnavailable(str(exc)) from exc

    async def _complete_raw(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._sandhi_demoted or _sg is None:
            return await super()._complete_raw(payload)  # type: ignore[misc]
        try:
            return await self._sandhi_complete(payload)
        except SandhiTransportUnavailable as exc:
            self._demote(exc)
            return await super()._complete_raw(payload)  # type: ignore[misc]

    async def _open_stream_lines(self, payload: Dict[str, Any]) -> Tuple[Any, AsyncIterator[str]]:
        if self._sandhi_demoted or _sg is None:
            return await super()._open_stream_lines(payload)  # type: ignore[misc]
        timeout = self._sandhi_timeout()
        try:
            byte_iter = _sg.stream(  # type: ignore[union-attr]
                self._sandhi_slug(),
                str(payload.get("model", "")),
                str(getattr(self, "base_url", "")),
                str(getattr(self, "_api_key", None) or getattr(self, "api_key", "") or ""),
                json.dumps(payload),
                None,
                timeout_secs=timeout,
                stream_idle_timeout_secs=90.0,
                max_retries=0,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001
            mapped = map_sandhi_error(exc, self._sandhi_slug(), timeout)
            if mapped is not None:
                raise mapped from exc
            self._demote(exc)
            return await super()._open_stream_lines(payload)  # type: ignore[misc]

        provider_name = self._sandhi_slug()
        demote = self._demote
        native_open = super()._open_stream_lines  # type: ignore[misc]

        async def _lines() -> AsyncIterator[str]:
            yielded = False
            try:
                async for line in sse_lines(_iterate(byte_iter)):
                    yielded = True
                    yield line
            except (GeneratorExit, asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                # GeneratorExit = the consumer stopped early (e.g. on [DONE]) — clean close,
                # never a transport failure, never a demotion cause.
                raise
            except BaseException as exc:  # noqa: BLE001
                mapped = map_sandhi_error(exc, provider_name, timeout)
                if not yielded:
                    # Failure before any output: demote and replay natively.
                    demote(exc)
                    closer, native_lines = await native_open(payload)
                    try:
                        async for line in native_lines:
                            yield line
                    finally:
                        await _maybe_call(closer)
                    return
                # After first output: no replay (would duplicate content) — typed raise.
                demote(exc)
                raise (
                    mapped
                    if mapped is not None
                    else ProviderConnectionError(
                        f"sandhi stream failed mid-stream: {exc}", provider=provider_name
                    )
                ) from exc

        async def _closer() -> None:
            aclose = getattr(byte_iter, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:  # best-effort
                    pass

        return _closer, _lines()


async def _iterate(byte_iter: Any) -> AsyncIterator[Dict[str, Any]]:
    """Adapt the binding's async iterator; StopAsyncIteration ends cleanly."""
    async for item in byte_iter:
        yield item


async def _maybe_call(closer: Any) -> None:
    if closer is None:
        return
    try:
        result = closer()
        if asyncio.iscoroutine(result):
            await result
    except Exception:  # best-effort cleanup
        pass


class SandhiDeepSeekProvider(SandhiHttpxTransportMixin, DeepSeekProvider):
    """DeepSeek with the wire layer on sandhi transport (pilot)."""


class SandhiXAIProvider(SandhiHttpxTransportMixin, XAIProvider):
    """xAI/Grok with the wire layer on sandhi transport (pilot)."""


class SandhiZAIProvider(SandhiHttpxTransportMixin, ZAIProvider):
    """Z.AI with the wire layer on sandhi transport (pilot)."""


_SANDHI_VARIANTS: Dict[Type[BaseProvider], Type[BaseProvider]] = {
    DeepSeekProvider: SandhiDeepSeekProvider,
    XAIProvider: SandhiXAIProvider,
    ZAIProvider: SandhiZAIProvider,
}

__all__ = [
    "SandhiDeepSeekProvider",
    "SandhiHttpxTransportMixin",
    "SandhiTransportUnavailable",
    "SandhiXAIProvider",
    "SandhiZAIProvider",
    "map_sandhi_error",
    "configure_from_settings",
    "resolve_transport_class",
    "sandhi_transport_available",
    "set_sandhi_transport_providers",
    "sse_lines",
]
