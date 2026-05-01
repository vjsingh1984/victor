# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chat & Completions routes: /chat, /chat/stream, /completions."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, AsyncIterator

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse

from victor.integrations.api.fastapi_server import (
    ChatRequest,
    ChatResponse,
    CompletionRequest,
    CompletionResponse,
    _new_chat_request_id,
)
from victor.observability.request_correlation import request_correlation_id
from victor.runtime.chat_runtime import resolve_chat_runtime

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create chat / completions routes bound to *server*."""
    router = APIRouter()

    @router.post("/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat(request: ChatRequest, response: Response) -> ChatResponse:
        """Chat endpoint (non-streaming)."""
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        request_id = _new_chat_request_id()
        response.headers["X-Victor-Request-Id"] = request_id
        orchestrator = await server._get_orchestrator()
        chat_runtime = resolve_chat_runtime(orchestrator)
        with request_correlation_id(request_id):
            chat_result = await chat_runtime.chat(request.messages[-1].content)

        content = getattr(chat_result, "content", None) or ""
        tool_calls = getattr(chat_result, "tool_calls", None) or []

        return ChatResponse(role="assistant", content=content, tool_calls=tool_calls)

    @router.post("/chat/stream", tags=["Chat"])
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        """Streaming chat endpoint (Server-Sent Events)."""
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        request_id = _new_chat_request_id()

        async def event_generator() -> AsyncIterator[str]:
            try:
                orchestrator = await server._get_orchestrator()
                chat_runtime = resolve_chat_runtime(orchestrator)
                yield f'data: {json.dumps({"type": "request", "request_id": request_id})}\n\n'

                with request_correlation_id(request_id):
                    async for chunk in chat_runtime.stream_chat(request.messages[-1].content):
                        if hasattr(chunk, "content") or hasattr(chunk, "tool_calls"):
                            content = getattr(chunk, "content", "")
                            tool_calls = getattr(chunk, "tool_calls", None)
                            if content:
                                event = {
                                    "type": "content",
                                    "content": content,
                                    "request_id": request_id,
                                }
                            elif tool_calls:
                                event = {
                                    "type": "tool_call",
                                    "tool_call": tool_calls,
                                    "request_id": request_id,
                                }
                            else:
                                continue
                        else:
                            event = chunk
                            if isinstance(event, dict):
                                event.setdefault("request_id", request_id)

                        yield f"data: {json.dumps(event)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception("Stream chat error")
                error_event = {
                    "type": "error",
                    "message": str(e),
                    "request_id": request_id,
                }
                yield f"data: {json.dumps(error_event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Victor-Request-Id": request_id,
            },
        )

    @router.post("/completions", response_model=CompletionResponse, tags=["Completions"])
    async def completions(request: CompletionRequest) -> CompletionResponse:
        """Get fast code completions with FIM support."""
        start_time = time.perf_counter()

        if not request.prompt:
            return CompletionResponse(completions=[], latency_ms=0.0)

        try:
            orchestrator = await server._get_orchestrator()
            provider = orchestrator.provider_manager.current_provider

            file_info = f" ({request.language})" if request.language else ""
            if request.file:
                file_info = f" in {request.file}{file_info}"

            if request.suffix:
                completion_prompt = f"""Complete the code at <FILL>. Only output the completion, nothing else.

{request.context or ''}

{request.prompt}<FILL>{request.suffix}"""
            else:
                completion_prompt = f"""Complete this {request.language or 'code'}{file_info}. Only output the completion.

{request.context or ''}

{request.prompt}"""

            stop_sequences = request.stop_sequences or [
                "\n\n",
                "\ndef ",
                "\nclass ",
                "\nfunction ",
                "\n//",
                "\n#",
            ]

            messages = [{"role": "user", "content": completion_prompt}]
            response = await provider.chat(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=stop_sequences,
            )

            content = getattr(response, "content", "") or ""
            completion = content.strip()
            if "\n\nExplanation:" in completion:
                completion = completion.split("\n\nExplanation:")[0]
            if "\n\nNote:" in completion:
                completion = completion.split("\n\nNote:")[0]

            latency_ms = (time.perf_counter() - start_time) * 1000

            return CompletionResponse(
                completions=[completion] if completion else [],
                latency_ms=round(latency_ms, 2),
            )

        except Exception as e:
            logger.exception("Completions error")
            latency_ms = (time.perf_counter() - start_time) * 1000
            return CompletionResponse(
                completions=[],
                error=str(e),
                latency_ms=round(latency_ms, 2),
            )

    return router
