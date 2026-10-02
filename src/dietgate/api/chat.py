"""POST /v1/chat/completions - OpenAI-compatible, stream + non-stream."""
from __future__ import annotations

import orjson
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from dietgate.core.context import RequestContext
from dietgate.core.errors import BadRequestError, DietGateError, error_body
from dietgate.utils.ids import new_request_id, new_task_id
from dietgate.utils.timing import now_ns

router = APIRouter()


def _json_error(exc: DietGateError) -> Response:
    return Response(
        orjson.dumps(error_body(exc)), status_code=exc.status, media_type="application/json"
    )


def _validate(body: dict) -> None:
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise BadRequestError("field 'model' is required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequestError("field 'messages' must be a non-empty list")
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise BadRequestError("each message needs 'role' and 'content'")


def _build_ctx(request: Request, body: dict, rt) -> RequestContext:
    headers = request.headers
    temperature = float(body.get("temperature", 1.0) or 0.0)
    return RequestContext(
        request_id=new_request_id(),
        task_id=headers.get("x-task-id") or new_task_id(),
        api_key_id=headers.get("authorization", ""),
        stream=bool(body.get("stream", False)),
        requested_model=str(body.get("model", "auto")),
        header_task_type=headers.get("x-task-type"),
        header_policy=headers.get("x-policy"),
        cache_allowed=(temperature == 0.0) or (headers.get("x-cache", "").lower() == "allow"),
        messages=body.get("messages", []),
        tools=body.get("tools"),
        response_format=body.get("response_format"),
        max_tokens=body.get("max_tokens"),
        temperature=temperature,
        metadata=dict(body.get("metadata") or {}),
        t_recv=now_ns(),
    )


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    rt = request.app.state.rt
    raw = await request.body()
    if len(raw) > rt.app_cfg.request_size_limit_bytes:
        return _json_error(BadRequestError("request body exceeds size limit"))
    try:
        body = orjson.loads(raw) if raw else {}
    except orjson.JSONDecodeError:
        return _json_error(BadRequestError("request body is not valid JSON"))
    if not isinstance(body, dict):
        return _json_error(BadRequestError("request body must be a JSON object"))
    try:
        _validate(body)
    except BadRequestError as exc:
        return _json_error(exc)

    ctx = _build_ctx(request, body, rt)
    try:
        if ctx.stream:
            gen, headers = await rt.pipeline.handle_stream(ctx, body)
            return StreamingResponse(gen, media_type="text/event-stream", headers=headers)
        payload, headers = await rt.pipeline.handle_nonstream(ctx, body)
        return Response(payload, media_type="application/json", headers=headers)
    except DietGateError as exc:
        return _json_error(exc)
