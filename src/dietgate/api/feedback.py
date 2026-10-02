"""POST /v1/feedback - outcome reporting (spec sections 6.5, 10)."""
from __future__ import annotations

import time

import orjson
from fastapi import APIRouter, Request, Response

from dietgate.core.errors import BadRequestError, NotFoundError

router = APIRouter()

_SOURCES = {"client", "checker", "shadow_judge"}


@router.post("/v1/feedback")
async def feedback(request: Request) -> Response:
    rt = request.app.state.rt
    raw = await request.body()
    if len(raw) > rt.app_cfg.request_size_limit_bytes:
        raise BadRequestError("request body exceeds size limit")
    try:
        body = orjson.loads(raw) if raw else {}
    except orjson.JSONDecodeError:
        raise BadRequestError("request body is not valid JSON")

    task_id = body.get("task_id")
    success = body.get("success")
    score = body.get("score")
    source = body.get("source", "client")
    if not isinstance(task_id, str) or not task_id:
        raise BadRequestError("field 'task_id' is required")
    if not isinstance(success, bool):
        raise BadRequestError("field 'success' must be a boolean")
    if score is not None and not isinstance(score, (int, float)):
        raise BadRequestError("field 'score' must be a number between 0 and 1")
    if source not in _SOURCES:
        raise BadRequestError(f"field 'source' must be one of {sorted(_SOURCES)}")

    learner = rt.learner
    if learner is None or not learner.has_task(task_id):
        raise NotFoundError(f"unknown task_id: {task_id}")

    accepted = learner.submit(
        task_id=task_id,
        success=success,
        score=float(score) if score is not None else None,
        source=source,
        ts=time.time(),
    )
    if not accepted:
        raise BadRequestError("feedback queue full; try again")
    return Response(
        orjson.dumps({"status": "ok", "task_id": task_id}), media_type="application/json"
    )
