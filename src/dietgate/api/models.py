"""GET /v1/models - configured models plus 'auto'."""
from __future__ import annotations

from fastapi import APIRouter, Request, Response

import orjson

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request) -> Response:
    rt = request.app.state.rt
    ids = ["auto", *rt.registry.model_ids()]
    data = [{"id": mid, "object": "model", "owned_by": "dietgate"} for mid in ids]
    return Response(orjson.dumps({"object": "list", "data": data}), media_type="application/json")
