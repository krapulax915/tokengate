"""FastAPI application factory, lifespan and router mounting."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import ORJSONResponse

from dietgate import __version__
from dietgate.core.runtime import build_runtime
from dietgate.settings import Settings, load_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    from dietgate.core.runtime import start_background, stop_background

    await start_background(app.state.rt)
    yield
    await stop_background(app.state.rt)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(
        title="DietGate",
        version=__version__,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
        docs_url=None, redoc_url=None, openapi_url=None,
    )
    app.state.settings = settings
    app.state.rt = build_runtime(settings)

    from fastapi.responses import JSONResponse

    from dietgate.core.errors import DietGateError, error_body

    @app.exception_handler(DietGateError)
    async def dietgate_error_handler(request: Request, exc: DietGateError) -> JSONResponse:
        return JSONResponse(error_body(exc), status_code=exc.status)

    from dietgate.api import admin, chat, feedback, models

    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(feedback.router)
    app.include_router(admin.router)

    @app.get("/healthz")
    async def healthz() -> Response:
        return Response('{"status":"ok"}', media_type="application/json")

    @app.get("/readyz")
    async def readyz() -> Response:
        return Response('{"status":"ready"}', media_type="application/json")

    # Dashboard must be mounted LAST: a "/" mount matches everything.
    from fastapi.staticfiles import StaticFiles

    if settings.dashboard_dir.exists():
        app.mount("/", StaticFiles(directory=settings.dashboard_dir, html=True), name="dashboard")

    return app


app = create_app()
