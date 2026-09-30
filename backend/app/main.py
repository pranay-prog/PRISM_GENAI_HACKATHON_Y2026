"""FastAPI application entry point.

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .api.service import ApiService
from .api.websocket import ws_router
from .config import Settings, settings
from .runtime.services import build_services

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(cfg: Settings = settings, services=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.api = ApiService(services or build_services(cfg))
        logging.getLogger("app").info("system: %s", app.state.api.services.describe())
        yield
        if app.state.api.services.db is not None:
            app.state.api.services.db.close()

    app = FastAPI(title="Streaming Live RAG", version="1.0.0", lifespan=lifespan,
                  description="Real-time incremental retrieval, multi-intent decomposition and "
                              "state-preserving answer refinement.")
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in cfg.cors_origins.split(",")],
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)
    app.include_router(ws_router)
    return app


app = create_app()
