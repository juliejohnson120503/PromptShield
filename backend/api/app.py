"""
PromptShield AI — Phase 6 FastAPI Application Factory
Provides REST API endpoints for detection, semantic masking, restoration, and telemetry.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import (
    health,
    analyze,
    sanitize,
    restore,
    process,
    policies,
    metrics,
)

logger = logging.getLogger("promptshield.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for PromptShield REST API."""
    logger.info("[PromptShield API] Starting up PromptShield REST service...")
    yield
    logger.info("[PromptShield API] Shutting down service.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="PromptShield AI REST Service",
        description=(
            "Context-aware security layer for LLM prompts: detection, context analysis, "
            "risk-based policy evaluation, semantic masking, and controlled restoration."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS Middleware ────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Permits browser extensions and local dashboards
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception Handlers ─────────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("[PromptShield API] Unhandled exception on %s: %s", request.url, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred while processing the prompt."},
        )

    # ── Include Routers ────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(sanitize.router)
    app.include_router(restore.router)
    app.include_router(process.router)
    app.include_router(policies.router)
    app.include_router(metrics.router)

    return app


app = create_app()
