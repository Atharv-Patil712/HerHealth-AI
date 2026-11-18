"""HerHealth AI FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routers import cycle, dashboard, education, health, questionnaires, screenings
from build_health_kb import ensure_knowledge_base

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_knowledge_base()
    yield


api = FastAPI(
    title="HerHealth AI",
    description=(
        "Women's health screening and wellness platform. Local prototype -- "
        "nothing here diagnoses any condition."
    ),
    version=settings.app_version,
    contact={"name": "HerHealth AI", "url": "https://example.com"},
    license_info={"name": "Private"},
    lifespan=lifespan,
)


@api.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception for %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred. Check backend logs for details."},
    )


api.include_router(health.router, prefix="/api")
api.include_router(dashboard.router, prefix="/api")
api.include_router(education.router, prefix="/api")
api.include_router(cycle.router, prefix="/api")
api.include_router(screenings.router)
api.include_router(questionnaires.router, prefix="/api")


@api.get("/api/root", include_in_schema=False)
async def root():
    return {"service": "HerHealth AI", "status": "ok", "docs": "/docs"}


app = CORSMiddleware(
    app=api,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)