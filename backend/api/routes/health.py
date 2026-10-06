"""
Health and Status Routes
"""

from fastapi import APIRouter
from backend.api.schemas import HealthResponse
from backend.api.service import PromptShieldService

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Check system health, version, and detector subsystem statuses."""
    service = PromptShieldService()
    return service.get_health()


@router.get("/")
def root():
    """Welcome route providing service name and documentation link."""
    return {
        "service": "PromptShield AI REST Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
