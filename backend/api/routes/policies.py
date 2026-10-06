"""
Privacy Policies Route
"""

from fastapi import APIRouter
from backend.api.schemas import PolicyListResponse
from backend.api.service import PromptShieldService

router = APIRouter(tags=["Policies"])


@router.get("/policies", response_model=PolicyListResponse)
def get_policies():
    """Retrieve supported security levels, entity types, and task classifications."""
    service = PromptShieldService()
    return service.get_policies()
